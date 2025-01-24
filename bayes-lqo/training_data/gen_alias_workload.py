import json
import os
import pdb
import time
from itertools import combinations, pairwise
from pprint import pformat
from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import sqlglot
from peewee import JOIN, AsIs, IntegrityError, chunked
from sqlglot import Expression, condition, select  # type: ignore

from logger.log import l
from workload.schema import (
    build_alias_join_graph,
    build_join_graph,
    build_join_graph_from_queries,
)
from workload.workloads import get_workload_set

from .planner import explain_batch
from .storage import AliasWorkloadPlan, AliasWorkloadQuery, PlanType, QueryType


def join_edges(
    join_graph: nx.Graph, join_tables: list[tuple[str, int]]
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """
    Find all join edges (((table1, alias1), col1), ((table2, alias2), col2)) given the `join_graph` and the list of
    `join_tables`
    """
    edges = []
    for t1, t2 in combinations(join_tables, 2):
        edge = join_graph.get_edge_data(t1, t2)
        if edge is not None:
            edges.append(
                (
                    (edge["table"], edge["attr"]),
                    (edge["referenced_table"], edge["referenced_attr"]),
                )
            )
    return edges


def make_join_expr(
    tables: list[tuple[str, int]],
    edges: list[tuple[tuple[tuple[str, int], str], tuple[tuple[str, int], str]]],
) -> Expression:
    """Make a SQLGlot equi-join query given the `tables` and `edges` of the join."""
    first_table, first_alias = tables[0]
    expr = select("*").from_(
        sqlglot.expressions.alias_(first_table, f"{first_table}{first_alias}")
    )

    for table, alias in tables[1:]:
        expr = expr.join(sqlglot.expressions.alias_(table, f"{table}{alias}"))

    join_preds = None
    for ((t1, a1), c1), ((t2, a2), c2) in edges:
        predicate = f"{t1}{a1}.{c1}={t2}{a2}.{c2}"
        if join_preds is None:
            join_preds = condition(predicate)
        else:
            join_preds = join_preds.and_(predicate)
    expr = expr.where(join_preds)
    return expr


def gen_alias_join_queries(join_graph: nx.Graph, schema_name: str):
    queries_with_alias_count: dict[int, int] = {}
    unique_queries = set()
    queries_of_length: dict[int, int] = {}
    to_insert = []
    for path_length in range(2, 30):
        l.info(f"Path length: {path_length}")
        paths = nx.generate_random_paths(join_graph, 10_000, path_length)
        for path in paths:
            # Discard paths that revisit the same table, alias
            # path = [(table, alias_num) for table, alias_num in path]
            # unique_nodes = set(path)
            # if len(unique_nodes) != len(path):
            #     continue

            # Normalize aliases
            normalized_path = []
            current_alias: dict[str, int] = {table: 1 for table, _ in path}
            alias_rewrites: dict[tuple[str, int], int] = {}
            for table, alias_num in path:
                if not (table, alias_num) in alias_rewrites:
                    alias_rewrites[(table, alias_num)] = current_alias[table]
                    current_alias[table] += 1
                normalized_path.append((table, alias_rewrites[(table, alias_num)]))

            # Get join edges from path
            seen_edges = set()
            edges: list[
                tuple[tuple[tuple[str, int], str], tuple[tuple[str, int], str]]
            ] = []
            for table_alias_1, table_alias_2 in pairwise(normalized_path):
                # We don't need to add the same join edge twice
                # This happens if the path loops back on itself
                if (table_alias_1, table_alias_2) in seen_edges:
                    continue
                seen_edges.add((table_alias_1, table_alias_2))
                seen_edges.add((table_alias_2, table_alias_1))

                edge_data = join_graph.get_edge_data(table_alias_1, table_alias_2)
                table1, alias1 = table_alias_1
                table2, alias2 = table_alias_2

                if edge_data["table"] == table1:
                    table1_attr = edge_data["attr"]
                    table2_attr = edge_data["referenced_attr"]
                elif edge_data["referenced_table"] == table1:
                    table1_attr = edge_data["referenced_attr"]
                    table2_attr = edge_data["attr"]
                else:
                    raise ValueError("Edge data is not consistent")
                edges.append(
                    (
                        ((table1, alias1), table1_attr),
                        ((table2, alias2), table2_attr),
                    )
                )

            # Remove duplicate nodes from path
            seen_nodes = set()
            deduped_path = []
            for node in normalized_path:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                deduped_path.append(node)

            join_expr = make_join_expr(deduped_path, edges)

            num_joins = len(deduped_path)
            if not tuple(deduped_path) in unique_queries:
                if not num_joins in queries_of_length:
                    queries_of_length[num_joins] = 0
                queries_of_length[num_joins] += 1

            unique_queries.add(tuple(deduped_path))
            unique_queries.add(tuple(reversed(deduped_path)))

            num_aliases = max(alias_rewrites.values())
            if not num_aliases in queries_with_alias_count:
                queries_with_alias_count[num_aliases] = 0
            queries_with_alias_count[num_aliases] += 1

            to_insert.append(
                {
                    "join_key": deduped_path,
                    "num_joins": num_joins,
                    "num_aliases": num_aliases,
                    "sql_text": join_expr.sql("postgres"),
                    "query_type": QueryType.ALIAS_JOIN,
                    "schema": schema_name,
                }
            )

        inserted = (
            AliasWorkloadQuery.insert_many(to_insert)
            .on_conflict_ignore()
            .as_rowcount()
            .execute()
        )
        to_insert = []

        total_queries = (
            AliasWorkloadQuery.select()
            .where(AliasWorkloadQuery.schema == schema_name)
            .count()
        )
        l.info(f"{inserted} new queries, {total_queries} total")

    l.info(f"Unique queries by length: {pformat(queries_of_length)}")
    l.info(f"Total unique queries: {len(unique_queries) / 2}")
    l.info(queries_with_alias_count)


def generate_plans(plan_type: PlanType, schema: str):
    PlansOfType = AliasWorkloadPlan.alias()
    plans_of_type = (
        PlansOfType.select()
        .where((PlansOfType.plan_type == plan_type) & (PlansOfType.schema == schema))
        .cte("plans_of_type")
    )

    unplanned_queries = (
        AliasWorkloadQuery.select(
            AliasWorkloadQuery.query_id, AliasWorkloadQuery.sql_text
        )
        .join(
            plans_of_type,
            JOIN.LEFT_OUTER,
            on=(AliasWorkloadQuery.query_id == plans_of_type.c.query_id),
        )
        .where(
            (AliasWorkloadQuery.schema == schema) & (plans_of_type.c.plan_json == None)
        )
        .with_cte(plans_of_type)
    )
    l.info(f"Total unplanned {plan_type} queries: {len(unplanned_queries)}")

    inserted = 0
    for batch in chunked(unplanned_queries, 100):
        explains = explain_batch((query.sql_text, plan_type) for query in batch)
        # TODO: handle more gracefully
        if None in explains:
            l.error("Failed to get one or more explains!")
            break
        to_insert = [
            {
                "query_id": query.query_id,
                "plan_json": json.dumps(explain),
                "plan_type": plan_type,
                "schema": schema,
            }
            for (query, explain) in zip(batch, explains)
        ]
        inserted += (
            AliasWorkloadPlan.insert_many(to_insert)
            .on_conflict_ignore()
            .as_rowcount()
            .execute()
        )
        l.info(
            f"{round((inserted / len(unplanned_queries)) * 100, 2)}% planned {plan_type}"
        )


if __name__ == "__main__":
    # join_graph = build_join_graph(
    #     os.path.join(os.path.dirname(__file__), "../workload/stack/schema.sql")
    # )
    # aliased_join_graph = build_alias_join_graph(join_graph, 3)
    # gen_alias_join_queries(aliased_join_graph, "STACK")
    # for plan_type in PlanType:
    #     generate_plans(plan_type, "STACK")

    dsb_workload_set = get_workload_set("DSB")
    aliased_join_graph = build_alias_join_graph(dsb_workload_set.join_graph, 3)
    # gen_alias_join_queries(aliased_join_graph, "DSB")
    for plan_type in PlanType:
        generate_plans(plan_type, "DSB")
