import json
import pdb
import time
from itertools import combinations, islice
from pprint import pformat
from typing import Iterable

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from peewee import JOIN, AsIs, IntegrityError, chunked
from sqlglot import Expression, condition, select  # type: ignore

from logger.log import l
from workload.schema import build_join_graph, build_join_graph_from_queries
from workload.workloads import get_workload_set

from .planner import explain_batch, get_explain
from .storage import PlanType, QueryType, WorkloadPlan, WorkloadQuery


def join_tables(join_graph: nx.Graph, length: int) -> Iterable[list[str]]:
    """
    Generates all joins of `length` tables that do not involve cross products.
    Returns lists of strings where each string is a table name
    """
    tables = list(join_graph.nodes())

    for join_candidate in combinations(tables, length):
        filtered_graph = nx.subgraph_view(
            join_graph,
            filter_node=lambda n: n in join_candidate,
        )
        if nx.is_connected(filtered_graph):
            yield list(join_candidate)


def join_edges(
    join_graph: nx.Graph, join_tables: list[str]
) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """
    Find all join edges ((table1, col1), (table2, col2)) given the `join_graph` and the list of
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
    tables: list[str], edges: list[tuple[tuple[str, str], tuple[str, str]]]
) -> Expression:
    """Make a SQLGlot equi-join query given the `tables` and `edges` of the join."""
    expr = select("*").from_(tables[0])

    for table in tables[1:]:
        expr = expr.join(table)

    join_preds = None
    for (t1, c1), (t2, c2) in edges:
        # condition(f"{table}.{attr}").eq(f"{referenced_table}.{referenced_attr}")
        predicate = f"{t1}.{c1}={t2}.{c2}"
        if join_preds is None:
            join_preds = condition(predicate)
        else:
            join_preds = join_preds.and_(predicate)
    expr = expr.where(join_preds)
    return expr


def generate_queries(join_graph: nx.Graph):
    # Generate all queries
    for join_size in range(2, 18):
        all_joins = list(join_tables(join_graph, join_size))

        l.info(f"There are {len(all_joins)} joins of size {join_size}")
        existing = (
            WorkloadQuery.select().where(WorkloadQuery.num_joins == join_size).count()
        )
        if existing == len(all_joins):
            l.info(f"All plain join queries of size {join_size} already present")
            continue

        to_insert = []
        for tables in all_joins:
            edges = join_edges(join_graph, tables)
            join_expr = make_join_expr(tables, edges)
            # l.debug(join_expr.sql("postgres"))
            sql = join_expr.sql("postgres")

            to_insert.append(
                {
                    "join_key": tables,
                    "num_joins": join_size,
                    "num_aliases": 1,
                    "sql_text": sql,
                    "query_type": QueryType.PLAIN_JOIN,
                }
            )
        inserted = 0
        for batch in chunked(to_insert, 1000):
            inserted += (
                WorkloadQuery.insert_many(batch)
                .on_conflict_ignore()
                .as_rowcount()
                .execute()
            )
            l.info(
                f"Join Size {join_size}: {round((inserted / len(all_joins)) * 100)}% of queries"
            )
        l.info(
            f"Wrote {inserted}/{len(all_joins)} plain join queries of size {join_size}"
        )


def generate_plans(plan_type: PlanType):
    for join_size in range(2, 18):
        PlansOfType = WorkloadPlan.alias()
        plans_of_type = (
            PlansOfType.select()
            .where(PlansOfType.plan_type == plan_type)
            .cte("plans_of_type")
        )

        unplanned_queries = (
            WorkloadQuery.select(WorkloadQuery.query_id, WorkloadQuery.sql_text)
            .join(
                plans_of_type,
                JOIN.LEFT_OUTER,
                on=(WorkloadQuery.query_id == plans_of_type.c.query_id),
            )
            .where(
                (WorkloadQuery.num_joins == join_size)
                & (plans_of_type.c.plan_json == None)
            )
            .with_cte(plans_of_type)
        )
        l.info(
            f"Total unplanned {plan_type} queries for join size {join_size}: {len(unplanned_queries)}"
        )

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
                }
                for (query, explain) in zip(batch, explains)
            ]
            inserted += (
                WorkloadPlan.insert_many(to_insert)
                .on_conflict_ignore()
                .as_rowcount()
                .execute()
            )
            l.info(
                f"{round((inserted / len(unplanned_queries)) * 100)}% planned for join size {join_size}"
            )


if __name__ == "__main__":
    dsb_workload_set = get_workload_set("DSB")
    join_graph = dsb_workload_set.join_graph
    generate_queries(join_graph)

    # for plan_type in PlanType:
    #     generate_plans(plan_type)
