import os
import pdb
import re
from dataclasses import dataclass
from itertools import combinations, product
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import sqlglot

from constants import USE_LOGGER

if (
    USE_LOGGER
):  # NOTE: option to remove logger becasue current Docker image doesn't support
    from logger.log import l
else:

    class l:
        @staticmethod
        def debug(*args, **kwargs):
            pass

        @staticmethod
        def info(*args, **kwargs):
            pass

        @staticmethod
        def error(*args, **kwargs):
            pass


def extract_join_keys(
    exprs,
) -> tuple[
    set[str],
    set[tuple[str, str]],
    dict[tuple[str, str], tuple[str, str]],
    dict[str, list[str]],
]:
    """
    Walks over all CREATE TABLE expressions in `exprs` and produces:
    - `tables`: the set of all tables in the schema
    - `key_columns`: the set of all (table, col) primary keys
    - `joins`: a dictionary of (table, col) => (table, col) with keys REFERENCES values
    """
    tables: set[str] = set()
    #  (table_name, column_name)
    key_columns: set[tuple[str, str]] = set()
    # (table_name, column_name) -> (table_name, column_name)
    joins: dict[tuple[str, str], tuple[str, str]] = {}
    # table_name -> [column_name]
    all_columns: dict[str, list[str]] = {}

    if USE_LOGGER:
        l.debug("table".center(15), "\t", "column".center(15), "\t", "type".center(11))
        l.debug("=" * 15, "\t", "=" * 15, "\t", "=" * 11)

    for expr in exprs:
        match expr:
            case sqlglot.expressions.Create():
                if not isinstance(expr.this.this, sqlglot.expressions.Table):
                    l.debug(f"Skipping CREATE for a non-table object: {expr.this.this}")
                    continue
                table_name = expr.this.this.name
                tables.add(table_name)
                info = ""
                # pdb.set_trace()
                for columndef_expr in expr.this.expressions:
                    match columndef_expr:
                        case sqlglot.expressions.ColumnDef():
                            column_name = columndef_expr.name
                            all_columns.setdefault(table_name, []).append(column_name)
                            info = ""
                            for constraint_expr in columndef_expr.constraints:
                                match constraint_expr.kind:
                                    case (
                                        sqlglot.expressions.PrimaryKeyColumnConstraint()
                                    ):
                                        info = "primary key"
                                        key_columns.add((table_name, column_name))
                                    case sqlglot.expressions.Reference():
                                        foreign_table_name = (
                                            constraint_expr.kind.this.this.name
                                        )
                                        foreign_column_name = (
                                            constraint_expr.kind.this.expressions[
                                                0
                                            ].name
                                        )
                                        joins[(table_name, column_name)] = (
                                            foreign_table_name,
                                            foreign_column_name,
                                        )
                                        info = "foreign key"
                            if not info:
                                info = "column"
                        case sqlglot.expressions.PrimaryKey():
                            info = "primary key"
                            key_columns.add((table_name, columndef_expr.name))
                    if USE_LOGGER:
                        l.debug(
                            f"{table_name.ljust(15)}\t{column_name.ljust(15)}\t{info}"
                        )
            case sqlglot.expressions.AlterTable():
                if "actions" in expr.args:
                    for action in expr.args["actions"]:
                        if isinstance(
                            action, sqlglot.expressions.AddConstraint
                        ) and isinstance(
                            action.expression, sqlglot.expressions.ForeignKey
                        ):
                            table = expr.this.this.name
                            column_names = [
                                e.name for e in action.expression.expressions
                            ]

                            foreign_table = action.expression.args[
                                "reference"
                            ].this.this.name
                            foreign_column_names = [
                                e.name
                                for e in action.expression.args[
                                    "reference"
                                ].this.expressions
                            ]
                            l.debug(
                                f"Found foreign key: {table}.({', '.join(column_names)}) -> {foreign_table}.({', '.join(foreign_column_names)})"
                            )

                            # This is not right, foreign keys can be on multiple columns.
                            # We treat each column separately and hope that it's only ever used to reference one foreign column.
                            for column_name, foreign_column_name in zip(
                                column_names, foreign_column_names
                            ):
                                joins[(table, column_name)] = (
                                    foreign_table,
                                    foreign_column_name,
                                )
            case _:
                if USE_LOGGER:
                    l.debug(f"Skipping expression with no join information: {expr}")
    return tables, key_columns, joins, all_columns


def build_join_graph(schema_file_path: str) -> nx.Graph:
    """
    Given a schema file with CREATE TABLE statements in `schema_file_path`, produce a directed graph
    with tables as nodes and foreign-key/primary-key relationships as edges. Edges point from
    foreign keys towards primary keys.
    """
    with open(schema_file_path) as f:
        schema_string = f.read()
        parsed = sqlglot.parse(schema_string, read="postgres")

    tables, key_columns, joins, all_columns = extract_join_keys(parsed)

    join_graph = nx.Graph()
    join_graph.add_nodes_from(tables)
    for (start_table, start_column), (end_table, end_column) in joins.items():
        join_graph.add_edge(
            start_table,
            end_table,
            table=start_table,
            attr=start_column,
            referenced_table=end_table,
            referenced_attr=end_column,
        )

    # nx.draw_networkx(join_graph)
    # plt.show()

    return join_graph


def get_all_columns(schema_file_path: str) -> dict[str, list[str]]:
    with open(schema_file_path) as f:
        schema_string = f.read()
        parsed = sqlglot.parse(schema_string, read="postgres")

    return extract_join_keys(parsed)[3]


def build_join_graph_from_queries(
    schema_file_path: str, queries: list[str]
) -> nx.Graph:
    with open(schema_file_path) as f:
        schema_string = f.read()
        parsed = sqlglot.parse(schema_string, read="postgres")

    join_graph = nx.Graph()

    tables, key_columns, _, all_columns = extract_join_keys(parsed)

    join_graph.add_nodes_from(tables)
    seen_tables = set()

    for sql in queries:
        expr = sqlglot.parse(sql, read="postgres")
        if len(expr) != 1:
            raise ValueError("Multiple statements in query")
        expr = expr[0]
        query_join_graph = build_query_join_graph(expr, all_columns)
        for edge in query_join_graph.edges(data=True):
            table1 = edge[2]["col1"][0][0]
            col1 = edge[2]["col1"]
            if isinstance(col1, tuple):
                col1 = col1[1]
            table2 = edge[2]["col2"][0][0]
            col2 = edge[2]["col2"]
            if isinstance(col2, tuple):
                col2 = col2[1]
            if table1 == table2:
                continue

            seen_tables.add(table1)
            seen_tables.add(table2)

            join_graph.add_edge(
                edge[0][0],
                edge[1][0],
                table=table1,
                attr=col1,
                referenced_table=table2,
                referenced_attr=col2,
            )

    # Cull tables that are not in the queries
    tables_to_remove = [table for table in tables if table not in seen_tables]
    for table in tables_to_remove:
        join_graph.remove_node(table)

    return join_graph


def build_alias_join_graph(join_graph: nx.Graph, num_aliases: int) -> nx.Graph:
    # nx.draw_networkx(join_graph)
    # plt.show()

    alias_join_graph = nx.Graph()
    # Add num_aliases aliases for each table
    for table in join_graph.nodes:
        for alias_num in range(1, num_aliases + 1):
            alias_join_graph.add_node((table, alias_num))

        # Include self-joins
        # for alias1, alias2 in combinations(range(num_aliases), 2):
        #     alias_join_graph.add_edge((table, alias1), (table, alias2))

    for table1, table2 in join_graph.edges:
        edge_data = join_graph.get_edge_data(table1, table2)
        for alias1, alias2 in product(range(1, num_aliases + 1), repeat=2):
            table_alias_1 = (table1, alias1)
            table_alias_2 = (table2, alias2)
            alias_join_graph.add_edge(
                table_alias_1,
                table_alias_2,
                table=edge_data["table"],
                attr=edge_data["attr"],
                referenced_table=edge_data["referenced_table"],
                referenced_attr=edge_data["referenced_attr"],
            )

    # nx.draw_networkx(alias_join_graph)
    # plt.show()

    return alias_join_graph


def parse_alias(alias: str) -> Optional[tuple[str, int]]:
    match = re.match(r"(?P<table_name>\D+)(?P<alias_num>\d+)", alias)
    if match:
        return match.group("table_name"), int(match.group("alias_num"))
    return None


def resolve_column_alias(
    alias: str, column_ref: str, all_columns: dict[str, list[str]]
) -> tuple[str, int]:
    parsed_alias = parse_alias(alias)
    if parsed_alias:
        return parsed_alias
    table_candidates = []
    for table, columns in all_columns.items():
        if column_ref in columns:
            table_candidates.append(table)
    if len(table_candidates) == 1:
        return table_candidates[0], 1
    raise ValueError(f"Could not resolve column alias {alias} for column {column_ref}")


def pretty_print_col(col_tuple: tuple[tuple[str, int], str]) -> str:
    ((table, alias), col) = col_tuple
    return f"{table}{alias}.{col}"


T_Col = tuple[tuple[str, int], str]


class EquiJoinSet:
    sets: set[frozenset[T_Col]]
    col_to_set: dict[T_Col, frozenset[T_Col]]

    def __init__(self):
        self.sets = set()
        self.col_to_set = {}

    def add_join(self, col1: T_Col, col2: T_Col):
        set1 = self.col_to_set.get(col1)
        set2 = self.col_to_set.get(col2)
        match set1, set2:
            case None, None:
                new_set = frozenset({col1, col2})
                self.col_to_set[col1] = new_set
                self.col_to_set[col2] = new_set
                self.sets.add(new_set)
            case None, _:
                new_set = set2.union({col1})
                self.col_to_set[col1] = new_set
                for col in set2:
                    self.col_to_set[col] = new_set
                self.sets.remove(set2)
                self.sets.add(new_set)
            case _, None:
                new_set = set1.union({col2})
                self.col_to_set[col2] = new_set
                for col in set1:
                    self.col_to_set[col] = new_set
                self.sets.remove(set1)
                self.sets.add(new_set)
            case _, _:
                if set1 == set2:
                    # For queries that specify the redundant predicates, we
                    # might already have these two columns in the same set.
                    return
                new_set = set1.union(set2).union({col1, col2})
                self.col_to_set[col1] = new_set
                self.col_to_set[col2] = new_set
                for col in set1.union(set2):
                    self.col_to_set[col] = new_set
                self.sets.remove(set1)
                self.sets.remove(set2)
                self.sets.add(new_set)

    def join_closure(self, col: T_Col) -> set[T_Col]:
        return self.col_to_set[col]


def build_query_join_graph(
    expr: sqlglot.expressions.Select, table_columns: dict[str, list[str]], pause=False
) -> nx.Graph:
    """Build a join graph containing only the join predictes in the query.

    Args:
        expr: the select expression with normalized alias numbers
    """
    query_join_graph = nx.Graph()
    predicates = expr.find(sqlglot.expressions.Where).this
    done = False
    join_sets = EquiJoinSet()

    for clause in predicates.find_all(sqlglot.expressions.EQ):
        if not isinstance(clause.this, sqlglot.expressions.Column) or not isinstance(
            clause.expression, sqlglot.expressions.Column
        ):
            continue

        try:
            table_alias1 = clause.this.table
            col1 = clause.this.name
            alias1 = resolve_column_alias(table_alias1, col1, table_columns)

            table_alias2 = clause.expression.table
            col2 = clause.expression.name
            alias2 = resolve_column_alias(table_alias2, col2, table_columns)
        except ValueError:
            # This only happens in subqueries where we didn't replace the aliases
            continue

        if not query_join_graph.has_node(alias1):
            query_join_graph.add_node(alias1)
        if not query_join_graph.has_node(alias2):
            query_join_graph.add_node(alias2)
        query_join_graph.add_edge(
            alias1, alias2, col1=(alias1, col1), col2=(alias2, col2)
        )
        join_sets.add_join((alias1, col1), (alias2, col2))

    # while not done:
    #     match predicates:
    #         case sqlglot.expressions.And():
    #             pass
    #         case sqlglot.expressions.Or():
    #             raise ValueError("Can only build query join graph for AND predicates")
    #         case _:
    #             # the last predicate
    #             done = True

    #     clause = predicates.expression if not done else predicates
    #     match clause:
    #         case sqlglot.expressions.EQ(
    #             this=sqlglot.expressions.Column(),
    #             expression=sqlglot.expressions.Column(),
    #         ):
    #             table_alias1 = clause.this.table
    #             col1 = clause.this.name
    #             alias1 = resolve_column_alias(table_alias1, col1, table_columns)

    #             table_alias2 = clause.expression.table
    #             col2 = clause.expression.name
    #             alias2 = resolve_column_alias(table_alias2, col2, table_columns)

    #             if pause:
    #                 print(f"{alias1}.{col1} = {alias2}.{col2}")

    #             if not query_join_graph.has_node(alias1):
    #                 query_join_graph.add_node(alias1)
    #             if not query_join_graph.has_node(alias2):
    #                 query_join_graph.add_node(alias2)
    #             query_join_graph.add_edge(
    #                 alias1, alias2, col1=(alias1, col1), col2=(alias2, col2)
    #             )
    #             join_sets.add_join((alias1, col1), (alias2, col2))
    #         case _:
    #             # Non-join predicates should be ignored
    #             pass
    #     # Step down the expression tree
    #     predicates = predicates.this

    # Also walk over the joins
    for join in expr.find_all(sqlglot.expressions.Join):
        if "on" not in join.args:
            continue
        for clause in join.find_all(sqlglot.expressions.EQ):
            col_expr_1 = clause.this
            col_1 = col_expr_1.name
            alias1 = resolve_column_alias(col_expr_1.table, col_1, table_columns)
            col_expr_2 = clause.expression
            col_2 = col_expr_2.name
            alias2 = resolve_column_alias(col_expr_2.table, col_2, table_columns)
            query_join_graph.add_edge(
                alias1, alias2, col1=(alias1, col_1), col2=(alias2, col_2)
            )
            join_sets.add_join((alias1, col_1), (alias2, col_2))

    # for i, join_set in enumerate(join_sets.sets):
    #     print(f"Join set {i}:")
    #     for col in join_set:
    #         print(f"\t{pretty_print_col(col)}")

    # plt.figure(1)
    # plt.title("Before adding missing edges")
    # pos = nx.spring_layout(query_join_graph)
    # nx.draw_networkx(query_join_graph, pos)

    # Make sure the join set edges are all present
    for join_set in join_sets.sets:
        for ((table1, alias1), col1), ((table2, alias2), col2) in combinations(
            join_set, 2
        ):
            if (not query_join_graph.has_edge((table1, alias1), (table2, alias2))) and (
                not query_join_graph.has_edge((table2, alias2), (table1, alias1))
            ):
                query_join_graph.add_edge(
                    (table1, alias1),
                    (table2, alias2),
                    col1=((table1, alias1), col1),
                    col2=((table2, alias2), col2),
                )

    # plt.figure(2)
    # plt.title("After adding missing edges")
    # nx.draw_networkx(query_join_graph, pos)
    # plt.show()
    # plt.close("all")

    # nx.draw_networkx_edge_labels(
    #     query_join_graph,
    #     pos,
    #     edge_labels={
    #         (
    #             u,
    #             v,
    #         ): f"{pretty_print_col(edge['col1'])} = {pretty_print_col(edge['col2'])}"
    #         for u, v, edge in query_join_graph.edges(data=True)
    #     },
    # )
    return query_join_graph


def get_all_table_names(exprs):
    tables = []
    for expr in exprs:
        match expr:
            case sqlglot.expressions.Create():
                if not isinstance(expr.this.this, sqlglot.expressions.Table):
                    continue
                tables.append(expr.this.this.name)
    return list(sorted(tables))


def build_table_order(schema_file_path: str) -> list[str]:
    with open(schema_file_path) as f:
        schema_string = f.read()
        parsed = sqlglot.parse(schema_string, read="postgres")

    return get_all_table_names(parsed)


if __name__ == "__main__":
    # join_graph = build_join_graph(
    #     os.path.join(os.path.dirname(__file__), "job/schema_fk.sql")
    # )
    # aliased_join_graph = build_alias_join_graph(join_graph, 3)
    # paths = nx.generate_random_paths(aliased_join_graph, 10_000, 3)
    # repeat_count = 0
    # for path in paths:
    #     path = [(t, a) for t, a in path]
    #     unique_nodes = set(path)
    #     if len(unique_nodes) != len(path):
    #         repeat_count += 1
    # if USE_LOGGER:
    #     l.info(f"Repeats: {repeat_count}/10,000")
    #     l.info(f"Number of edges: {len(aliased_join_graph.edges)}")
    join_graph = build_join_graph(
        os.path.join(os.path.dirname(__file__), "stack/schema.sql")
    )
    nx.draw_networkx(join_graph)
    plt.show()
