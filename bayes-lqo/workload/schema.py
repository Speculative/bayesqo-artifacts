import os
import re
from itertools import combinations, product

import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import sqlglot
from constants import USE_LOGGER

if USE_LOGGER:  # NOTE: option to remove logger becasue current Docker image doesn't support
    from logger.log import l
else:

    class l:  # noqa: E742
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
) -> tuple[set[str], set[tuple[str, str]], dict[tuple[str, str], tuple[str, str]]]:
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
                for columndef_expr in expr.this.expressions:
                    match columndef_expr:
                        case sqlglot.expressions.ColumnDef():
                            column_name = columndef_expr.name
                            info = ""
                            for constraint_expr in columndef_expr.constraints:
                                match constraint_expr.kind:
                                    case sqlglot.expressions.PrimaryKeyColumnConstraint():
                                        info = "primary key"
                                        key_columns.add((table_name, column_name))
                                    case sqlglot.expressions.Reference():
                                        foreign_table_name = constraint_expr.kind.this.this.name
                                        foreign_column_name = constraint_expr.kind.this.expressions[0].name
                                        joins[(table_name, column_name)] = (
                                            foreign_table_name,
                                            foreign_column_name,
                                        )
                                        info = "foreign key"
                            if not info:
                                info = "column"
                    if USE_LOGGER:
                        l.debug(f"{table_name.ljust(15)}\t{column_name.ljust(15)}\t{info}")
            case sqlglot.expressions.AlterTable():
                if "actions" in expr.args:
                    for action in expr.args["actions"]:
                        if isinstance(action, sqlglot.expressions.AddConstraint) and isinstance(
                            action.expression, sqlglot.expressions.ForeignKey
                        ):
                            table = expr.this.this.name
                            column_names = [e.name for e in action.expression.expressions]

                            foreign_table = action.expression.args["reference"].this.this.name
                            foreign_column_names = [
                                e.name for e in action.expression.args["reference"].this.expressions
                            ]
                            l.debug(
                                f"Found foreign key: {table}.({', '.join(column_names)}) -> {foreign_table}.({', '.join(foreign_column_names)})"
                            )

                            # This is not right, foreign keys can be on multiple columns.
                            # We treat each column separately and hope that it's only ever used to reference one foreign column.
                            for column_name, foreign_column_name in zip(column_names, foreign_column_names):
                                joins[(table, column_name)] = (
                                    foreign_table,
                                    foreign_column_name,
                                )
            case _:
                if USE_LOGGER:
                    l.debug(f"Skipping expression with no join information: {expr}")
    return tables, key_columns, joins


def build_join_graph(schema_file_path: str) -> nx.Graph:
    """
    Given a schema file with CREATE TABLE statements in `schema_file_path`, produce a directed graph
    with tables as nodes and foreign-key/primary-key relationships as edges. Edges point from
    foreign keys towards primary keys.
    """
    with open(schema_file_path) as f:
        schema_string = f.read()
        parsed = sqlglot.parse(schema_string, read="postgres")

    tables, key_columns, joins = extract_join_keys(parsed)

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


def parse_alias(alias: str) -> tuple[str, int]:
    match = re.match(r"(?P<table_name>\D+)(?P<alias_num>\d+)", alias)
    if match:
        return match.group("table_name"), int(match.group("alias_num"))
    raise ValueError(f"Invalid alias: {alias}")


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


def build_query_join_graph(expr: sqlglot.expressions.Select) -> nx.Graph:
    """Build a join graph containing only the join predictes in the query.

    Args:
        expr: the select expression with normalized alias numbers
    """
    query_join_graph = nx.Graph()
    predicates = expr.find(sqlglot.expressions.Where).this
    done = False
    join_sets = EquiJoinSet()
    while not done:
        match predicates:
            case sqlglot.expressions.And():
                pass
            case sqlglot.expressions.Or():
                raise ValueError("Can only build query join graph for AND predicates")
            case _:
                # the last predicate
                done = True

        clause = predicates.expression if not done else predicates
        match clause:
            case sqlglot.expressions.EQ(
                this=sqlglot.expressions.Column(),
                expression=sqlglot.expressions.Column(),
            ):
                alias1 = parse_alias(clause.this.table)
                col1 = clause.this.name
                alias2 = parse_alias(clause.expression.table)
                col2 = clause.expression.name
                if not query_join_graph.has_node(alias1):
                    query_join_graph.add_node(alias1)
                if not query_join_graph.has_node(alias2):
                    query_join_graph.add_node(alias2)
                query_join_graph.add_edge(alias1, alias2, col1=(alias1, col1), col2=(alias2, col2))
                join_sets.add_join((alias1, col1), (alias2, col2))
            case _:
                # Non-join predicates should be ignored
                pass
        # Step down the expression tree
        predicates = predicates.this

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
        for ((table1, alias1), col1), ((table2, alias2), col2) in combinations(join_set, 2):
            if (not query_join_graph.has_edge((table1, alias1), (table2, alias2))) and (
                not query_join_graph.has_edge((table2, alias2), (table1, alias1))
            ):
                query_join_graph.add_edge(
                    (table1, alias1),
                    (table2, alias2),
                    col1=col1,
                    col2=col2,
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
    join_graph = build_join_graph(os.path.join(os.path.dirname(__file__), "stack/schema.sql"))
    nx.draw_networkx(join_graph)
    plt.show()
