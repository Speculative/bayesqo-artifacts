import os
import pickle
from dataclasses import dataclass
from enum import Enum

import networkx as nx
import sqlglot

from .schema import build_join_graph, build_query_join_graph, build_table_order


class OracleCodec(Enum):
    # The original codec, 2 characters per join for 2 tables
    JoinOrder = "join-order"

    # Codec with 3 characters per join: 2 for tables, 1 for operator
    JoinOrderOperators = "join-order-operators"

    # Codec with 5 characters per join: 2 for each table (name, alias), 1 for operator
    Aliases = "aliases"


@dataclass
class WorkloadSchema:
    # Join graph for the whole database (no aliases)
    db_join_graph: nx.Graph
    # Join graph (with aliases) for only tables in the query
    query_join_graph: nx.Graph


@dataclass
class WorkloadSpecDefinition:
    # MUST be in the canonical order
    all_tables: list[str]

    # The tables present in the query.
    # Each tuple is (table name, number of aliases)
    query_tables: list[tuple[str, int]]

    # MUST contain %s.
    # ALL tables MUST use numbered aliases starting from 1, even if there is
    # only one copy of a given table.
    query_template: str

    # The schema that the query executes against
    schema: WorkloadSchema

    # Name of the database to execute against
    db: str

    # The user to connect to the database as
    db_user: str

    # Whether to prewarm the tables before running the query
    prewarm: bool


@dataclass
class WorkloadSpec(WorkloadSpecDefinition):
    # The codec used to encode the workload inputs
    codec: OracleCodec

    @staticmethod
    def from_definition(definition: WorkloadSpecDefinition, codec: OracleCodec):
        return WorkloadSpec(
            definition.all_tables,
            definition.query_tables,
            definition.query_template,
            definition.schema,
            definition.db,
            definition.db_user,
            definition.prewarm,
            codec,
        )


class WorkloadDefinitionSet:
    # The order of this list defines the canonical order (mapping of numbers to table names)
    tables: list[str]

    # The join graph of the schema (no aliases)
    join_graph: nx.Graph

    # Query name -> WorkloadSpecDefinition
    # Retrieve query by name and combine with target language to create workload.
    queries: dict[str, WorkloadSpecDefinition]

    # Name of the database to execute against
    db: str

    # The user to connect to the database as
    db_user: str

    # Whether to prewarm the tables before running the query
    prewarm: bool

    def __init__(
        self,
        schema_file_path: str,
        queries_sql: dict[str, str],
        db="imdb",
        db_user="imdb",
        prewarm=False,
    ):
        self.tables = build_table_order(schema_file_path)
        self.join_graph = build_join_graph(schema_file_path)
        self.db = db
        self.db_user = db_user
        self.prewarm = prewarm
        self.queries = {}
        for query_name, query_sql in queries_sql.items():
            self.queries[query_name] = self.extract_query(query_sql, query_name)

    def extract_query(self, query_sql: str, query_name: str) -> WorkloadSpecDefinition:
        expr = sqlglot.parse_one(query_sql, read="postgres")
        if not isinstance(expr, sqlglot.expressions.Select):
            raise ValueError("Top-level of query should be SELECT")

        # =============
        # Process query
        # =============
        # table -> number of occurrences
        alias_count: dict[str, int] = {}
        # old alias -> new alias
        new_aliases: dict[str, str] = {}

        join_exprs = [expr.args["from"]] + expr.args["joins"]
        top_level_tables = [t for e in join_exprs for t in e.find_all(sqlglot.expressions.Table)]
        for table in top_level_tables:
            if table.alias not in new_aliases:
                if table.name not in alias_count:
                    alias_count[table.name] = 0
                alias_count[table.name] += 1
                alias = table.alias if table.alias else table.name
                new_aliases[alias] = f"{table.name}{alias_count[table.name]}"

        # ============================
        # Create query string template
        # ============================
        normalized_expr = expr.copy()

        # Replace FROM, remove JOINs
        # sqlglot makes it awkward to create a bare From expr
        new_from = sqlglot.expressions.from_("__REPLACE__").find(sqlglot.expressions.From)
        normalized_expr.find(sqlglot.expressions.From).replace(new_from)
        # for join in normalized_expr.find_all(sqlglot.expressions.Join):
        for join in normalized_expr.args["joins"]:
            assert isinstance(join, sqlglot.expressions.Join)
            join.pop()

        # Rewrite all tables to use the right aliases
        for col in normalized_expr.find_all(sqlglot.expressions.Column):
            try:
                col.replace(sqlglot.column(col.this.this, new_aliases[col.table]))
            except KeyError:
                # pdb.set_trace()
                # We may intentionally do not rewrite the aliases of tables not in the top-level join
                # i.e. this is probably within a subquery
                continue

        # Turn the modified query into a true format string
        sql_template = normalized_expr.sql(pretty=True).replace("__REPLACE__", "{}")

        # Build the WorkloadSchema
        schema = WorkloadSchema(
            db_join_graph=self.join_graph,
            query_join_graph=build_query_join_graph(normalized_expr),
        )

        return WorkloadSpecDefinition(
            self.tables,
            list(alias_count.items()),
            sql_template,
            schema,
            self.db,
            self.db_user,
            self.prewarm,
        )


# Very bad no good side effect on import. Sorry.
WORKLOAD_DEF_CACHE_DIR = os.path.join(os.path.dirname(__file__))

# ===
# JOB
# ===
IMDB_DIR = os.path.join(os.path.dirname(__file__), "job")
IMDB_SCHEMA_PATH = os.path.join(IMDB_DIR, "schema.sql")
imdb_queries: dict[str, str] = {}
for file_name in os.listdir(IMDB_DIR):
    if file_name.endswith(".sql") and file_name != "schema.sql":
        with open(os.path.join(IMDB_DIR, file_name)) as query_file:
            # Make the query name something like JOB_1A
            imdb_queries["JOB_" + file_name[:-4].upper()] = query_file.read()
IMDB_WORKLOAD_SET = WorkloadDefinitionSet(IMDB_SCHEMA_PATH, imdb_queries)

# ======
# CEB 3K
# ======
CEB_3K_WORKLOAD_SET = WorkloadDefinitionSet(IMDB_SCHEMA_PATH, {})


def load_ceb3k():
    # Takes about 15 seconds on my machine
    global CEB_3K_WORKLOAD_SET

    spec_cahe_path = os.path.join(WORKLOAD_DEF_CACHE_DIR, "ceb3k_spec_cache.pkl")
    if os.path.exists(spec_cahe_path):
        with open(spec_cahe_path, "rb") as f:
            CEB_3K_WORKLOAD_SET = pickle.load(f)
    else:
        CEB_3K_DIR = os.path.join(os.path.dirname(__file__), "ceb-3k")
        ceb_3k_queries: dict[str, str] = {}
        for template in os.listdir(CEB_3K_DIR):
            template_dir = os.path.join(CEB_3K_DIR, template)
            for file_name in os.listdir(template_dir):
                if file_name.endswith(".sql"):
                    with open(os.path.join(template_dir, file_name)) as query_file:
                        # Make the query name something like CEB_1A3
                        ceb_3k_queries["CEB_" + file_name[:-4].upper()] = query_file.read()
        CEB_3K_WORKLOAD_SET = WorkloadDefinitionSet(IMDB_SCHEMA_PATH, ceb_3k_queries)

        with open(spec_cahe_path, "wb") as f:
            pickle.dump(CEB_3K_WORKLOAD_SET, f)


# =======
# CEB 13K
# =======
CEB_13K_WORKLOAD_SET = WorkloadDefinitionSet(IMDB_SCHEMA_PATH, {})


def load_ceb13k():
    # Takes about 1 minute on my machine
    global CEB_13K_WORKLOAD_SET

    spec_cache_path = os.path.join(WORKLOAD_DEF_CACHE_DIR, "ceb13k_spec_cache.pkl")
    if os.path.exists(spec_cache_path):
        with open(spec_cache_path, "rb") as f:
            CEB_13K_WORKLOAD_SET = pickle.load(f)
    else:
        CEB_13K_DIR = os.path.join(os.path.dirname(__file__), "ceb-13k")
        ceb_13k_queries: dict[str, str] = {}
        for template in os.listdir(CEB_13K_DIR):
            template_dir = os.path.join(CEB_13K_DIR, template)
            for file_name in os.listdir(template_dir):
                if file_name.endswith(".sql"):
                    with open(os.path.join(template_dir, file_name)) as query_file:
                        # Make the query name something like CEB_1A3
                        ceb_13k_queries["CEB_" + file_name[:-4].upper()] = query_file.read()
        CEB_13K_WORKLOAD_SET = WorkloadDefinitionSet(IMDB_SCHEMA_PATH, ceb_13k_queries)

        with open(spec_cache_path, "wb") as f:
            pickle.dump(CEB_13K_WORKLOAD_SET, f)


# ==============
# Stack Overflow
# ==============

STACK_DIR = os.path.join(os.path.dirname(__file__), "stack")
STACK_SCHEMA_PATH = os.path.join(STACK_DIR, "schema.sql")
SO_PAST_WORKLOAD_SET = WorkloadDefinitionSet(STACK_SCHEMA_PATH, {})
SO_FUTURE_WORKLOAD_SET = WorkloadDefinitionSet(STACK_SCHEMA_PATH, {})


def load_so_past():
    global SO_PAST_WORKLOAD_SET

    spec_cache_path = os.path.join(WORKLOAD_DEF_CACHE_DIR, "so_past_spec_cache.pkl")
    if os.path.exists(spec_cache_path):
        with open(spec_cache_path, "rb") as f:
            SO_PAST_WORKLOAD_SET = pickle.load(f)
    else:
        STACK_DIR = os.path.join(os.path.dirname(__file__), "stack")
        stack_queries: dict[str, str] = {}
        for template in os.listdir(STACK_DIR):
            if not os.path.isdir(os.path.join(STACK_DIR, template)):
                continue
            template_dir = os.path.join(STACK_DIR, template)
            for file_name in os.listdir(template_dir):
                if file_name.endswith(".sql"):
                    with open(os.path.join(template_dir, file_name)) as query_file:
                        # Make the query name something like STACK_Q1-001
                        stack_queries["STACK_" + file_name[:-4].upper()] = query_file.read()
        SO_PAST_WORKLOAD_SET = WorkloadDefinitionSet(
            STACK_SCHEMA_PATH, stack_queries, db="so_past", db_user="so", prewarm=True
        )

        with open(spec_cache_path, "wb") as f:
            pickle.dump(SO_PAST_WORKLOAD_SET, f)


def load_so_future():
    global SO_FUTURE_WORKLOAD_SET

    spec_cache_path = os.path.join(WORKLOAD_DEF_CACHE_DIR, "so_future_spec_cache.pkl")
    if os.path.exists(spec_cache_path):
        with open(spec_cache_path, "rb") as f:
            SO_FUTURE_WORKLOAD_SET = pickle.load(f)
    else:
        STACK_DIR = os.path.join(os.path.dirname(__file__), "stack")
        stack_queries: dict[str, str] = {}
        for template in os.listdir(STACK_DIR):
            if not os.path.isdir(os.path.join(STACK_DIR, template)):
                continue
            template_dir = os.path.join(STACK_DIR, template)
            for file_name in os.listdir(template_dir):
                if file_name.endswith(".sql"):
                    with open(os.path.join(template_dir, file_name)) as query_file:
                        # Make the query name something like STACK_Q1-001
                        stack_queries["STACK_" + file_name[:-4].upper()] = query_file.read()
        SO_FUTURE_WORKLOAD_SET = WorkloadDefinitionSet(
            STACK_SCHEMA_PATH, stack_queries, db="so_future", db_user="so", prewarm=True
        )

        with open(spec_cache_path, "wb") as f:
            pickle.dump(SO_FUTURE_WORKLOAD_SET, f)


# Legacy workloads
EASY_WORKLOAD = IMDB_WORKLOAD_SET.queries["JOB_1A"]
HARD_WORKLOAD = IMDB_WORKLOAD_SET.queries["JOB_16B"]
JOB_17F = IMDB_WORKLOAD_SET.queries["JOB_17F"]
JOB_8C = IMDB_WORKLOAD_SET.queries["JOB_8C"]
JOB_7C = IMDB_WORKLOAD_SET.queries["JOB_7C"]
JOB_19D = IMDB_WORKLOAD_SET.queries["JOB_19D"]
JOB_9D = IMDB_WORKLOAD_SET.queries["JOB_9D"]
JOB_6F = IMDB_WORKLOAD_SET.queries["JOB_6F"]

# Keep this at the end
ALL_WORKLOADS = {
    name.upper(): workload for name, workload in globals().items() if isinstance(workload, WorkloadSpecDefinition)
}


def get_workload_set(workload_set: str) -> WorkloadDefinitionSet:
    if workload_set == "JOB":
        return IMDB_WORKLOAD_SET
    elif workload_set == "CEB_3K":
        if len(CEB_3K_WORKLOAD_SET.queries) == 0:
            load_ceb3k()
        return CEB_3K_WORKLOAD_SET
    elif workload_set == "CEB_13K":
        if len(CEB_13K_WORKLOAD_SET.queries) == 0:
            load_ceb13k()
        return CEB_13K_WORKLOAD_SET
    elif workload_set == "SO_PAST":
        if len(SO_PAST_WORKLOAD_SET.queries) == 0:
            load_so_past()
        return SO_PAST_WORKLOAD_SET
    elif workload_set == "SO_FUTURE":
        if len(SO_FUTURE_WORKLOAD_SET.queries) == 0:
            load_so_future()
        return SO_FUTURE_WORKLOAD_SET
    else:
        raise ValueError(f"Unknown workload set {workload_set}")


if __name__ == "__main__":
    stack_workload_set = get_workload_set("SO_FUTURE")
    print(stack_workload_set.queries["STACK_Q9-033"].query_template)
    print(stack_workload_set.queries["STACK_Q9-033"].query_tables)
