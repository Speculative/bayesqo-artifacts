from enum import StrEnum, auto

from peewee import (
    AutoField,
    Field,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

db = SqliteDatabase("workload.db")


class BaseModel(Model):
    class Meta:
        database = db


class CanonicalListField(Field):
    field_type = "clist"

    def db_value(self, value):
        return ",".join(sorted(value))

    def python_value(self, value):
        return value.split(",")


class CanonicalAliasListField(Field):
    field_type = "clist"

    def db_value(self, value):
        return ";".join(f"{table},{alias}" for table, alias in value)

    def python_value(self, value):
        parsed = []
        for table_alias in value.split(";"):
            table, alias = table_alias.split(",")
            parsed.append((table, int(alias)))
        return parsed


class QueryType(StrEnum):
    # SELECT * FROM t1, ...tn WHERE t1.a=t2.a AND ...tn-1.n = tn.n
    PLAIN_JOIN = auto()
    # SELECT * FROM table1 as t1_1, table1 as t1_2 ... tablen as tn_1 WHERE t1_1.a=t2_1.a ...tn-1.n = tn.n
    ALIAS_JOIN = auto()


class WorkloadQuery(BaseModel):
    query_id = AutoField()
    join_key = CanonicalListField()
    num_joins = IntegerField(index=True)
    sql_text = TextField()
    query_type = TextField()

    class Meta:
        # Unique constraint
        indexes = ((("join_key", "query_type"), True),)


class AliasWorkloadQuery(BaseModel):
    query_id = AutoField()
    join_key = CanonicalAliasListField()
    num_joins = IntegerField(index=True)
    num_aliases = IntegerField()
    sql_text = TextField()
    query_type = TextField()
    schema = TextField()

    class Meta:
        # Unique constraint
        indexes = ((("join_key", "query_type"), True),)


class PlanType(StrEnum):
    # No hints
    POSTGRES_DEFAULT = auto()
    # set enable_hashjoin=off
    NO_HASH_JOIN = auto()
    # set enable_hashjoin=off
    NO_LOOP_JOIN = auto()
    NO_SEQ_SCAN = auto()
    NO_INDEX_SCAN = auto()


class WorkloadPlan(BaseModel):
    plan_id = AutoField()
    query_id = ForeignKeyField(WorkloadQuery)
    plan_json = TextField()
    plan_type = TextField()

    class Meta:
        # Unique constraint
        indexes = ((("query_id", "plan_type"), True),)


class AliasWorkloadPlan(BaseModel):
    plan_id = AutoField()
    query_id = ForeignKeyField(AliasWorkloadQuery)
    plan_json = TextField()
    plan_type = TextField()
    schema = TextField()

    class Meta:
        # Unique constraint
        indexes = ((("query_id", "plan_type"), True), (("schema", "plan_type"), False))


db.connect()
db.create_tables([WorkloadQuery, WorkloadPlan, AliasWorkloadQuery, AliasWorkloadPlan])
