from enum import StrEnum, auto

from peewee import (
    AsIs,
    AutoField,
    BooleanField,
    DateTimeField,
    Field,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
    TextField,
)

db = SqliteDatabase("eval.db", timeout=5000)


class BaseModel(Model):
    class Meta:
        database = db


class EncodedPlanField(Field):
    field_type = "clist"

    def db_value(self, value):
        if value is None:
            return None
        return ",".join([str(c) for c in value])

    def python_value(self, value):
        if value is None:
            return None
        return [int(c) for c in value.split(",")]


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


class PlanType(StrEnum):
    # Selected by choosing random edges until the join tree is connected.
    # Operators are chosen uniformly at random.
    NonCrossUniformOp = auto()

    # Can contain cross joins.
    UniformRandom = auto()


class ExecutionResult(StrEnum):
    Success = auto()
    TimedOut = auto()
    Error = auto()


class RandomPlan(BaseModel):
    plan_id = AutoField()
    plan_type = TextField()
    workload_set = TextField()
    query_name = TextField()
    run_id = IntegerField()
    encoded_plan = EncodedPlanField()
    runtime_secs = FloatField()
    result = TextField()

    class Meta:
        indexes = ((("plan_type", "query_name", "run_id", "encoded_plan"), False),)


class SaturatedRandomRun(BaseModel):
    plan_type = TextField()
    workload_set = TextField()
    query_name = TextField()
    run_id = IntegerField()

    class Meta:
        indexes = ((("plan_type", "workload_set", "query_name", "run_id"), False),)


class PostgresPlan(BaseModel):
    plan_id = AutoField()
    workload_set = TextField()
    query_name = TextField()
    runtime_secs = FloatField()

    class Meta:
        indexes = ((("query_name",), False),)


class BaoJoinHint(StrEnum):
    NoHint = auto()
    NoHash = auto()
    NoMerge = auto()
    NoNestedLoops = auto()
    NoHashNoMerge = auto()
    NoHashNoNestedLoops = auto()
    NoMergeNoNestedLoops = auto()


class BaoScanHint(StrEnum):
    NoHint = auto()
    NoIndex = auto()
    NoSeq = auto()
    NoIndexOnly = auto()
    NoIndexNoSeq = auto()
    NoIndexNoIndexOnly = auto()
    NoSeqNoIndexOnly = auto()


class BaoPlan(BaseModel):
    plan_id = AutoField()
    workload_set = TextField()
    query_name = TextField()
    join_hint = TextField()
    scan_hint = TextField()
    runtime_secs = FloatField()
    encoded_plan = EncodedPlanField()

    class Meta:
        indexes = ((("query_name", "join_hint", "scan_hint"), False),)


class BaoInitialization(BaseModel):
    plan_id = AutoField()
    workload_set = TextField()
    query_name = TextField()
    join_hint = TextField()
    scan_hint = TextField()
    encoded_plan = EncodedPlanField()
    timed_out = BooleanField()
    runtime_secs = FloatField()

    class Meta:
        indexes = ((("query_name", "join_hint", "scan_hint"), False),)


class BayesRun(BaseModel):
    run_name = TextField(primary_key=True)
    workload_set = TextField()
    query_name = TextField()
    log_path = TextField()
    init = TextField()
    cross_joins = BooleanField()
    language = TextField()
    length = IntegerField()

    class Meta:
        indexes = ((("query_name", "init", "cross_joins", "language"), False),)


# Queries are calculated against SO_FUTURE
# But the same set of queries is used to evaluate against both SO_FUTURE and SO_PAST
class StackWorkload(BaseModel):
    template = TextField()
    query_name = TextField()
    join_hint = TextField()
    scan_hint = TextField()
    runtime_secs = FloatField()

    class Meta:
        indexes = ((("query_name",), False),)


class StackWorkloadTimeout(BaseModel):
    query_name = TextField()
    join_hint = TextField()
    scan_hint = TextField()
    timeout_secs = FloatField()


class StackBaoInitialization(BaseModel):
    plan_id = AutoField()
    workload_set = TextField()
    query_name = TextField()
    join_hint = TextField()
    scan_hint = TextField()
    encoded_plan = EncodedPlanField()
    timed_out = BooleanField()
    runtime_secs = FloatField()

    class Meta:
        indexes = ((("query_name", "join_hint", "scan_hint"), False),)


class StackBaoPlan(BaseModel):
    plan_id = AutoField()
    workload_set = TextField()
    query_name = TextField()
    join_hint = TextField()
    scan_hint = TextField()
    runtime_secs = FloatField()
    encoded_plan = EncodedPlanField()

    class Meta:
        indexes = ((("query_name", "join_hint", "scan_hint"), False),)


class StackBayesBestPlan(BaseModel):
    run_name = TextField(primary_key=True)
    workload_set = TextField()
    query_name = TextField()
    init = TextField()
    cross_joins = BooleanField()
    language = TextField()
    length = IntegerField()
    best_encoded_plan = EncodedPlanField()
    best_runtime_secs = FloatField()

    class Meta:
        indexes = ((("query_name", "init", "cross_joins", "language"), False),)


class BayesValidationRun(BaseModel):
    id = AutoField(primary_key=True)
    run_name = TextField()
    workload_set = TextField()
    query_name = TextField()
    encoded_plan = EncodedPlanField()
    runtime_secs = FloatField()

    class Meta:
        indexes = ((("query_name",), False),)


class StackShiftedOptimizedPlan(BaseModel):
    id = AutoField(primary_key=True)
    shifted_to = DateTimeField()
    query_name = TextField()
    plan_version = TextField()
    encoded_plan = EncodedPlanField()
    runtime_secs = FloatField()

    class Meta:
        indexes = (
            (
                (
                    "shifted_to",
                    "query_name",
                ),
                False,
            ),
        )


db.connect()
db.create_tables(
    [
        RandomPlan,
        SaturatedRandomRun,
        PostgresPlan,
        BaoPlan,
        BaoInitialization,
        BayesRun,
        StackWorkload,
        StackWorkloadTimeout,
        StackBaoInitialization,
        StackBayesBestPlan,
        BayesValidationRun,
        StackShiftedOptimizedPlan,
    ]
)
