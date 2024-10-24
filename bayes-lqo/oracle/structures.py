import time
from asyncio import Task
from dataclasses import dataclass
from typing import Union

from .provisioning import ExecutionEnvironment


@dataclass
class QueryExecutionSpec:
    id: str
    query: str
    timeout_secs: float


class UnscheduledQuery:
    spec: QueryExecutionSpec

    def __init__(self, query: QueryExecutionSpec):
        self.spec = query

    def __repr__(self):
        return f"UnscheduledQuery({self.spec})"


class UnfinishedQuery:
    spec: QueryExecutionSpec
    task: Task["QueryResult"]

    _start_time_secs: float

    def __init__(self, query: QueryExecutionSpec, task: Task["QueryResult"]):
        self.spec = query
        self.task = task
        self._start_time_secs = time.time()

    @property
    def time_since_start(self) -> float:
        return time.time() - self._start_time_secs

    def timeout_elapsed(self) -> bool:
        """The timeout has passed, though the query might have completed successfully"""
        return self.time_since_start >= self.spec.timeout_secs

    def adjust_timeout(self, new_timeout: float):
        self.spec.timeout_secs = new_timeout

    def __repr__(self):
        return f"UnfinishedQuery({self.query})"


@dataclass
class CompletedQuery:
    spec: QueryExecutionSpec
    elapsed_secs: float

    def __repr__(self):
        return f"CompletedQuery([{self.spec.id}] {self.spec.query} : {self.elapsed_secs})"


@dataclass
class TimedOutQuery:
    spec: QueryExecutionSpec
    elapsed_secs: float

    def __repr__(self):
        return f"TimedOutQuery([{self.spec.id}] {self.spec.query} : {self.elapsed_secs})"


@dataclass
class FailedQuery:
    spec: QueryExecutionSpec
    elapsed_secs: float
    error: Union[Exception, str]

    def __repr__(self):
        return f"FailedQuery([{self.spec.id}] {self.spec.query} : {self.error})"


QueryStatus = Union[CompletedQuery, TimedOutQuery, FailedQuery, UnfinishedQuery, UnscheduledQuery]
QueryResult = Union[CompletedQuery, TimedOutQuery, FailedQuery]


class ExecutionManager:
    envs: list[ExecutionEnvironment]
    work: dict[str, QueryStatus]

    _envs_in_use: dict[str, ExecutionEnvironment]

    def __init__(self, envs: list[ExecutionEnvironment], query_specs: list[QueryExecutionSpec]):
        self.envs = envs
        self.work = {spec.id: UnscheduledQuery(spec) for spec in query_specs}
        self._envs_in_use = {}

    def count_incomplete(self):
        return len(self.get_unfinished()) + len(self.get_unscheduled())

    def get_unscheduled(self) -> list[UnscheduledQuery]:
        return [q for q in self.work.values() if isinstance(q, UnscheduledQuery)]

    def get_unfinished(self) -> list[UnfinishedQuery]:
        return [q for q in self.work.values() if isinstance(q, UnfinishedQuery)]

    def get_available_envs(self) -> list[ExecutionEnvironment]:
        return [e for e in self.envs if e not in self._envs_in_use.values()]

    def begin_work(
        self,
        env: ExecutionEnvironment,
        query: UnscheduledQuery,
        task: Task[QueryResult],
    ):
        if query not in self.get_unscheduled():
            raise ValueError("Can only begin work on unscheduled tasks")
        self._envs_in_use[query.spec.id] = env
        self.work[query.spec.id] = UnfinishedQuery(query.spec, task)

    def finish_work(self, result: QueryResult):
        if result.spec.id not in [unfinished.spec.id for unfinished in self.get_unfinished()]:
            raise ValueError("Can only finish work on unfinished tasks")
        del self._envs_in_use[result.spec.id]
        self.work[result.spec.id] = result

    def result(self) -> list[QueryResult]:
        if self.count_incomplete() > 0:
            raise ValueError("Cannot get result until all queries are complete")

        return [
            q
            for q in self.work.values()
            if isinstance(q, CompletedQuery) or isinstance(q, TimedOutQuery) or isinstance(q, FailedQuery)
        ]
