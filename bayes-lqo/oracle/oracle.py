import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import psycopg  # type: ignore
from codec.codec import (
    AliasesCodec,
    Codec,
    HashProbeStackMachineCodec,
    HashProbeStackMachineWithOperatorsCodec,
    JoinTree,
    JoinTreeBranch,
    JoinTreeLeaf,
)
from constants import USE_LOGGER

if USE_LOGGER:
    from logger.log import l

from workload.workloads import OracleCodec, WorkloadSpec, WorkloadSpecDefinition

from .provisioning import ExecutionEnvironment
from .structures import (
    CompletedQuery,
    ExecutionManager,
    FailedQuery,
    QueryExecutionSpec,
    QueryResult,
    TimedOutQuery,
)


@dataclass
class WorkloadInput:
    id: str
    encoded_query: list[int]
    timeout_secs: float


@dataclass
class CostEstimateInput:
    id: str
    encoded_query: list[int]


@dataclass
class CostEstimateOutput:
    id: str
    encoded_query: list[int]
    cost: float


def _resolve_codec(workload: WorkloadSpec) -> Codec:
    match workload.codec:
        case OracleCodec.JoinOrder:
            return HashProbeStackMachineCodec(workload.all_tables)
        case OracleCodec.JoinOrderOperators:
            return HashProbeStackMachineWithOperatorsCodec(workload.all_tables)
        case OracleCodec.Aliases:
            return AliasesCodec(workload.all_tables)
        case _:
            raise ValueError("Unknown codec")


def _default_plan(workload: WorkloadSpecDefinition) -> str:
    return workload.query_template.format(
        ", ".join(
            [
                f"{table} AS {table}{alias_num}"
                for (table, num_aliases) in workload.query_tables
                for alias_num in range(1, num_aliases + 1)
            ]
        )
    )


def _decode_query(workload: WorkloadSpec, encoded: list[int]) -> str:
    codec = _resolve_codec(workload)
    join_tree = codec.decode(workload.query_tables, encoded)
    join_clause = join_tree.to_join_clause()
    if workload.codec == OracleCodec.JoinOrder:
        return workload.query_template.format(join_clause)
    elif workload.codec == OracleCodec.JoinOrderOperators:
        return f"/*+\n{join_tree.to_operator_hint()}\n*/\n{workload.query_template.format(join_clause)}"
    elif workload.codec == OracleCodec.Aliases:
        return f"/*+\n{join_tree.to_operator_hint()}\n*/\n{workload.query_template.format(join_clause)}"


async def execute_query(
    env: ExecutionEnvironment, spec: QueryExecutionSpec
) -> QueryResult:
    start_time = time.time()
    try:
        async with await psycopg.AsyncConnection.connect(
            host=env.host,
            port=env.port,
            user=env.user,
            password=env.password,
            dbname="imdb",
        ) as aconn:
            async with aconn.cursor() as cur:
                await cur.execute(
                    f"EXPLAIN (ANALYZE, FORMAT JSON, SETTINGS ON, TIMING OFF) {spec.query}"
                )
                res: Any = await cur.fetchone()
                execution_time = res[0][0]["Execution Time"]
                # Postgres returns ms, convert to seconds for consistency with other structures
                return CompletedQuery(spec, execution_time / 1000)
    except psycopg.Error as e:
        return FailedQuery(spec, time.time() - start_time, e)
    except (TypeError, IndexError, KeyError) as e:
        return FailedQuery(spec, time.time() - start_time, e)
    except asyncio.CancelledError:
        # Async connection context automatically cleans up/rolls back the postgres queries for us
        return TimedOutQuery(spec, time.time() - start_time)
    except Exception as e:
        if USE_LOGGER:
            l.opt(exception=True).warning("Unexpected error when executing query?")
        return FailedQuery(spec, time.time() - start_time, e)
    finally:
        # Destruct EC2 instance?
        pass


async def _oracle(
    envs: list[ExecutionEnvironment],
    query_specs: list[QueryExecutionSpec],
    progress_callback: Callable[[QueryResult], dict[str, float]],
) -> list[QueryResult]:
    """
    Runs a batch of queries in parallel and returns the execution time in milliseconds

    :param queries: List of queries to run
    :param progress_callback: Report progress on query completion/timeout. Called with the query and
        time that elapsed. Returns new timeouts for all queries.
    :return: List of execution times in milliseconds
    """

    manager = ExecutionManager(envs, query_specs)

    while manager.count_incomplete() > 0:
        if USE_LOGGER:
            l.debug(f"{manager.count_incomplete()} remaining queries in batch")
            l.debug(f"{len(manager.get_available_envs())} available environments")

        available_envs = manager.get_available_envs()
        unscheduled_work = manager.get_unscheduled()
        for env, unscheduled in zip(available_envs, unscheduled_work):
            if USE_LOGGER:
                l.info(f"Starting a query on {env.host}")
            manager.begin_work(
                env,
                unscheduled,
                asyncio.create_task(execute_query(env, unscheduled.spec)),
            )

        done, still_running = await asyncio.wait(
            [status.task for status in manager.get_unfinished()], timeout=1.0
        )

        # It's not worth it to complicate the progress callback by passing the batch,
        # we normally expect this to only contain one completion
        for finished_task in done:
            task_result = finished_task.result()
            manager.finish_work(task_result)
            if USE_LOGGER:
                l.info(f"Finished a query after {task_result.elapsed_secs:.2f} seconds")

            # Notify the caller that we made some progress
            new_timeouts = progress_callback(task_result)
            if len(new_timeouts) > 0:
                if USE_LOGGER:
                    l.warning("Adjusting timeouts mid-batch is no longer supported")

        # Check for timeouts.
        # We don't tell the ExecutionManager now, these tasks will show up as finished in the next loop
        for unfinished in manager.get_unfinished():
            if unfinished.timeout_elapsed():
                unfinished.task.cancel()

    return manager.result()


def oracle_for_workload_aws(
    workload: WorkloadSpec,
    envs: list[ExecutionEnvironment],
    workload_inputs: list[WorkloadInput],
    progress_callback: Callable[[QueryResult], dict[str, float]],
):
    return asyncio.run(
        _oracle(
            envs,
            [
                QueryExecutionSpec(
                    workload_input.id,
                    _decode_query(workload, workload_input.encoded_query),
                    workload_input.timeout_secs,
                )
                for i, workload_input in enumerate(workload_inputs)
            ],
            progress_callback,
        )
    )


PG_PASS = ""

INSERT_SQL = """
INSERT INTO job (sql_statement, target_db, db_user, timeout_ms) 
VALUES (%s, %s, %s, %s)
RETURNING id
"""

"""
Lifespan:
not-submitted -> sent-to-pg -> submitted -> in-progress -> complete
"""


class QueryTask:
    def __init__(self, sql, timeout_ms, db, user):
        self.__target_db = db
        self.__db_user = user
        self.__sql = sql
        self.__status = "not-submitted"

        assert type(timeout_ms) is int
        self.__timeout_ms = timeout_ms
        self.__id = None
        self.__result = None

    def submit(self):
        assert self.__status == "not-submitted"
        with psycopg.connect(
            host="<REMOVED FOR ANONYMIZATION>",
            user="bayesopt",
            dbname="bayesopt",
            password=PG_PASS,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    INSERT_SQL,
                    (self.__sql, self.__target_db, self.__db_user, self.__timeout_ms),
                )
                self.__id = cur.fetchone()[0]
                print("Submitted job ID:", self.__id)
                self.__status = "sent-to-pg"
        return self.__id

    def status(self):
        if self.__status == "not-submitted":
            return "not-submitted"

        with psycopg.connect(
            host="<REMOVED FOR ANONYMIZATION>",
            user="bayesopt",
            dbname="bayesopt",
            password=PG_PASS,
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, taken_by, result, issued_at, taken_at FROM job WHERE id=%s",
                    (self.__id,),
                )
                status, taken_by, result, issued_at, taken_at = cur.fetchone()
                self.__status = status

                if result is not None:
                    self.__result = json.loads(result)
                    self.__result["host"] = taken_by
                    self.__result["issued_at"] = issued_at.timestamp()
                    self.__result["taken_at"] = taken_at.timestamp()

                return status

    def result(self):
        while self.__status != "complete":
            self.status()
            time.sleep(1)

        return self.__result


def oracle_for_workload_cluster(
    workload: WorkloadSpec,
    workload_inputs: list[WorkloadInput],
) -> list[QueryResult]:
    # create a query execution spec and a future for each query
    task_futures = []
    for workload_input in workload_inputs:
        decoded = _decode_query(workload, workload_input.encoded_query)
        timeout_ms = int(workload_input.timeout_secs * 1000)
        task = QueryTask(decoded, timeout_ms, workload.db, workload.db_user)
        task.submit()
        spec = QueryExecutionSpec(workload_input.id, decoded, timeout_ms / 1000.0)
        task_futures.append((spec, task))

    # poll each future and collect the results
    query_results = []
    for spec, future in task_futures:
        result = future.result()
        if result["status"] == "complete":
            query_results.append(
                CompletedQuery(spec, result["duration (ns)"] / 1_000_000_000)
            )
        elif result["status"] == "timeout":
            query_results.append(
                TimedOutQuery(spec, result["duration (ns)"] / 1_000_000_000)
            )
        elif result["status"] == "failed":
            query_results.append(
                FailedQuery(
                    spec, result["duration (ns)"] / 1_000_000_000, result["message"]
                )
            )
        elif result["status"] == "error":
            raise RuntimeError("error from query executor: " + result["message"])
        else:
            if USE_LOGGER:
                l.opt(exception=True).warning("Unexpected error when executing query?")
            query_results.append(FailedQuery(spec, 0, result))

    return query_results


def _prepare_oracle(
    envs: list[ExecutionEnvironment], workload: WorkloadSpec, timeout: int
):
    def make_progress_callback():
        num_finished = 0

        start_time = time.time()

        def prepare_progress(result: QueryResult) -> dict[str, float]:
            nonlocal num_finished
            num_finished += 1
            if USE_LOGGER:
                l.info(
                    f"Finished preparing {num_finished} instances in {(time.time() - start_time):.2f} seconds"
                )
            return {}

        return prepare_progress

    asyncio.run(
        _oracle(
            envs,
            [
                QueryExecutionSpec(f"prepare-{i}", _default_plan(workload), timeout)
                for i in range(len(envs))
            ],
            make_progress_callback(),
        )
    )


async def _batch_estimate_cost(
    env: ExecutionEnvironment, workload: WorkloadSpec, batch: list[CostEstimateInput]
):
    codec = _resolve_codec(workload)  # noqa: F841

    output: dict[str, CostEstimateOutput] = {}

    try:
        async with await psycopg.AsyncConnection.connect(
            host=env.host,
            port=env.port,
            user=env.user,
            password=env.password,
            dbname="imdb",
        ) as aconn:
            # This isn't actually in parallel, but it seems to be faster than using a connection pool?
            for spec in batch:
                async with aconn.cursor() as cur:
                    query = _decode_query(workload, spec.encoded_query)
                    await cur.execute(
                        f"EXPLAIN (FORMAT JSON, SETTINGS ON, TIMING OFF) {query}"
                    )
                    res: Any = await cur.fetchone()
                    cost_estimate = float(res[0][0]["Plan"]["Total Cost"])
                    output[spec.id] = CostEstimateOutput(
                        spec.id, spec.encoded_query, cost_estimate
                    )
    except psycopg.Error as e:
        raise e
    except (TypeError, IndexError, KeyError) as e:
        raise e

    return output


async def _estimate_cost(
    envs: list[ExecutionEnvironment],
    workload: WorkloadSpec,
    cost_inputs: list[CostEstimateInput],
) -> list[CostEstimateOutput]:
    work_chunks = [cost_inputs[i :: len(envs)] for i in range(len(envs))]
    query_tasks = [
        asyncio.create_task(_batch_estimate_cost(env, workload, chunk))
        for env, chunk in zip(envs, work_chunks)
    ]
    cost_estimate_outputs: list[dict[str, CostEstimateOutput]] = await asyncio.gather(
        *query_tasks
    )
    output = {}
    for batch in cost_estimate_outputs:
        output.update(batch)
    return list(output.values())


def estimate_cost_for_workload(
    workload: WorkloadSpec,
    envs: list[ExecutionEnvironment],
    cost_inputs: list[CostEstimateInput],
):
    return asyncio.run(_estimate_cost(envs, workload, cost_inputs))


def _input_equivalent(
    workload: WorkloadSpec, encoded_1: list[int], encoded_2: list[int]
) -> bool:
    codec = _resolve_codec(workload)
    join_tree_1 = codec.decode(workload.query_tables, encoded_1)
    join_tree_2 = codec.decode(workload.query_tables, encoded_2)
    return join_tree_1.equal(join_tree_2)


def workload_input_equivalent(
    workload: WorkloadSpec, input_1: list[int], input_2: list[int]
) -> bool:
    return _input_equivalent(workload, input_1, input_2)


def _input_hash(workload: WorkloadSpec, encoded: list[int]) -> str:
    codec = _resolve_codec(workload)
    join_tree = codec.decode(workload.query_tables, encoded)
    return join_tree.stable_hash()


def workload_input_hash(workload: WorkloadSpec, encoded: list[int]) -> str:
    return _input_hash(workload, encoded)


def plan_has_crossjoin(workload: WorkloadSpec, encoded: list[int]) -> bool:
    codec = _resolve_codec(workload)
    join_tree = codec.decode(workload.query_tables, encoded)
    return _join_tree_has_crossjoin(workload, join_tree)


def _crossjoin_at_branch(
    workload: WorkloadSpec, left: JoinTree, right: JoinTree
) -> bool:
    left_aliases = left.tables_aliases()
    right_aliases = right.tables_aliases()

    filtered_graph = nx.subgraph_view(
        workload.schema.query_join_graph,
        filter_node=lambda n: n in left_aliases or n in right_aliases,
        filter_edge=lambda u, v: (u in left_aliases or u in right_aliases)
        and (v in left_aliases or v in right_aliases),
    )

    if not nx.is_connected(filtered_graph):
        # print("Found crossjoin:", left_aliases, right_aliases)
        return True


def _join_tree_has_crossjoin(workload: WorkloadSpec, join_tree: JoinTree) -> bool:
    match join_tree:
        case JoinTreeLeaf(_, _):
            return False
        case JoinTreeBranch(left, right, _):
            return (
                _crossjoin_at_branch(workload, left, right)
                or _join_tree_has_crossjoin(workload, left)
                or _join_tree_has_crossjoin(workload, right)
            )


if __name__ == "__main__":
    import time

    from workload.workloads import get_workload_set

    known_plan = [
        17,
        1,
        2,
        1,
        1,
        17,
        1,
        19,
        1,
        2,
        17,
        1,
        20,
        1,
        1,
        17,
        1,
        10,
        1,
        2,
        17,
        1,
        13,
        1,
        1,
        17,
        1,
        8,
        1,
        2,
    ]

    # test some IMDB queries
    JOB_QUERIES = get_workload_set(workload_set="CEB_3K")
    qdef = JOB_QUERIES.queries["CEB_10A0"]
    spec = WorkloadSpec.from_definition(definition=qdef, codec=OracleCodec.Aliases)

    inputs = [WorkloadInput("q1", known_plan, 5.0) for _ in range(25)]

    results = oracle_for_workload_cluster(spec, inputs)
    for r in results:
        print(r.elapsed_secs)
