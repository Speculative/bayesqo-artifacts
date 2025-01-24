import pdb
import time
from itertools import product
from random import shuffle
from statistics import median
from subprocess import DEVNULL, run

import typer
from peewee import fn
from psycopg import sql

from logger.log import l
from oracle.oracle import QueryTask, _default_plan, _resolve_codec
from training_data.codec import build_join_tree
from workload.workloads import OracleCodec, WorkloadSpec, get_workload_set

from .bao import get_hints, ranked_hint_badness
from .stack import workload_queries as stack_workload_queries
from .storage import BaoInitialization, BaoJoinHint, BaoScanHint, PostgresPlan

app = typer.Typer(no_args_is_help=True)


def initialize_query(workload_set: str, query: str, prewarm_factor: int):
    """
    The main difference between this and bao.py is that these plans can time out,
    and we prioritize them based on which hints tend to perform best. Most plans
    will not be executed to completion.
    """
    workload_def_set = get_workload_set(workload_set)
    spec = workload_def_set.queries[query]
    workload = WorkloadSpec.from_definition(spec, OracleCodec.Aliases)

    best_seen = 10 * 60

    postgres_best = (
        PostgresPlan.select(fn.MIN(PostgresPlan.runtime_secs))
        .where(
            (PostgresPlan.workload_set == workload_set)
            & (PostgresPlan.query_name == query)
        )
        .scalar()
    )
    if postgres_best is not None and postgres_best < best_seen:
        best_seen = postgres_best
        l.info(f"{query}: Continuing run with timeout {best_seen:.2f}s")

    existing_best = (
        BaoInitialization.select(fn.MIN(BaoInitialization.runtime_secs).alias("best"))
        .where(
            (BaoInitialization.workload_set == workload_set)
            & (BaoInitialization.query_name == query)
        )
        .scalar()
    )
    if existing_best is not None and existing_best < best_seen:
        best_seen = existing_best
        l.info(f"{query}: Continuing run with timeout {best_seen:.2f}s")

    for join_hint, scan_hint in ranked_hint_badness():
        if (
            BaoInitialization.select()
            .where(
                (BaoInitialization.workload_set == workload_set)
                & (BaoInitialization.query_name == query)
                & (BaoInitialization.join_hint == join_hint)
                & (BaoInitialization.scan_hint == scan_hint)
            )
            .count()
            > 0
        ):
            # Skip any hint sets we've already done for this query
            l.info(f"{query} ({join_hint}, {scan_hint}): skipped")
            continue

        hint_statements = get_hints(join_hint, scan_hint)
        query_statements = hint_statements + [
            f"EXPLAIN (ANALYZE, FORMAT JSON, SETTINGS ON, TIMING OFF) {_default_plan(spec)}",
        ]
        sql = "; ".join(query_statements)
        task = QueryTask(
            sql,
            timeout_ms=int(best_seen * 1000),
            db=workload_def_set.db,
            user=workload_def_set.db_user,
        )
        task.submit()
        result = task.result()

        # We do not record times for failed queries, but we'll need to redo them later
        if result["status"] == "failed":
            l.error(f"{query} failed: {result['message']}")
            return

        timed_out = result["status"] == "timeout"
        runtime_secs = result["duration (ns)"] / 1_000_000_000

        if not timed_out:
            best_seen = runtime_secs
        else:
            # Timed out queries still need EXPLAIN so we can record the encoded plan
            sql = "; ".join(
                hint_statements
                + [f"EXPLAIN (FORMAT JSON, SETTINGS ON) {_default_plan(spec)}"]
            )
            task = QueryTask(
                sql,
                timeout_ms=60 * 1000,
                db=workload_def_set.db,
                user=workload_def_set.db_user,
            )
            task.submit()
            result = task.result()

        if not "result" in result:
            pdb.set_trace()
        join_tree = build_join_tree(result["result"][0][0][0]["Plan"])
        codec_inst = _resolve_codec(workload)
        encoded = codec_inst.encode(join_tree)

        l.info(
            f"{query} ({join_hint}, {scan_hint}): {'completed after ' if not timed_out else 'timed out after '}{runtime_secs:.2f}s"
        )

        if (
            BaoInitialization.select()
            .where(
                (BaoInitialization.workload_set == workload_set)
                & (BaoInitialization.query_name == query)
                & (BaoInitialization.join_hint == join_hint)
                & (BaoInitialization.scan_hint == scan_hint)
            )
            .count()
            > 0
        ):
            # This happens when two parallel runs end up on the same query
            return

        BaoInitialization.create(
            workload_set=workload_set,
            query_name=query,
            join_hint=join_hint,
            scan_hint=scan_hint,
            encoded_plan=encoded,
            timed_out=timed_out,
            runtime_secs=runtime_secs,
        )


@app.command()
def initialize_workload(
    workload_set: str = typer.Option(),
    prewarm_factor: int = typer.Option(0),
    force_recheck: bool = typer.Option(False),
):
    workload_def_set = get_workload_set(workload_set)
    all_queries = list(workload_def_set.queries.keys())
    shuffle(all_queries)
    for query in all_queries:
        if (
            not force_recheck
            and BaoInitialization.select()
            .where(BaoInitialization.query_name == query)
            .count()
            > 0
        ):
            continue
        initialize_query(workload_set, query, prewarm_factor)


def cancel_queries():
    run(
        ["ansible-playbook", "-i", "hosts", "cancel-queries.yaml"],
        cwd="oracle/pg_celery_worker/ansible",
        stdout=DEVNULL,
        stderr=DEVNULL,
    )


@app.command()
def concurrent_initialize(
    workload_set: str = typer.Option(), threads: int = typer.Option(8)
):
    workload_def_set = get_workload_set(workload_set)
    prioritized_hints = list(ranked_hint_badness())
    for query, spec_def in workload_def_set.queries.items():
        # If we have a postgres plan, it was a shorter query
        if PostgresPlan.select().where(PostgresPlan.query_name == query).count() > 0:
            continue

        filtered_hints = [
            (join_hint, scan_hint)
            for join_hint, scan_hint in prioritized_hints
            if (
                BaoInitialization.select()
                .where(
                    (BaoInitialization.workload_set == workload_set)
                    & (BaoInitialization.query_name == query)
                    & (BaoInitialization.join_hint == join_hint)
                    & (BaoInitialization.scan_hint == scan_hint)
                )
                .count()
                == 0
            )
        ]
        grouped_hints = [
            filtered_hints[i : i + threads]
            for i in range(0, len(filtered_hints), threads)
        ]
        l.info(f"Initializing {query}")
        best_seen = 10 * 60
        maybe_existing_best = (
            BaoInitialization.select(fn.MIN(BaoInitialization.runtime_secs))
            .where(BaoInitialization.query_name == query)
            .scalar()
        )
        if maybe_existing_best is not None:
            best_seen = maybe_existing_best
        spec = WorkloadSpec.from_definition(spec_def, OracleCodec.Aliases)
        for hint_group in grouped_hints:
            # Dispatch this set of hints concurrently
            l.info(
                f"Dispatching {query} ({len(hint_group)} hints), timeout {best_seen} secs"
            )
            futures = {}
            for join_hint, scan_hint in hint_group:
                hint_statements = get_hints(join_hint, scan_hint)
                query_statements = hint_statements + [
                    f"EXPLAIN (ANALYZE, FORMAT JSON, SETTINGS ON, TIMING OFF) {_default_plan(spec_def)}",
                ]
                result_future = tasks.pg_execute_query_high.delay(
                    query_statements,
                    best_seen * 1000,
                    return_result=True,
                    db=workload_def_set.db,
                    db_user=workload_def_set.db_user,
                    prewarm_factor=1,
                )
                futures[(join_hint, scan_hint)] = result_future

            l.info("Waiting for queries to complete...")
            # Wait until at least one finishes
            results = {}
            start = time.time()
            while True:
                l.info("...")
                done = False
                for hint, future in futures.items():
                    if future.ready():
                        result = future.get()
                        results[hint] = result
                        if result["status"] == "complete":
                            elapsed = result["duration (ns)"] / 1_000_000_000
                            l.info(f"{query} ({hint}): completed in {elapsed} secs")
                            if elapsed < best_seen:
                                best_seen = elapsed
                        done = True
                if done:
                    break
                else:
                    if (time.time() - start) > max(2 * best_seen, 30):
                        l.info("Timed out waiting for queries to complete")
                        break
                    time.sleep(1)

            # Cancel the remaining queries
            l.info("Cancelling queries...")
            for future in futures.values():
                if not future.ready():
                    future.revoke()
            cancel_queries()
            l.info("Queries cancelled")

            for result in results.values():
                if result["status"] == "complete":
                    best_seen = min(best_seen, result["duration (ns)"] / 1_000_000_000)

            for join_hint, scan_hint in hint_group:
                # Done on previous runs
                if (
                    BaoInitialization.select()
                    .where(
                        (BaoInitialization.workload_set == workload_set)
                        & (BaoInitialization.query_name == query)
                        & (BaoInitialization.join_hint == join_hint)
                        & (BaoInitialization.scan_hint == scan_hint)
                    )
                    .count()
                    > 0
                ):
                    continue

                result = results.get((join_hint, scan_hint))
                runtime_secs = (
                    result["duration (ns)"] / 1_000_000_000
                    if result is not None
                    else best_seen
                )
                timed_out = result is None or result["status"] == "timeout"
                if timed_out:
                    # Timed out queries still need EXPLAIN so we can record the encoded plan
                    hint_statements = get_hints(join_hint, scan_hint)
                    result = tasks.pg_execute_query_high.delay(
                        hint_statements
                        + [f"EXPLAIN (FORMAT JSON, SETTINGS ON) {_default_plan(spec)}"],
                        60 * 1000,
                        return_result=True,
                        db=workload_def_set.db,
                        db_user=workload_def_set.db_user,
                        prewarm_factor=0,
                    ).get()
                join_tree = build_join_tree(result["result"][0][0][0]["Plan"])
                codec_inst = _resolve_codec(spec)
                encoded = codec_inst.encode(join_tree)
                BaoInitialization.create(
                    workload_set=workload_set,
                    query_name=query,
                    join_hint=join_hint,
                    scan_hint=scan_hint,
                    encoded_plan=encoded,
                    timed_out=timed_out,
                    runtime_secs=runtime_secs,
                )


@app.command()
def summarize(workload_set: str = typer.Option()):
    workload_def_set = get_workload_set(workload_set)
    total_queries = len(workload_def_set.queries)
    initialized_queries = (
        BaoInitialization.select(BaoInitialization.query_name)
        .distinct()
        .where(BaoInitialization.workload_set == workload_set)
        .count()
    )
    print(f"{initialized_queries}/{total_queries} queries initialized")


if __name__ == "__main__":
    app()
