import pdb
import re
import time
from itertools import product
from random import shuffle

import typer
import wandb  # type: ignore
from logger.log import l
from oracle.oracle import _default_plan, _resolve_codec
from peewee import fn
from tqdm import tqdm

# from oracle.pg_celery_worker.pg_worker import tasks
from training_data.codec import build_join_tree
from workload.workloads import (
    OracleCodec,
    WorkloadSpec,
    WorkloadSpecDefinition,
    get_workload_set,
)

from .bao import get_hints, ranked_hint_badness
from .bao import sample as bao_sample
from .storage import (
    BaoJoinHint,
    BaoPlan,
    BaoScanHint,
    PostgresPlan,
    StackBaoInitialization,
    StackBayesBestPlan,
    StackWorkload,
    StackWorkloadTimeout,
)
from .utils import pretty_time

app = typer.Typer(no_args_is_help=True)


def template_name(query: str):
    return re.match(r"STACK_(Q\d+)-\d+", query).group(1)


@app.command()
def missing_queries():
    workload_def_set = get_workload_set("SO_FUTURE")
    all_work = sorted(list(workload_def_set.queries.keys()))
    for query in all_work:
        existing = (
            StackWorkload.select().where(StackWorkload.query_name == query).count() > 0
        )
        if existing:
            continue
        print(template_name(query), query)


def execute_with_hints(
    spec: WorkloadSpecDefinition,
    join_hint: BaoJoinHint,
    scan_hint: BaoScanHint,
    prewarm_factor: int,
    timeout: int,
):
    hint_statements = get_hints(join_hint, scan_hint)
    query_statements = hint_statements + [_default_plan(spec)]
    # TODO: BROKEN
    assert False
    result = tasks.pg_execute_query_high.delay(
        sql=query_statements,
        timeout=10 * 60 * 1000,
        db="so_future",
        db_user="so",
        prewarm_factor=prewarm_factor,
    ).get()
    return result


def record_result(query: str, result, join_hint: BaoJoinHint, scan_hint: BaoScanHint):
    template = template_name(query)
    if result["status"] == "complete":
        elapsed = result["duration (ns)"] / 1_000_000_000
        l.info(f"{query} ({join_hint}, {scan_hint}) in {elapsed} secs")
        StackWorkload.create(
            template=template,
            query_name=query,
            join_hint=join_hint,
            scan_hint=scan_hint,
            runtime_secs=elapsed,
        )
    elif result["status"] == "timeout":
        l.warning(f"{query} ({join_hint}, {scan_hint}) timed out on {result['worker']}")
        StackWorkloadTimeout.create(
            query_name=query,
            join_hint=join_hint,
            scan_hint=scan_hint,
            timeout_secs=result["duration (ns)"] / 1_000_000_000,
        )
    elif result["status"] == "failed":
        l.error(f"Query for {query} failed: {result['message']}")


# @app.command()
# def clean_account():
#     for query, spec in get_workload_set("SO_FUTURE").queries.items():
#         if "account_id" in spec.query_template:
#             print(query)
#             StackWorkload.delete().where(StackWorkload.query_name == query).execute()


@app.command()
def fill_stack():
    all_work = list(get_workload_set("SO_FUTURE").queries.items())
    shuffle(all_work)
    for work in all_work:
        query, spec = work

        existing = (
            StackWorkload.select().where(StackWorkload.query_name == query).count()
        )
        if existing > 0:
            continue

        join_hint = BaoJoinHint.NoHint
        scan_hint = BaoScanHint.NoHint
        prewarm_factor = 1
        timeout = 10 * 60

        if (
            StackWorkloadTimeout.select()
            .where(
                (StackWorkloadTimeout.query_name == query)
                & (StackWorkloadTimeout.join_hint == join_hint)
                & (StackWorkloadTimeout.scan_hint == scan_hint)
            )
            .count()
            > 0
        ):
            l.warning(f"{query} timed out previously")
            continue

        result = execute_with_hints(
            spec=spec,
            join_hint=join_hint,
            scan_hint=scan_hint,
            prewarm_factor=prewarm_factor,
            timeout=timeout,
        )
        record_result(query, result, join_hint, scan_hint)


@app.command()
def disk_test(query: str = typer.Option()):
    workload_def_set = get_workload_set("SO_FUTURE")
    spec = workload_def_set.queries[query]
    query_sql = _default_plan(spec)
    while True:
        # TODO: BROKEN
        assert False
        run_solo = tasks.pg_execute_query_high.delay(
            sql=query_sql,
            timeout=60 * 1000,
            db="so_future",
            db_user="so",
            prewarm_factor=1,
        ).get()
        l.info(f"{query}: {run_solo['duration (ns)'] / 1_000_000_000:.2f}")


@app.command()
def top_n(n: int):
    for template_num in range(1, 17):
        template = f"Q{template_num}"
        top_n = (
            StackWorkload.select()
            .where(StackWorkload.template == template)
            .order_by(StackWorkload.runtime_secs.desc())
            .limit(n)
        )
        print(f"Top {n} for {template}")
        for workload in top_n:
            print(workload.query_name, workload.runtime_secs)


def workload_queries(target: int, min_runtime: float) -> list[str]:
    n = 1
    while True:
        total = 0
        for template_num in range(1, 17):
            template = f"Q{template_num}"
            count = (
                StackWorkload.select()
                .where(StackWorkload.template == template)
                .where(StackWorkload.runtime_secs > min_runtime)
                .limit(n)
                .count()
            )
            total += count
        if total >= target:
            break
        else:
            n += 1

    queries = []
    for template_num in range(1, 17):
        template = f"Q{template_num}"
        top_n = (
            StackWorkload.select()
            .where(StackWorkload.template == template)
            .where(StackWorkload.runtime_secs > min_runtime)
            .order_by(StackWorkload.runtime_secs.desc())
            .limit(n)
        )
        for workload in top_n:
            queries.append(workload.query_name)
    return queries


@app.command()
def show_workload_queries(
    target: int = typer.Option(100), min_runtime: float = typer.Option(1.0)
):
    queries = workload_queries(target, min_runtime)
    for query in queries:
        runtime = (
            StackWorkload.select()
            .where(StackWorkload.query_name == query)
            .first()
            .runtime_secs
        )
        print(f"{query}: {runtime:.2f}")
    print(f"Total queries: {len(queries)}")
    template_counts = {}
    for query in queries:
        template = template_name(query)
        if template not in template_counts:
            template_counts[template] = 0
        template_counts[template] += 1
    print(f"n is {max(template_counts.values())}")


def initialize_query(workload_set: str, query: str, prewarm_factor: int):
    workload_def_set = get_workload_set(workload_set)
    spec = workload_def_set.queries[query]
    workload = WorkloadSpec.from_definition(spec, OracleCodec.Aliases)

    best_seen = (
        StackWorkload.select()
        .where(StackWorkload.query_name == query)
        .first()
        .runtime_secs
    )

    existing_best = (
        StackBaoInitialization.select(
            fn.MIN(StackBaoInitialization.runtime_secs).alias("best")
        )
        .where(
            (StackBaoInitialization.workload_set == workload_set)
            & (StackBaoInitialization.query_name == query)
        )
        .scalar()
    )
    if existing_best is not None and existing_best < best_seen:
        best_seen = existing_best
        l.info(f"{query}: Continuing run with timeout {best_seen:.2f}s")

    for join_hint, scan_hint in ranked_hint_badness():
        if (
            StackBaoInitialization.select()
            .where(
                (StackBaoInitialization.workload_set == workload_set)
                & (StackBaoInitialization.query_name == query)
                & (StackBaoInitialization.join_hint == join_hint)
                & (StackBaoInitialization.scan_hint == scan_hint)
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

        # First, check if this exact plan has already been run
        # TODO: BROKEN
        assert False
        explain_result = tasks.pg_execute_query_high.delay(
            hint_statements
            + [f"EXPLAIN (FORMAT JSON, SETTINGS ON) {_default_plan(spec)}"],
            60 * 1000,
            return_result=True,
            db=workload_def_set.db,
            db_user=workload_def_set.db_user,
            prewarm_factor=0,
        ).get()
        join_tree = build_join_tree(explain_result["result"][0][0][0]["Plan"])
        codec_inst = _resolve_codec(workload)
        encoded = codec_inst.encode(join_tree)
        existing_plans = (
            StackBaoInitialization.select()
            .where(
                (StackBaoInitialization.workload_set == workload_set)
                & (StackBaoInitialization.query_name == query)
            )
            .order_by(StackBaoInitialization.plan_id)
        )
        skip = False
        for plan in existing_plans:
            if tuple(plan.encoded_plan) == tuple(encoded):
                l.info(
                    f"{query} ({join_hint}, {scan_hint}): same plan already run, skipping"
                )
                StackBaoInitialization.create(
                    workload_set=workload_set,
                    query_name=query,
                    join_hint=join_hint,
                    scan_hint=scan_hint,
                    encoded_plan=encoded,
                    timed_out=plan.timed_out,
                    runtime_secs=plan.runtime_secs,
                )
                skip = True
                break
        if skip:
            continue

        # TODO: BROKEN
        assert False
        result = tasks.pg_execute_query_high.delay(
            query_statements,
            best_seen * 1000,
            return_result=True,
            db=workload_def_set.db,
            db_user=workload_def_set.db_user,
            prewarm_factor=prewarm_factor,
        ).get()

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
            # TODO: BROKEN
            assert False
            result = tasks.pg_execute_query_high.delay(
                hint_statements
                + [f"EXPLAIN (FORMAT JSON, SETTINGS ON) {_default_plan(spec)}"],
                60 * 1000,
                return_result=True,
                db=workload_def_set.db,
                db_user=workload_def_set.db_user,
                prewarm_factor=0,
            ).get()

        if not "result" in result:
            pdb.set_trace()
        join_tree = build_join_tree(result["result"][0][0][0]["Plan"])
        codec_inst = _resolve_codec(workload)
        encoded = codec_inst.encode(join_tree)

        l.info(
            f"{query} ({join_hint}, {scan_hint}): {'completed after ' if not timed_out else 'timed out after '}{runtime_secs:.2f}s"
        )

        if (
            StackBaoInitialization.select()
            .where(
                (StackBaoInitialization.workload_set == workload_set)
                & (StackBaoInitialization.query_name == query)
                & (StackBaoInitialization.join_hint == join_hint)
                & (StackBaoInitialization.scan_hint == scan_hint)
            )
            .count()
            > 0
        ):
            # This happens when two parallel runs end up on the same query
            return

        StackBaoInitialization.create(
            workload_set=workload_set,
            query_name=query,
            join_hint=join_hint,
            scan_hint=scan_hint,
            encoded_plan=encoded,
            timed_out=timed_out,
            runtime_secs=runtime_secs,
        )


@app.command()
def initialize(
    target: int = typer.Option(50),
    min_runtime: float = typer.Option(1.0),
    workload_set: str = typer.Option("SO_FUTURE"),
    force_check: bool = False,
):
    all_queries = workload_queries(target=target, min_runtime=min_runtime)
    shuffle(all_queries)
    for query in all_queries:
        if not force_check and (
            StackBaoInitialization.select()
            .where(
                (StackBaoInitialization.query_name == query)
                & (StackBaoInitialization.workload_set == workload_set)
            )
            .count()
            > 0
        ):
            continue
        l.info(f"Initializing {query}")
        initialize_query(workload_set, query, 1)


@app.command()
def check_initialized(
    workload_set: str = typer.Option("SO_FUTURE"),
    target: int = typer.Option(50),
    min_runtime: float = typer.Option(1.0),
):
    all_queries = workload_queries(target=target, min_runtime=min_runtime)
    print(len(all_queries), len(set(all_queries)))
    for query in all_queries:
        for join_hint, scan_hint in ranked_hint_badness():
            if (
                StackBaoInitialization.select()
                .where(
                    (StackBaoInitialization.workload_set == workload_set)
                    & (StackBaoInitialization.query_name == query)
                    & (StackBaoInitialization.join_hint == join_hint)
                    & (StackBaoInitialization.scan_hint == scan_hint)
                )
                .count()
                == 0
            ):
                l.info(f"Missing {query} ({join_hint}, {scan_hint})")


@app.command()
def show_initialization(
    workload_set: str = typer.Option("SO_FUTURE"),
    target: int = typer.Option(50),
    min_runtime: float = typer.Option(1.0),
):
    all_queries = workload_queries(target=target, min_runtime=min_runtime)
    print("query;runtime_secs;timed_out;encoded")
    for query in all_queries:
        initializations = StackBaoInitialization.select().where(
            (StackBaoInitialization.workload_set == workload_set)
            & (StackBaoInitialization.query_name == query)
        )
        already_shown = set()
        seen_inits = 0
        for init in initializations:
            seen_inits += 1
            encoded_key = tuple(init.encoded_plan)
            if encoded_key in already_shown:
                continue
            already_shown.add(encoded_key)
            print(
                f"{query};{init.runtime_secs:.2f};{init.timed_out};{','.join(str(c) for c in init.encoded_plan)}"
            )


@app.command()
def fill_bao(target: int = typer.Option(50), min_runtime: float = typer.Option(1.0)):
    all_queries = workload_queries(target=50, min_runtime=1.0)
    shuffle(all_queries)
    for query in all_queries:
        for join_hint, scan_hint in product(list(BaoJoinHint), list(BaoScanHint)):
            if (
                BaoPlan.select()
                .where(
                    (BaoPlan.query_name == query)
                    & (BaoPlan.workload_set == "SO_FUTURE")
                    & (BaoPlan.join_hint == join_hint)
                    & (BaoPlan.scan_hint == scan_hint)
                )
                .count()
                == 0
            ):
                bao_sample(query, join_hint, scan_hint, workload_set="SO_FUTURE")
            if (
                BaoPlan.select()
                .where(
                    (BaoPlan.query_name == query)
                    & (BaoPlan.workload_set == "SO_PAST")
                    & (BaoPlan.join_hint == join_hint)
                    & (BaoPlan.scan_hint == scan_hint)
                )
                .count()
                == 0
            ):
                bao_sample(query, join_hint, scan_hint, workload_set="SO_PAST")


# Separated from bayes.py because we need the encoded plans
@app.command()
def retrieve_bayes_best_plans():
    api = wandb.Api()
    runs = api.runs("<REMOVED FOR ANONYMIZATION>", order="+created_at")
    for wr in (r for r in tqdm(runs)):
        workload_name = wr.config.get("workload_name")
        run_name = wr.name
        workload_set = None
        if workload_name.startswith("STACK_"):
            if wr.config.get("so_future"):
                workload_set = "SO_FUTURE"
            else:
                workload_set = "SO_PAST"
        else:
            tqdm.write(f"Skipping {workload_name} ({run_name})")
            continue
        if (
            StackBayesBestPlan.select()
            .where(
                (StackBayesBestPlan.run_name == wr.name)
                & (StackBayesBestPlan.query_name == workload_name)
            )
            .count()
            > 0
        ):
            tqdm.write(f"Skipping {workload_name} ({run_name})")
            continue

        init = None
        if wr.config.get("init_w_random"):
            init = "random"
        elif wr.config.get("init_w_bao"):
            init = "bao"
        elif wr.config.get("init_w_llm"):
            init = "llm"
        else:
            init = "unknown"

        cross_joins = bool(wr.config.get("allow_cross_joins")) or False
        language = wr.config.get("which_query_language")

        history = list(
            wr.scan_history(
                keys=["non_parallel_runtime", "best_found", "best_input_seen"],
                page_size=100_000,
            )
        )
        length = len(history)
        last = history[-1]
        encoded_plan = last["best_input_seen"]
        runtime_secs = -1 * last["best_found"]
        tqdm.write(
            f"Retrieving {workload_name} ({run_name}): {length} iterations, best {pretty_time(runtime_secs)}"
        )

        StackBayesBestPlan.create(
            run_name=run_name,
            workload_set=workload_set,
            query_name=workload_name,
            init=init,
            cross_joins=cross_joins,
            language=language,
            length=length,
            best_encoded_plan=encoded_plan,
            best_runtime_secs=runtime_secs,
        )


@app.command()
def past_plan_on_future():
    all_queries = workload_queries(target=50, min_runtime=1.0)
    for query in all_queries:
        for join_hint, scan_hint in product(list(BaoJoinHint), list(BaoScanHint)):
            pass


@app.command()
def populate_bao(version: str = "SO_FUTURE"):
    queries = workload_queries(200, 1.0)
    shuffle(queries)
    for query in queries:
        join_hint = BaoJoinHint.NoHint
        scan_hint = BaoScanHint.NoHint
        existing = (
            BaoPlan.select()
            .where(
                (BaoPlan.query_name == query)
                & (BaoPlan.join_hint == join_hint)
                & (BaoPlan.scan_hint == scan_hint)
            )
            .count()
        )
        if existing > 0:
            continue
        bao_sample(query, join_hint, scan_hint, workload_set=version)


if __name__ == "__main__":
    app()
