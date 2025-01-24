import os
import pdb
import sys
import time
from datetime import date
from enum import StrEnum, auto
from subprocess import PIPE, Popen
from typing import Literal, Optional, Union

import pandas as pd
import typer  # type: ignore
import wandb  # type: ignore
from dateutil.relativedelta import relativedelta
from logger.log import l
from oracle.oracle import WorkloadInput, oracle_for_workload_cluster
from oracle.structures import CompletedQuery, FailedQuery, TimedOutQuery
from peewee import fn
from tqdm import tqdm
from workload.workloads import (
    OracleCodec,
    WorkloadSpec,
    WorkloadSpecDefinition,
    get_workload_set,
)

from .bao import ranked_hint_badness, top_bot_improvement, top_pg_runtime
from .eval_workloads import EvalWorkload, resolve_eval_queries, resolve_workload_set
from .shift_so import generate_shift_sql
from .stack import workload_queries as stack_queries
from .storage import (
    BaoPlan,
    BayesRun,
    BayesValidationRun,
    StackBaoInitialization,
    StackShiftedOptimizedPlan,
    db,
)
from .utils import JOB_QUERIES_SORTED, pretty_time

app = typer.Typer(no_args_is_help=True)


def matches_config(workload_run, cross_joins: bool, bao_init: bool, llm_init: bool):
    run_cross_joins = workload_run.config.get("allow_cross_joins")
    run_bao_init = workload_run.config.get("init_w_bao")
    run_llm_init = workload_run.config.get("init_w_llm")
    return (
        bool(run_cross_joins) == cross_joins
        and bool(run_bao_init) == bao_init
        and bool(run_llm_init) == llm_init
    )


def completed_runs():
    api = wandb.Api()
    runs = api.runs(
        "<REMOVED FOR ANONYMIZATION>",
        filters={"tags": {"$in": ["Paper"]}},
        order="+created_at",
    )
    for wr in (r for r in tqdm(runs)):
        workload_name = wr.config.get("workload_name")
        if not workload_name:
            tqdm.write(f"Got a run without a workload_name? {wr}")
            continue
        yield wr


@app.command()
def list_runs(cross_joins: bool = False, bao_init: bool = True, llm_init: bool = False):
    run_count = {}
    for wr in completed_runs():
        workload_name = wr.config.get("workload_name")
        run_count[workload_name] = run_count.get(workload_name, 0) + 1

    for workload_name, count in run_count.items():
        print(f"{workload_name}: {count}")

    print(f"Total: {sum(run_count.values())}")


def get_one_run(id: str):
    api = wandb.Api()
    return api.run(f"<REMOVED FOR ANONYMIZATION>/{id}")


@app.command()
def download_runs(run_id: Optional[str] = None):
    new_runs = 0
    seen_runs = []
    for wr in completed_runs() if not run_id else [get_one_run(run_id)]:
        workload_name = wr.config.get("workload_name")
        run_name = wr.name
        # if (run_name, workload_name) in seen_runs:
        #     pdb.set_trace()
        seen_runs.append((run_name, workload_name))

        target_dir = os.path.join(
            os.path.dirname(__file__),
            f"bayes/{workload_name}",
        )
        os.makedirs(target_dir, exist_ok=True)

        workload_set = None
        if workload_name.startswith("STACK_"):
            if wr.config.get("so_future"):
                if wr.config.get("force_past_vae") == True:
                    workload_set = "SO_DRIFT"
                else:
                    workload_set = "SO_FUTURE"
            else:
                workload_set = "SO_PAST"
        else:
            if workload_name.startswith("JOB_"):
                workload_set = "JOB"
            elif workload_name.startswith("CEB_"):
                workload_set = "CEB_3K"
            elif workload_name.startswith("DSB_"):
                workload_set = "DSB"

        if (
            BayesRun.select()
            .where(
                (BayesRun.query_name == workload_name) & (BayesRun.run_name == run_name)
            )
            .exists()
        ):
            # tqdm.write(f"Skipping {run_name}")
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

        target_file = os.path.join(target_dir, f"{run_name}.csv")
        if os.path.exists(target_file):
            tqdm.write(f"Skipping downloading {target_file}")
        else:
            history = list(
                wr.scan_history(
                    keys=[
                        "non_parallel_runtime",
                        "best_found",
                        "best_input_seen",
                        "n_oracle_calls",
                    ],
                    page_size=100_000,
                )
            )
            length = len(history)
            tqdm.write(f"Downloading {run_name} ({workload_name}, {init}, {length})")
            # history = wr.history()
            # length = history.shape[0]

            df = pd.DataFrame(history)
            df.to_csv(target_file)

            BayesRun.create(
                run_name=run_name,
                workload_set=workload_set,
                query_name=workload_name,
                log_path=target_file,
                init=init,
                cross_joins=cross_joins,
                language=language,
                length=length,
            )

            new_runs += 1

        # Run configs/logs
        # files = list(run.files())
        # for file in files:
        #     print(f"Downloading {file.name} to {target_dir}")
        #     file.download(target_dir)
    tqdm.write(f"Downloaded {new_runs} new runs")
    # pdb.set_trace()


@app.command()
def redownload_runs():
    for run in tqdm(BayesRun.select()):
        target_file = run.log_path
        if os.path.exists(target_file):
            tqdm.write(f"Skipping downloading {target_file}")
            continue

        api = wandb.Api()
        runs = api.runs(
            "<REMOVED FOR ANONYMIZATION>",
            filters={"display_name": run.run_name},
        )
        if len(runs) != 1:
            tqdm.write(f"Expected 1 run for {run.run_name}, got {len(runs)}")
            continue
        wr = runs[0]
        history = list(
            wr.scan_history(
                keys=[
                    "non_parallel_runtime",
                    "best_found",
                    "best_input_seen",
                    "n_oracle_calls",
                    "max_score_most_recent_batch",
                ],
                page_size=100_000,
            )
        )
        length = len(history)
        tqdm.write(
            f"Downloading {run.run_name} ({run.query_name}, {run.init}, {length})"
        )
        target_dir = os.path.join(
            os.path.dirname(__file__),
            f"bayes/{run.query_name}",
        )
        os.makedirs(target_dir, exist_ok=True)
        df = pd.DataFrame(history)
        df.to_csv(target_file)


@app.command()
def download_retro_runs():
    api = wandb.Api()
    runs = api.runs(
        "<REMOVED FOR ANONYMIZATION>",
        filters={"tags": {"$in": ["STACK_PAST_BO_INIT"]}},
    )
    for wr in (r for r in tqdm(runs)):
        history = list(
            wr.scan_history(
                keys=[
                    "non_parallel_runtime",
                    "best_found",
                    "best_input_seen",
                    "n_oracle_calls",
                    "max_score_most_recent_batch",
                ],
                page_size=100_000,
            )
        )
        length = len(history)
        run_name = wr.name
        workload_name = wr.config.get("workload_name")
        cross_joins = bool(wr.config.get("allow_cross_joins")) or False
        language = wr.config.get("which_query_language")
        tqdm.write(f"Downloading {run_name} ({workload_name}, {length})")
        target_dir = os.path.join(
            os.path.dirname(__file__),
            f"bayes/{workload_name}",
        )
        target_file = os.path.join(target_dir, f"{run_name}.csv")

        df = pd.DataFrame(history)
        df.to_csv(target_file)
        BayesRun.create(
            run_name=run_name,
            workload_set="SO_RETRO",
            query_name=workload_name,
            log_path=target_file,
            init="bao",
            cross_joins=cross_joins,
            language=language,
            length=length,
        )


class InitType(StrEnum):
    random = auto()
    bao = auto()
    llm = auto()
    unknown = auto()


def bayes_series(
    query: str,
    workload_set: str,
    init: set[InitType],
    cross_joins: bool = False,
) -> list[list[tuple[float, float]]]:
    """Produces a list of all salient optimization points in a given
    optimization run, i.e. each point where runtime improved.

    Each tuple is (cumulative optimization time, plan runtime)
    """

    run_paths = BayesRun.select(BayesRun.log_path).where(
        (BayesRun.query_name == query)
        & (BayesRun.workload_set == workload_set)
        & (BayesRun.init.in_(init))
        & (BayesRun.cross_joins == cross_joins)
        & (BayesRun.length > 30)
    )

    runs = []
    for run_path in run_paths:
        run_df = pd.read_csv(run_path.log_path)
        if run_df.empty:
            continue
        run_points = []
        try:
            for _, row in run_df.dropna(
                subset=["non_parallel_runtime", "best_found"]
            ).iterrows():
                # non_parallel_runtime is in hours
                cumulative_time = row["non_parallel_runtime"] * (60 * 60)
                # scores are inverted because optimization process tries to make score higher
                plan_time = -1 * row["best_found"]
                run_points.append((cumulative_time, plan_time))
        except KeyError:
            continue
        runs.append(run_points)

    return runs


def best_bayes_series(
    query: str, workload_set: str, init: set[InitType], cross_joins: bool = False
):
    runs = bayes_series(query, workload_set, init, cross_joins)
    if not runs:
        raise ValueError(f"No bayes runs found for {query}")
    return min(runs, key=lambda r: r[-1][1])


@app.command()
def eval_status():
    print("Missing for JOB:")
    missing = 0
    for query in JOB_QUERIES_SORTED:
        if BayesRun.select().where(BayesRun.query_name == query).count() == 0:
            print(query)
            missing += 1
    print(f"Missing {missing} of {len(JOB_QUERIES_SORTED)}")

    print()
    print("Missing for CEB_3K:")
    top, bot = top_bot_improvement(workload_set="CEB_3K", n=100)
    top_100 = top_pg_runtime("CEB_3K", 100)
    all_queries = set(top + bot + top_100)
    missing = 0
    for query in all_queries:
        if BayesRun.select().where(BayesRun.query_name == query).count() == 0:
            print(query)
            missing += 1
    print(f"Missing {missing} of {len(all_queries)}")

    so_50 = set(stack_queries(target=50, min_runtime=1.0))
    print()
    print("Missing for SO_PAST:")
    missing = 0
    for query in so_50:
        if (
            BayesRun.select()
            .where(
                (BayesRun.query_name == query) & (BayesRun.workload_set == "SO_PAST")
            )
            .count()
            == 0
        ):
            print(query)
            missing += 1
    print(f"Missing {missing} of {len(so_50)}")

    print()
    print("Missing for SO_DRIFT:")
    missing = 0
    for query in so_50:
        if (
            BayesRun.select()
            .where(
                (BayesRun.query_name == query) & (BayesRun.workload_set == "SO_DRIFT")
            )
            .count()
            == 0
        ):
            print(query)
            missing += 1
    print(f"Missing {missing} of {len(so_50)}")

    so_200 = set(stack_queries(target=200, min_runtime=1.0))
    print()
    print("Missing for SO_FUTURE:")
    missing = 0
    for query in so_200:
        if (
            BayesRun.select()
            .where(
                (BayesRun.query_name == query) & (BayesRun.workload_set == "SO_FUTURE")
            )
            .count()
            == 0
        ):
            print(query)
            missing += 1
    print(f"Missing {missing} of {len(so_200)}")

    dsb = set(get_workload_set("DSB").queries.keys())
    print()
    print("Missing for DSB:")
    missing = 0
    for query in dsb:
        if (
            BayesRun.select()
            .where((BayesRun.query_name == query) & (BayesRun.workload_set == "DSB"))
            .count()
            == 0
        ):
            print(query)
            missing += 1
    print(f"Missing {missing} of {len(dsb)}")


def get_optimal_plan(
    query: str, workload_set_name: str
) -> Optional[tuple[list[int], float]]:
    workload_set = get_workload_set(workload_set_name)
    run_paths = BayesRun.select(BayesRun.run_name, BayesRun.log_path).where(
        (BayesRun.query_name == query) & (BayesRun.workload_set == workload_set_name)
    )

    if run_paths.count() != 1:
        l.warning(f"Expected 1 run for {query}, got {run_paths.count()}")
        return None

    run_path = run_paths[0]
    run_df = pd.read_csv(run_path.log_path)
    if run_df.empty:
        l.warning(f"Empty run for {query}")
        return None
    last_row = run_df.dropna(subset=["best_found", "best_input_seen"]).iloc[-1]
    best_plan = [int(s) for s in last_row["best_input_seen"][1:-1].split(",")]
    best_runtime = -1 * last_row["best_found"]

    return best_plan, best_runtime


@app.command()
def show_optimal_plan(query: str = typer.Option(), workload_set: str = typer.Option()):
    result = get_optimal_plan(query, workload_set)
    if result:
        plan, runtime = result
        print(f"Plan: {plan}")
        print(f"Runtime: {runtime}")
    else:
        print("No optimal plan found")


@app.command()
def validate_best_plans():
    # for eval_workload in EvalWorkload:
    for eval_workload in [EvalWorkload.STACK_200]:
        workload_set_name = resolve_workload_set(eval_workload)
        workload_set = get_workload_set(workload_set_name)
        queries = resolve_eval_queries(eval_workload)
        for query in queries:
            existing_runs = (
                BayesValidationRun.select()
                .where(
                    (BayesValidationRun.query_name == query)
                    & (BayesValidationRun.workload_set == workload_set_name)
                )
                .count()
            )

            if existing_runs == 5:
                l.info(f"Done with {query}")
                continue

            # Extract the best plan
            result = get_optimal_plan(query, workload_set_name)
            if not result:
                l.warning(f"No optimal plan found for {query}")
                continue
            best_plan, best_runtime = result

            # Record some new oracle executions for the best plan
            workload = workload_set.queries[query]
            workload_spec = WorkloadSpec.from_definition(workload, OracleCodec.Aliases)
            for i in range(existing_runs, 5):
                result = oracle_for_workload_cluster(
                    workload_spec,
                    [
                        WorkloadInput(
                            f"{query}_validation_{i}", best_plan, 2 * best_runtime + 1
                        )
                    ],
                )[0]
                match result:
                    case CompletedQuery(_, elapsed_secs):
                        l.info(
                            f"{query} #{i}: Finished in {pretty_time(elapsed_secs)}, BO record was {pretty_time(best_runtime)}"
                        )
                        BayesValidationRun.create(
                            run_name=run_path.run_name,
                            workload_set=workload_set_name,
                            query_name=query,
                            encoded_plan=best_plan,
                            runtime_secs=elapsed_secs,
                        )
                    case TimedOutQuery(_, elapsed_secs):
                        l.warning(
                            f"{query} #{i}: Timed out after {pretty_time(elapsed_secs)}"
                        )
                    case FailedQuery(_, elapsed_secs, error):
                        l.info(
                            f"{query} #{i}: Failed query after {pretty_time(elapsed_secs)}: {error}"
                        )


@app.command()
def best_plan_comparison():
    print("query_name,validation_samples,validation_avg,bo_runtime,diff")
    # for eval_workload in EvalWorkload:
    out: tuple[str, int, float, float, float] = []
    for eval_workload in [EvalWorkload.STACK_200]:
        workload_set_name = resolve_workload_set(eval_workload)
        workload_set = get_workload_set(workload_set_name)
        queries = resolve_eval_queries(eval_workload)
        for query in queries:
            bo_result = get_optimal_plan(query, workload_set_name)
            if not bo_result:
                l.warning(f"No optimal plan found for {query}")
                continue
            _, bo_runtime = bo_result
            validation_samples = (
                BayesValidationRun.select()
                .where(
                    (BayesValidationRun.query_name == query)
                    & (BayesValidationRun.workload_set == workload_set_name)
                )
                .count()
            )
            validation_avg = (
                BayesValidationRun.select(fn.AVG(BayesValidationRun.runtime_secs))
                .where(
                    (BayesValidationRun.query_name == query)
                    & (BayesValidationRun.workload_set == workload_set_name)
                )
                .scalar()
            )
            out.append(
                (
                    query,
                    validation_samples,
                    validation_avg,
                    bo_runtime,
                    validation_avg - bo_runtime,
                )
            )
    out = sorted(out, key=lambda x: x[-1])
    for row in out:
        print(",".join(str(x) for x in row))


@app.command()
def past_to_future():
    queries = resolve_eval_queries(EvalWorkload.STACK_50)
    for query in queries:
        if (
            BayesValidationRun.select()
            .where(
                (BayesValidationRun.query_name == query)
                & (BayesValidationRun.workload_set == "SO_RETRO")
            )
            .exists()
        ):
            continue
        best_plan, best_runtime = get_optimal_plan(query, "SO_PAST")
        workload = get_workload_set("SO_FUTURE").queries[query]
        workload_spec = WorkloadSpec.from_definition(workload, OracleCodec.Aliases)
        result = oracle_for_workload_cluster(
            workload_spec,
            [WorkloadInput(f"{query}_retro", best_plan, 4 * best_runtime + 60)],
        )[0]
        match result:
            case CompletedQuery(_, elapsed_secs):
                l.info(
                    f"Finished for {query} in {elapsed_secs} vs original {best_runtime}"
                )
                BayesValidationRun.create(
                    run_name="retro",
                    workload_set="SO_RETRO",
                    query_name=query,
                    encoded_plan=best_plan,
                    runtime_secs=elapsed_secs,
                )
            case TimedOutQuery(_, elapsed_secs):
                l.warning(f"Timed out for {query} after {4 * best_runtime}")
            case FailedQuery(_, elapsed_secs, error):
                l.warning(f"Failed for {query} after {4 * best_runtime}: {error}")


@app.command()
def past_to_future_init():
    print("query;runtime_secs;timed_out;encoded_plan")
    queries = resolve_eval_queries(EvalWorkload.STACK_50)
    for query in queries:
        bao_inits = StackBaoInitialization.select(
            StackBaoInitialization.encoded_plan,
            StackBaoInitialization.runtime_secs,
            StackBaoInitialization.timed_out,
        ).where(
            (StackBaoInitialization.query_name == query)
            & (StackBaoInitialization.workload_set == "SO_FUTURE")
        )
        for init in bao_inits:
            print(
                f"{query};{init.runtime_secs};{init.timed_out};{','.join(str(c) for c in init.encoded_plan)}"
            )

        retro_init = (
            BayesValidationRun.select(
                BayesValidationRun.encoded_plan, BayesValidationRun.runtime_secs
            )
            .where(
                (BayesValidationRun.query_name == query)
                & (BayesValidationRun.workload_set == "SO_RETRO")
            )
            .first()
        )
        print(
            f"{query};{retro_init.runtime_secs};False;{','.join(str(c) for c in retro_init.encoded_plan)}"
        )


def get_bayes_run_history(query_name: str, workload_set: str):
    run_paths = BayesRun.select(BayesRun.log_path).where(
        (BayesRun.query_name == query_name) & (BayesRun.workload_set == workload_set)
    )

    if run_paths.count() != 1:
        l.warning(f"Expected 1 run for {query_name}, got {run_paths.count()}")
        return None

    run_path = run_paths[0]
    run_df = pd.read_csv(run_path.log_path)
    if run_df.empty:
        l.warning(f"Empty run for {query_name}")
        return None

    return run_df


@app.command()
def fill_so_shift():
    eval_workload = EvalWorkload.STACK_50
    past_workload_set = get_workload_set("SO_PAST")
    future_workload_set = get_workload_set("SO_FUTURE")
    shifted_workload_set = get_workload_set("SO_SHIFTED")
    queries = resolve_eval_queries(eval_workload)
    past_optimal_plans: dict[str, tuple[list[int], float]] = {}
    future_optimal_plans: dict[str, tuple[list[int], float]] = {}
    for query in queries:
        # Extract the best plan
        future_result = get_optimal_plan(query, "SO_FUTURE")
        if not future_result:
            l.warning(f"No optimal plan found for {query}")
            continue
        best_plan, best_runtime = future_result
        future_optimal_plans[query] = (best_plan, best_runtime)

        past_result = get_optimal_plan(query, "SO_PAST")
        if not past_result:
            l.warning(f"No optimal plan found for {query}")
            continue
        best_plan, best_runtime = past_result
        past_optimal_plans[query] = (best_plan, best_runtime)

    BASE_DATE = date(2018, 1, 1)
    for date_spec in [
        date(2019, 9, 2),
        BASE_DATE + relativedelta(months=18),
        BASE_DATE + relativedelta(months=12),
        BASE_DATE + relativedelta(months=6),
        BASE_DATE + relativedelta(months=3),
        BASE_DATE + relativedelta(months=2),
        BASE_DATE + relativedelta(months=1),
        BASE_DATE + relativedelta(weeks=1),
        BASE_DATE + relativedelta(days=1),
        BASE_DATE,
    ]:
        done = True
        for query in queries:
            if not (
                StackShiftedOptimizedPlan.select()
                .where(
                    (StackShiftedOptimizedPlan.shifted_to == date_spec)
                    & (StackShiftedOptimizedPlan.query_name == query)
                )
                .exists()
            ):
                done = False
        if done:
            l.info(f"Already done {date_spec}")
            continue

        l.info(f"Setting date to {date_spec}")

        timing = r"\"\timing\""
        sql = r"\"" + generate_shift_sql(date_spec) + r"\""
        sudo = f'sudo -u postgres psql -U so -d so_shift -c {timing} -c {sql} -c "VACUUM ANALYZE"'
        ssh_command = f'ssh <REMOVED FOR ANONYMIZATION> "{sudo}"'

        Popen(
            ssh_command,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
            bufsize=0,
        ).wait()

        for query in queries:
            if (
                StackShiftedOptimizedPlan.select()
                .where(
                    (StackShiftedOptimizedPlan.shifted_to == date_spec)
                    & (StackShiftedOptimizedPlan.query_name == query)
                )
                .count()
                >= 4
            ):
                l.info(f"Already done {query}")
                continue

            workload = shifted_workload_set.queries[query]
            workload_spec = WorkloadSpec.from_definition(workload, OracleCodec.Aliases)

            past_plan, past_runtime = past_optimal_plans[query]
            result = oracle_for_workload_cluster(
                workload_spec,
                [WorkloadInput(f"{query}_shifted", past_plan, 10 * past_runtime + 1)],
            )[0]
            match result:
                case CompletedQuery(_, elapsed_secs):
                    l.info(
                        f"{query}: Finished past in {pretty_time(elapsed_secs)}, BO record was {pretty_time(past_runtime)}"
                    )
                    StackShiftedOptimizedPlan.create(
                        shifted_to=date_spec,
                        query_name=query,
                        plan_version="past",
                        encoded_plan=past_plan,
                        runtime_secs=elapsed_secs,
                    )
                case TimedOutQuery(_, elapsed_secs):
                    l.warning(
                        f"{query}: Timed out past after {pretty_time(elapsed_secs)}"
                    )
                case FailedQuery(_, elapsed_secs, error):
                    l.info(
                        f"{query}: Failed past query after {pretty_time(elapsed_secs)}: {error}"
                    )

            future_plan, future_runtime = future_optimal_plans[query]
            result = oracle_for_workload_cluster(
                workload_spec,
                [
                    WorkloadInput(
                        f"{query}_shifted", future_plan, 10 * future_runtime + 1
                    )
                ],
            )[0]
            match result:
                case CompletedQuery(_, elapsed_secs):
                    l.info(
                        f"{query}: Finished future in {pretty_time(elapsed_secs)}, BO record was {pretty_time(future_runtime)}"
                    )
                    StackShiftedOptimizedPlan.create(
                        shifted_to=date_spec,
                        query_name=query,
                        plan_version="future",
                        encoded_plan=future_plan,
                        runtime_secs=elapsed_secs,
                    )
                case TimedOutQuery(_, elapsed_secs):
                    l.warning(
                        f"{query}: Timed out future after {pretty_time(elapsed_secs)}"
                    )
                case FailedQuery(_, elapsed_secs, error):
                    l.info(
                        f"{query}: Failed future query after {pretty_time(elapsed_secs)}: {error}"
                    )


@app.command()
def stack_retro_stats():
    past_plans = BayesValidationRun.select().where(
        BayesValidationRun.workload_set == "SO_RETRO"
    )
    past_runtimes = [past_plan.runtime_secs for past_plan in past_plans]
    queries = [past_plan.query_name for past_plan in past_plans]
    future_runtimes = [get_optimal_plan(query, "SO_FUTURE")[1] for query in queries]

    past_median = np.median(past_runtimes)
    past_p90 = np.percentile(past_runtimes, 90)
    future_median = np.median(future_runtimes)
    future_p90 = np.percentile(future_runtimes, 90)

    print(f"Past median: {past_median}")
    print(f"Future median: {future_median}")
    print(f"Past P90: {past_p90}")
    print(f"Future P90: {future_p90}")

    from datetime import date

    future_shifted_plans = (
        StackShiftedOptimizedPlan.select(fn.AVG(StackShiftedOptimizedPlan.runtime_secs))
        .where(
            (StackShiftedOptimizedPlan.plan_version == "future")
            & (StackShiftedOptimizedPlan.shifted_to == date(2019, 9, 2))
        )
        .group_by(StackShiftedOptimizedPlan.query_name)
    )
    past_shifted_plans = (
        StackShiftedOptimizedPlan.select(fn.AVG(StackShiftedOptimizedPlan.runtime_secs))
        .where(
            (StackShiftedOptimizedPlan.plan_version == "past")
            & (StackShiftedOptimizedPlan.shifted_to == date(2019, 9, 2))
        )
        .group_by(StackShiftedOptimizedPlan.query_name)
    )
    future_shifted_runtimes = [plan.runtime_secs for plan in future_shifted_plans]
    past_shifted_runtimes = [plan.runtime_secs for plan in past_shifted_plans]

    future_shifted_median = np.median(future_shifted_runtimes)
    future_shifted_p90 = np.percentile(future_shifted_runtimes, 90)
    past_shifted_median = np.median(past_shifted_runtimes)
    past_shifted_p90 = np.percentile(past_shifted_runtimes, 90)

    print(f"Past shifted median: {past_shifted_median}")
    print(f"Future shifted median: {future_shifted_median}")
    print(f"Past shifted P90: {past_shifted_p90}")
    print(f"Future shifted P90: {future_shifted_p90}")


@app.command()
def limeqo_compare(until_secs: int):
    print(len(JOB_QUERIES_SORTED))
    print("query_name,cumulative_time,best_runtime")
    for query in JOB_QUERIES_SORTED:
        # print(f"--- {query} ---")
        cumulative_time = 0
        hints = ranked_hint_badness()
        best_runtime = None
        for join_hint, scan_hint in hints:
            runtime_secs = (
                BaoPlan.select(fn.AVG(BaoPlan.runtime_secs))
                .where(
                    (BaoPlan.query_name == query)
                    & (BaoPlan.join_hint == join_hint)
                    & (BaoPlan.scan_hint == scan_hint)
                )
                .scalar()
            )
            if best_runtime is None or runtime_secs < best_runtime:
                best_runtime = runtime_secs
            cumulative_time += best_runtime
            if cumulative_time > until_secs:
                break
            print(f"{query},{cumulative_time},{best_runtime}")
        if cumulative_time > until_secs:
            continue
        # print("--- done bao ---")

        bo_run = best_bayes_series(query, "JOB", {InitType.bao}, cross_joins=False)
        bo_run = [(t + cumulative_time, r) for t, r in bo_run]
        for t, r in bo_run:
            if t > until_secs:
                break
            print(f"{query},{t},{r}")

        # print("=================")


if __name__ == "__main__":
    # wandb.login()
    app()
