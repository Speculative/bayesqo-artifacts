import os
from enum import StrEnum, auto

import pandas as pd
import typer  # type: ignore
import wandb  # type: ignore
from tqdm import tqdm

from .bao import top_bot_improvement, top_pg_runtime
from .stack import workload_queries as stack_queries
from .storage import BayesRun
from .utils import JOB_QUERIES_SORTED

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


@app.command()
def download_runs():
    new_runs = 0
    seen_runs = []
    for wr in completed_runs():
        workload_name = wr.config.get("workload_name")
        run_name = wr.name
        seen_runs.append((run_name, workload_name))

        target_dir = os.path.join(
            os.path.dirname(__file__),
            f"bayes/{workload_name}",
        )
        os.makedirs(target_dir, exist_ok=True)

        workload_set = None
        if workload_name.startswith("STACK_"):
            if wr.config.get("so_future"):
                workload_set = "SO_FUTURE"
            else:
                workload_set = "SO_PAST"
        else:
            if workload_name.startswith("JOB_"):
                workload_set = "JOB"
            elif workload_name.startswith("CEB_"):
                workload_set = "CEB_3K"

        tqdm.write(workload_name)

        if (
            BayesRun.select()
            .where(
                (BayesRun.query_name == workload_name) & (BayesRun.run_name == run_name)
            )
            .exists()
        ):
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

        history = list(
            wr.scan_history(
                keys=["non_parallel_runtime", "best_found"], page_size=100_000
            )
        )
        length = len(history)

        language = wr.config.get("which_query_language")

        tqdm.write(f"Downloading {run_name} ({workload_name}, {init}, {length})")

        target_file = os.path.join(target_dir, f"{run_name}.csv")
        if os.path.exists(target_file):
            tqdm.write(f"Skipping downloading {target_file}")
        else:
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

    tqdm.write(f"Downloaded {new_runs} new runs")


class InitType(StrEnum):
    random = auto()
    bao = auto()
    llm = auto()
    unknown = auto()


def bayes_series(
    query: str,
    init: set[InitType],
    cross_joins: bool = False,
) -> list[list[tuple[float, float]]]:
    """Produces a list of all salient optimization points in a given
    optimization run, i.e. each point where runtime improved.

    Each tuple is (cumulative optimization time, plan runtime)
    """

    run_paths = BayesRun.select(BayesRun.log_path).where(
        (BayesRun.query_name == query)
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


def best_bayes_series(query: str, init: set[InitType], cross_joins: bool = False):
    runs = bayes_series(query, init, cross_joins)
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

    so_queries = set(stack_queries(target=50, min_runtime=1.0))
    print()
    print("Missing for SO_PAST:")
    missing = 0
    for query in so_queries:
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
    print(f"Missing {missing} of {len(so_queries)}")

    print()
    print("Missing for SO_FUTURE:")
    missing = 0
    for query in so_queries:
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
    print(f"Missing {missing} of {len(so_queries)}")


if __name__ == "__main__":
    wandb.login()
    app()
