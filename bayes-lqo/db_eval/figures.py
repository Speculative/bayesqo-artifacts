import os
import pickle
from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer  # type: ignore
from tqdm import tqdm
from workload.workloads import get_workload_set

from .bao import bao_optimal_time, postgres_time, top_bot_improvement, top_pg_runtime
from .bayes import InitType, best_bayes_series
from .combined import best_at_time
from .random_plans import best_random_series
from .stack import workload_queries as stack_queries
from .utils import AGG_FUNCS, Aggregate, compact_time, pretty_time

app = typer.Typer(no_args_is_help=True)


def load_or_calc(cache_name: str, calc_func):
    if os.path.exists(f"db_eval/figures/{cache_name}.pkl"):
        with open(f"db_eval/figures/{cache_name}.pkl", "rb") as f:
            return pickle.load(f)
    else:
        result = calc_func()
        with open(f"db_eval/figures/{cache_name}.pkl", "wb") as f:
            pickle.dump(result, f)
        return result


def agg_over_time(
    queries: list[str],
    workload_set: str,
    figure_name: str,
    max_time_secs: int = 10 * 60 * 60,
    inc_time_secs: int = 10 * 60,
    include_postgres: bool = False,
    include_random: bool = True,
):
    time_series = np.arange(0, max_time_secs, inc_time_secs)
    postgres_query_times = None
    if include_postgres:
        postgres_query_times = load_or_calc(
            f"{figure_name}_postgres_query_times",
            lambda: [postgres_time(query) for query in queries],
        )
    bao_query_times = load_or_calc(
        f"{figure_name}_bao_query_times",
        lambda: [bao_optimal_time(query, workload_set) for query in queries],
    )
    bayes_query_times = load_or_calc(
        f"{figure_name}_bayes_query_times",
        lambda: [
            best_bayes_series(query, {InitType.bao}, cross_joins=False)
            for query in tqdm(queries, desc="Loading bayes series")
        ],
    )
    random_query_times = None
    if include_random:
        random_query_times = load_or_calc(
            f"{figure_name}_random_query_times",
            lambda: [
                best_random_series(query, max_samples=5000) for query in tqdm(queries, desc="Loading random series")
            ],
        )

    for agg in Aggregate:
        agg_func = AGG_FUNCS[agg]
        if include_postgres:
            pg_line = agg_func(postgres_query_times)
        bao_line = agg_func(bao_query_times)
        if include_random:
            random_line = load_or_calc(
                f"{figure_name}_random_{agg}",
                lambda: [
                    agg_func([best_at_time([query_run], time) for query_run in random_query_times])
                    for time in tqdm(time_series, desc=f"Calculating random {agg}")
                ],
            )
        bayes_line = load_or_calc(
            f"{figure_name}_bayes_{agg}",
            lambda: [
                agg_func([best_at_time([query_run], time) for query_run in bayes_query_times])
                for time in tqdm(time_series, desc=f"Calculating bayes {agg}")
            ],
        )

        balsa_x = None
        balsa_y = None
        try:
            balsa_series = pd.read_csv(f"db_eval/balsa/{figure_name}_{agg}.csv")
            balsa_x = balsa_series["Time (m)"] * 60
            balsa_y = balsa_series["Best latency (s)"]
        except:  # noqa: E722
            pass

        fig = plt.figure()
        plt.grid(axis="y", zorder=0)
        plt.grid(axis="x", zorder=0)
        plt.xticks(ticks=list(range(0, (ceil(max_time_secs / 3600) + 1) * 3600, 3600)))
        plt.gca().set_xlim(0, max_time_secs)
        if include_postgres:
            plt.axhline(pg_line, label="Postgres", linestyle="--", color="orange", zorder=2)
        plt.axhline(bao_line, label="Bao", linestyle="--", color="green", zorder=2)
        if include_random:
            plt.plot(time_series, random_line, label="Random", color="red", zorder=3)
        plt.plot(time_series, bayes_line, label="Bayes", color="blue", zorder=5)
        if balsa_x is not None and balsa_y is not None:
            plt.plot(balsa_x, balsa_y, label="Balsa", color="purple", zorder=4)
        plt.legend()
        plt.title(f"{str(agg).capitalize()} query latency over time ({figure_name})")
        plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: compact_time(x)))
        plt.gca().yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda y, _: compact_time(y)))
        # plt.gca().set_yscale("symlog")
        plt.gca().set_ylim(bottom=0)
        plt.xlabel("Cumulative optimization time (per-query)")
        # plt.ylabel("Plan runtime (log scale)")
        plt.ylabel("Plan runtime")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"db_eval/figures/{figure_name}_{agg}.png", dpi=fig.dpi)


def performance_improvement_distribution(queries: list[str], workload_set: str, figure_name: str):
    # postgres_query_times = load_or_calc(
    #     f"{figure_name}_postgres_query_names_best",
    #     lambda: list(
    #         sorted(
    #             [(query, postgres_time(query)) for query in queries], key=lambda x: x[1]
    #         )
    #     ),
    # )
    bao_query_times = load_or_calc(
        f"{figure_name}_bao_query_names_best",
        lambda: ({query: bao_optimal_time(query, workload_set) for query in queries}),
    )
    bayes_query_times = load_or_calc(
        f"{figure_name}_bayes_query_names_best",
        lambda: (
            {
                query: best_bayes_series(query, {InitType.bao}, cross_joins=False)[-1][1]
                for query in tqdm(queries, desc="Loading bayes series")
            }
        ),
    )
    random_query_times = load_or_calc(
        f"{figure_name}_random_query_names_best",
        lambda: (
            {
                query: best_random_series(query, max_samples=5000)[-1][1]
                for query in tqdm(queries, desc="Loading random series")
            }
        ),
    )
    bayes_improvements = list(
        sorted(
            [(query, bayes_query_times[query] - bao_query_times[query]) for query in queries],
            key=lambda x: x[1],
        )
    )
    random_improvements = list(
        sorted(
            [(query, random_query_times[query] - bao_query_times[query]) for query in queries],
            key=lambda x: x[1],
        )
    )

    xs = list(range(len(queries)))
    fig = plt.figure()

    biggest_improvement = min(x[1] for x in bayes_improvements + random_improvements)

    plt.xticks(xs, [x[0] for x in bayes_improvements], rotation=90)
    plt.grid(axis="y", zorder=0)
    plt.grid(axis="x", zorder=0)

    plt.bar(
        xs,
        [x[1] for x in bayes_improvements],
        align="edge",
        width=1,
        edgecolor="black",
        linewidth=0.1,
        zorder=3,
    )
    plt.title(f"Bayes Improvement over Bao ({figure_name})")
    fig.set_size_inches(20, 10)
    ax = fig.gca()
    ax.set_xlim(-1, len(queries) + 1)
    ax.set_ylim(
        # min(x[1] for x in bayes_improvements), max(x[1] for x in bayes_improvements)
        biggest_improvement,
        0,
    )

    plt.xlabel("Query")
    plt.ylabel("Improvement over Bao (s)")
    plt.savefig(
        f"db_eval/figures/{figure_name}_bayes_improvement.png",
        dpi=fig.dpi,
        bbox_inches="tight",
    )
    plt.tight_layout()
    plt.close()

    fig = plt.figure()

    plt.xticks(xs, [x[0] for x in bayes_improvements], rotation=90)
    plt.grid(axis="y", zorder=0)
    plt.grid(axis="x", zorder=0)

    plt.bar(
        xs,
        [x[1] for x in random_improvements],
        align="edge",
        width=1,
        edgecolor="black",
        linewidth=0.1,
        zorder=3,
    )
    plt.title(f"Random Improvement over Bao ({figure_name})")
    fig.set_size_inches(20, 10)
    ax = fig.gca()
    ax.set_xlim(-1, len(queries) + 1)
    ax.set_ylim(
        # min(x[1] for x in random_improvements),
        # max(x[1] for x in random_improvements),
        biggest_improvement,
        0,
    )

    plt.xlabel("Query")
    plt.ylabel("Improvement over Bao (s)")
    plt.savefig(
        f"db_eval/figures/{figure_name}_random_improvement.png",
        dpi=fig.dpi,
        bbox_inches="tight",
    )
    plt.close()


@app.command()
def ceb_top_comparison():
    top, bot = top_bot_improvement(workload_set="CEB_3K", n=100)
    top_pg = [query for query in top_pg_runtime("CEB_3K", 100) if query != "CEB_11B82"]

    agg_over_time(
        top,
        "CEB_3K",
        "CEB-3K Top 100 Improvement",
    )
    agg_over_time(
        bot,
        "CEB_3K",
        "CEB-3K Bottom 100 Improvement",
    )
    agg_over_time(
        top_pg,
        "CEB_3K",
        "CEB-3K Top 100 PG Runtime",
    )


@app.command()
def job_comparison():
    workload_set = get_workload_set("JOB")
    queries = list(query for query in workload_set.queries.keys())
    agg_over_time(
        queries,
        "JOB",
        "JOB",
        include_postgres=True,
    )
    performance_improvement_distribution(
        queries,
        "JOB",
        "JOB",
    )


@app.command()
def stack_comparison():
    def has_bayes_run(query: str) -> bool:
        try:
            best_bayes_series(query, {InitType.bao}, cross_joins=False)
            return True
        except ValueError:
            return False

    queries = load_or_calc(
        "stack_query_list",
        lambda: list(query for query in stack_queries(target=50, min_runtime=1.0) if has_bayes_run(query)),
    )
    agg_over_time(
        queries,
        "SO_FUTURE",
        "Stack Future",
        include_postgres=True,
        include_random=False,
    )


@app.command()
def random_runtimes():
    # workload_set = get_workload_set("JOB")
    # queries = workload_set.queries.keys()
    top, bot = top_bot_improvement(workload_set="CEB_3K", n=100)
    top_100 = top_pg_runtime("CEB_3K", 100)
    all_queries = list(set(top + bot + top_100))
    for query in all_queries:
        series = best_random_series(query)
        capped_series = best_random_series(query, max_samples=5000)
        total_time = sum(point[1] for point in series)
        capped_total_time = sum(point[1] for point in capped_series)
        print(
            f"{query}: {pretty_time(capped_total_time)}, {len(capped_series)} points, best {pretty_time(capped_series[0][1])}"
        )
        print(
            f"{''.join(' ' for _ in query)}  {pretty_time(total_time)}, {len(series)} points, best {pretty_time(series[0][1])}"
        )


if __name__ == "__main__":
    app()
