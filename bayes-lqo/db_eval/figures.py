import json
import os
import pdb
import pickle
from math import ceil
from typing import Callable, Literal, Optional, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer  # type: ignore
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from peewee import Select, fn
from tqdm import tqdm

from workload.workloads import get_workload_set

from .balsa import balsa_series
from .bao import (
    bao_optimal_time,
    limeqo_hint_order,
    postgres_time,
    ranked_hint_badness,
    top_bot_improvement,
    top_pg_runtime,
)
from .bayes import (
    InitType,
    bayes_series,
    best_bayes_series,
    get_bayes_run_history,
    get_optimal_plan,
)
from .combined import best_at_time
from .eval_workloads import (
    EvalWorkload,
    nice_title,
    resolve_eval_queries,
    resolve_workload_set,
)
from .random_plans import all_random_series, best_random_series
from .stack import workload_queries as stack_queries
from .storage import (
    BaoJoinHint,
    BaoPlan,
    BaoScanHint,
    BayesRun,
    BayesValidationRun,
    PostgresPlan,
    RandomPlan,
    StackShiftedOptimizedPlan,
    db,
)
from .utils import AGG_FUNCS, Aggregate, compact_time, pretty_time

app = typer.Typer(no_args_is_help=True)


TReturn = TypeVar("TReturn")


def load_or_calc(cache_name: str, calc_func: Callable[[], TReturn]) -> TReturn:
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
            best_bayes_series(query, workload_set, {InitType.bao}, cross_joins=False)
            for query in tqdm(queries, desc="Loading bayes series")
        ],
    )
    random_query_times = None
    if include_random:
        random_query_times = load_or_calc(
            f"{figure_name}_random_query_times",
            lambda: [
                best_random_series(query, max_samples=5000)
                for query in tqdm(queries, desc="Loading random series")
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
                    agg_func(
                        [
                            best_at_time([query_run], time)
                            for query_run in random_query_times
                        ]
                    )
                    for time in tqdm(time_series, desc=f"Calculating random {agg}")
                ],
            )
        bayes_line = load_or_calc(
            f"{figure_name}_bayes_{agg}",
            lambda: [
                agg_func(
                    [best_at_time([query_run], time) for query_run in bayes_query_times]
                )
                for time in tqdm(time_series, desc=f"Calculating bayes {agg}")
            ],
        )

        balsa_x = None
        balsa_y = None
        try:
            balsa_series = pd.read_csv(f"db_eval/balsa/{figure_name}_{agg}.csv")
            balsa_x = balsa_series["Time (m)"] * 60
            balsa_y = balsa_series["Best latency (s)"]
        except:
            pass

        fig = plt.figure()
        plt.grid(axis="y", zorder=0)
        plt.grid(axis="x", zorder=0)
        plt.xticks(ticks=list(range(0, (ceil(max_time_secs / 3600) + 1) * 3600, 3600)))
        plt.gca().set_xlim(0, max_time_secs)
        if include_postgres:
            plt.axhline(
                pg_line, label="Postgres", linestyle="--", color="orange", zorder=2
            )
        plt.axhline(bao_line, label="Bao", linestyle="--", color="green", zorder=2)
        if include_random:
            plt.plot(time_series, random_line, label="Random", color="red", zorder=3)
        plt.plot(time_series, bayes_line, label="Bayes", color="blue", zorder=5)
        if balsa_x is not None and balsa_y is not None:
            plt.plot(balsa_x, balsa_y, label="Balsa", color="purple", zorder=4)
        plt.legend()
        plt.title(f"{str(agg).capitalize()} query latency over time ({figure_name})")
        plt.gca().xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: compact_time(x))
        )
        plt.gca().yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda y, _: compact_time(y))
        )
        # plt.gca().set_yscale("symlog")
        plt.gca().set_ylim(bottom=0)
        plt.xlabel("Cumulative optimization time (per-query)")
        # plt.ylabel("Plan runtime (log scale)")
        plt.ylabel("Plan runtime")
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"db_eval/figures/{figure_name}_{agg}.png", dpi=fig.dpi)


def performance_improvement_distribution(
    queries: list[str], workload_set: str, figure_name: str
):
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
                query: best_bayes_series(
                    query, workload_set, {InitType.bao}, cross_joins=False
                )[-1][1]
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
            [
                (query, bayes_query_times[query] - bao_query_times[query])
                for query in queries
            ],
            key=lambda x: x[1],
        )
    )
    random_improvements = list(
        sorted(
            [
                (query, random_query_times[query] - bao_query_times[query])
                for query in queries
            ],
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
    queries = load_or_calc(
        "stack_query_list",
        lambda: list(query for query in stack_queries(target=200, min_runtime=1.0)),
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


@app.command()
def time_boxplots(eval_workload: EvalWorkload = typer.Option()):
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    bao_query_times = load_or_calc(
        f"{eval_workload}_bao_boxplot.times",
        lambda: {
            query: bao_optimal_time(query, workload_set_name) for query in queries
        },
    )
    bayes_query_series = load_or_calc(
        f"{eval_workload}_bayes_boxplot.times",
        lambda: (
            {
                query: best_bayes_series(
                    query, workload_set_name, {InitType.bao}, cross_joins=False
                )
                for query in tqdm(queries, desc="Loading bayes series")
            }
        ),
    )

    times = [mins * 60 for mins in range(10, 6 * 60 + 10, 10)]
    bayes_time_improvements = load_or_calc(
        f"{eval_workload}_bayes_boxplot.figure",
        lambda: [
            [
                min(
                    (best_at_time([bayes_series], time) or bao_query_times[query])
                    / bao_query_times[query],
                    1.0,
                )
                for query, bayes_series in bayes_query_series.items()
            ]
            for time in tqdm(times, desc="Calculating bayes improvements per time")
        ],
    )

    fig = plt.figure()
    plt.boxplot(
        bayes_time_improvements,
        labels=[compact_time(time) if time % 3600 == 0 else "" for time in times],
    )
    plt.title(f"Bayes time improvement over Bao ({nice_title(eval_workload)})")
    plt.xlabel("Optimization Time (per-query)")
    plt.ylabel("Improvement factor")
    plt.tight_layout()
    plt.savefig(f"db_eval/figures/{eval_workload}_boxplot.png", dpi=fig.dpi)


def measure_text(ax, text, fontsize):
    f = plt.figure(figsize=(12, 6))
    r = f.canvas.get_renderer()
    t = plt.text(0.5, 0.5, text, fontsize=fontsize, rotation=90)

    bb = t.get_window_extent(renderer=r).transformed(ax.transData.inverted())
    width = bb.width
    height = bb.height
    plt.close(f)
    return (width, height)


def autolabel(ax, bars, ylim, fontsize=None, ymin=0):
    for bar in bars:
        height = bar.get_height()
        label = f"{round(height, 2)}"
        label_position_y = max(
            ymin,
            min(
                height + (ylim * 0.02),
                # Subtract width (because label is rotated 90 degrees)
                (ylim * 0.98) - measure_text(ax, label, fontsize or 8)[1],
            ),
        )  # Keep label within chart boundary

        # If the bar is taller than our ylim, adjust label position and add arrow
        if height > ylim:
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, ylim),
                xytext=(bar.get_x() + bar.get_width() / 2, label_position_y),
                textcoords="data",
                ha="center",
                va="bottom",
                arrowprops=dict(arrowstyle="-|>", color="black"),
                rotation=90,
                fontsize=fontsize,
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_position_y,
                label,
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=fontsize,
            )


@app.command()
def time_agg_bars(eval_workload: EvalWorkload = typer.Option()):
    # Plot sum, median, p90 for pg vs. bao vs. bayes opt at time
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    include_postgres = eval_workload != EvalWorkload.CEB

    # time_series = [60 * 60, 3 * 60 * 60, 6 * 60 * 60]
    time_series = [60 * 60, 6 * 60 * 60]
    postgres_query_times = None
    postgres_query_times = load_or_calc(
        f"{eval_workload}_postgres.times",
        lambda: [postgres_time(query, workload_set_name) for query in queries],
    )
    bao_query_times = load_or_calc(
        f"{eval_workload}_bao.times",
        lambda: [bao_optimal_time(query, workload_set_name) for query in queries],
    )
    bayes_query_times = load_or_calc(
        f"{eval_workload}_bayes.times",
        lambda: [
            best_bayes_series(
                query, workload_set_name, {InitType.bao}, cross_joins=False
            )
            for query in tqdm(queries, desc="Loading bayes series")
        ],
    )
    # balsa_query_times = load_or_calc(
    #     f"{eval_workload}_balsa.times",
    #     lambda: [
    #         balsa_series(query) for query in tqdm(queries, desc="Loading balsa series")
    #     ],
    # )
    random_query_times = load_or_calc(
        f"{eval_workload}_random.times",
        lambda: [
            best_random_series(query)
            for query in tqdm(queries, desc="Loading random series")
        ],
    )

    group_labels = (
        (["Postgres"] if include_postgres else [])
        + ["Bao"]
        + [f"BO {pretty_time(time)}" for time in time_series]
        # + [f"Balsa {pretty_time(time)}" for time in time_series]
        + [f"Rand {pretty_time(time)}" for time in time_series]
    )

    aggs = [Aggregate.median, Aggregate.avg, Aggregate.p90]

    def calc_bars():
        bars = []
        for agg in aggs:
            agg_func = AGG_FUNCS[agg]
            agg_bars = (
                [agg_func(postgres_query_times)] if include_postgres else []
            ) + [agg_func(bao_query_times)]
            for time in time_series:
                agg_bars.append(
                    agg_func(
                        [
                            best_at_time([query_run], time)
                            for query_run in bayes_query_times
                        ]
                    )
                )
            # for time in time_series:
            #     agg_bars.append(
            #         agg_func(
            #             [
            #                 best_at_time([query_run], time)
            #                 for query_run in balsa_query_times
            #             ]
            #         )
            #     )
            for time in time_series:
                agg_bars.append(
                    agg_func(
                        [
                            best_at_time([query_run], time)
                            for query_run in random_query_times
                        ]
                    )
                )
            bars.append(agg_bars)
        return bars

    bars = load_or_calc(
        f"{eval_workload}_bayes_time_agg_bars.figure",
        calc_bars,
    )

    # Bar chart grouped by time
    # MARGIN = 0.1
    # BAR_WIDTH = (1.0 - 2.0 * MARGIN) / len(group_labels)
    BAR_WIDTH = 0.25
    fig = plt.figure()
    ax = fig.gca()
    xs = np.arange(len(group_labels))
    plt.grid(axis="y")

    group_rects = []
    for i, agg in enumerate(aggs):
        rects = ax.bar(
            xs + i * BAR_WIDTH,
            bars[i],
            label=str(agg).capitalize(),
            width=BAR_WIDTH,
            zorder=3,
            edgecolor="black",
        )
        group_rects.append(rects)
        # ax.bar_label(
        #     rects, [f"{x:.2f}" for x in bars[i]], padding=3, zorder=4, rotation=90
        # )

    if eval_workload == EvalWorkload.CEB:
        ax.set_ylim(bottom=0, top=10)
    else:
        # Increase ylim for labels
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    ylim = ax.get_ylim()[1]
    for rects in group_rects:
        autolabel(ax, rects, ylim)

    ax.set_xticks(xs + BAR_WIDTH, group_labels)
    ax.set_ylabel("Query runtime (secs)")
    ax.legend()
    # plt.title(f"Query runtime by optimization method ({nice_title(eval_workload)})")
    plt.tight_layout()

    # plt.show()
    plt.savefig(f"db_eval/figures/{eval_workload}_time_agg_bars.png", dpi=fig.dpi)


@app.command()
def oracle_calls_agg_bars(eval_workload: EvalWorkload = typer.Option()):
    # Plot sum, median, p90 for pg vs. bao vs. bayes opt at time
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    include_postgres = eval_workload != EvalWorkload.CEB

    # time_series = [60 * 60, 3 * 60 * 60, 6 * 60 * 60]
    n_calls_series = list(range(1000, 6000, 2000))
    postgres_query_times = None
    postgres_query_times = load_or_calc(
        f"{eval_workload}_postgres.times",
        lambda: [postgres_time(query, workload_set_name) for query in queries],
    )
    bao_query_times = load_or_calc(
        f"{eval_workload}_bao.times",
        lambda: [bao_optimal_time(query, workload_set_name) for query in queries],
    )

    def load_bayes_oracle_calls_times(
        query: str, workload_set: str
    ) -> list[tuple[int, float]]:
        run_path = (
            BayesRun.select(BayesRun.log_path)
            .where(
                (BayesRun.query_name == query) & (BayesRun.workload_set == workload_set)
            )
            .first()
        )
        run_df = pd.read_csv(run_path.log_path)
        run_points = []
        oracle_calls = -1
        for _, row in run_df.dropna(
            subset=["non_parallel_runtime", "best_found"]
        ).iterrows():
            if row["n_oracle_calls"] > oracle_calls:
                oracle_calls = row["n_oracle_calls"]
                run_points.append((oracle_calls, -1 * row["best_found"]))
        return run_points

    bayes_oracle_calls_times = load_or_calc(
        f"{eval_workload}_bayes.calls",
        lambda: [
            load_bayes_oracle_calls_times(query, workload_set_name)
            for query in tqdm(queries, desc="Loading bayes series")
        ],
    )

    def load_random_oracle_calls_times(
        query: str, workload_set: str
    ) -> list[tuple[int, float]]:
        plans = (
            RandomPlan.select()
            .where(
                (RandomPlan.query_name == query)
                & (RandomPlan.workload_set == workload_set)
            )
            .order_by(RandomPlan.plan_id)
        )
        oracle_calls = -1
        run_points = []
        for plan in plans:
            oracle_calls += 1
            run_points.append((oracle_calls, plan.runtime_secs))
        return run_points

    random_oracle_calls_times = load_or_calc(
        f"{eval_workload}_random.calls",
        lambda: [
            load_random_oracle_calls_times(query, workload_set_name)
            for query in tqdm(queries, desc="Loading random series")
        ],
    )

    group_labels = (
        (["Postgres"] if include_postgres else [])
        + ["Bao"]
        + [f"BO {n_calls}" for n_calls in n_calls_series]
        + [f"Rand {n_calls}" for n_calls in n_calls_series]
    )

    aggs = [Aggregate.median, Aggregate.avg, Aggregate.p90]

    def calc_bars():
        bars = []
        for agg in aggs:
            agg_func = AGG_FUNCS[agg]
            agg_bars = (
                [agg_func(postgres_query_times)] if include_postgres else []
            ) + [agg_func(bao_query_times)]
            for n_calls in n_calls_series:
                agg_bars.append(
                    agg_func(
                        [
                            min(
                                runtime
                                for calls, runtime in query_run
                                if calls <= n_calls
                            )
                            for query_run in bayes_oracle_calls_times
                        ]
                    )
                )
            for n_calls in n_calls_series:
                agg_bars.append(
                    agg_func(
                        [
                            min(
                                runtime
                                for calls, runtime in query_run
                                if calls <= n_calls
                            )
                            for query_run in random_oracle_calls_times
                        ]
                    )
                )
            bars.append(agg_bars)
        return bars

    bars = load_or_calc(
        f"{eval_workload}_bayes_oracle_calls_agg_bars.figure",
        calc_bars,
    )

    # Bar chart grouped by time
    # MARGIN = 0.1
    # BAR_WIDTH = (1.0 - 2.0 * MARGIN) / len(group_labels)
    BAR_WIDTH = 0.25
    fig = plt.figure()
    ax = fig.gca()
    xs = np.arange(len(group_labels))
    plt.grid(axis="y")

    def autolabel(bars, ylim):
        for bar in bars:
            height = bar.get_height()
            label_position_y = min(
                height + (ylim * 0.02), ylim * 0.9
            )  # Keep label within chart boundary

            # If the bar is taller than our ylim, adjust label position and add arrow
            if height > ylim:
                ax.annotate(
                    f"{round(height, 2)}",
                    xy=(bar.get_x() + bar.get_width() / 2, ylim),
                    xytext=(bar.get_x() + bar.get_width() / 2, label_position_y),
                    textcoords="data",
                    ha="center",
                    va="bottom",
                    arrowprops=dict(arrowstyle="-|>", color="black"),
                    rotation=90,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_position_y,
                    f"{round(height, 2)}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                )

    group_rects = []
    for i, agg in enumerate(aggs):
        rects = ax.bar(
            xs + i * BAR_WIDTH,
            bars[i],
            label=str(agg).capitalize(),
            width=BAR_WIDTH,
            zorder=3,
            edgecolor="black",
        )
        group_rects.append(rects)
        # ax.bar_label(
        #     rects, [f"{x:.2f}" for x in bars[i]], padding=3, zorder=4, rotation=90
        # )

    # if eval_workload == EvalWorkload.CEB:
    #     ax.set_ylim(bottom=0, top=10)
    # else:
    #     # Increase ylim for labels
    #     ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    ylim = ax.get_ylim()[1]
    for rects in group_rects:
        autolabel(rects, ylim)

    ax.set_xticks(xs + BAR_WIDTH, group_labels, rotation=45)
    ax.set_ylabel("Query runtime (secs)")

    ax.legend()
    plt.title(f"Query runtime by optimization method ({nice_title(eval_workload)})")
    plt.tight_layout()

    # plt.show()
    plt.savefig(
        f"db_eval/figures/{eval_workload}_oracle_calls_agg_bars.png", dpi=fig.dpi
    )


@app.command()
def eval_query_runtimes():
    print("query_name,pg_runtime_secs,bao_runtime_secs,bayes_optimal_runtime_secs")

    def print_query_runtimes(query: str, workload_set: str):
        pg_runtime = None
        bao_runtime = None
        bayes_optimal_runtime = None
        try:
            pg_runtime = postgres_time(query, workload_set)
            bao_runtime = bao_optimal_time(query, workload_set)
            bayes_optimal_runtime = best_bayes_series(
                query, workload_set, {InitType.bao}, cross_joins=False
            )[-1][1]
        except:
            pass
        print(f"{query},{pg_runtime},{bao_runtime},{bayes_optimal_runtime}")

    for query in resolve_eval_queries(EvalWorkload.JOB):
        print_query_runtimes(query, resolve_workload_set(EvalWorkload.JOB))

    ceb_queries = set()
    ceb_queries.update(resolve_eval_queries(EvalWorkload.CEB_PG))
    ceb_queries.update(resolve_eval_queries(EvalWorkload.CEB_TOP))
    ceb_queries.update(resolve_eval_queries(EvalWorkload.CEB_BOT))
    for query in sorted(ceb_queries):
        print_query_runtimes(query, resolve_workload_set(EvalWorkload.CEB_PG))

    for query in resolve_eval_queries(EvalWorkload.STACK_200):
        print_query_runtimes(query, resolve_workload_set(EvalWorkload.STACK_200))


@app.command()
def trials_to_first_improvement(eval_workload: EvalWorkload = typer.Option()):
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    trials_until_success = []
    for query in tqdm(queries):
        run = (
            RandomPlan.select()
            .where(
                (RandomPlan.query_name == query)
                & (RandomPlan.workload_set == workload_set_name)
            )
            .order_by(RandomPlan.plan_id)
        )
        for i, plan in enumerate(run):
            if plan.result == "success":
                trials_until_success.append(i)
                break
    print(f"Mean trials until success: {np.mean(trials_until_success)}")
    print(f"Median trials until success: {np.median(trials_until_success)}")
    print(f"P90 trials until success: {np.percentile(trials_until_success, 90)}")


@app.command()
def query_space_size(eval_workload: EvalWorkload = typer.Option()):
    workload_set_name = resolve_workload_set(eval_workload)
    workload_set = get_workload_set(workload_set_name)
    query_space_sizes = []
    for query in workload_set.queries.values():
        num_tables = sum(alias_num for _, alias_num in query.query_tables)
        query_space_sizes.append(num_tables)
    print(f"Mean tables: {np.mean(query_space_sizes)}")
    print(f"Median tables: {np.median(query_space_sizes)}")
    print(f"P90 tables: {np.percentile(query_space_sizes, 90)}")
    print(f"Max tables: {max(query_space_sizes)}")


@app.command()
def sorted_improvement(eval_workload: EvalWorkload = typer.Option()):
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    postgres_query_times_map = load_or_calc(
        f"{eval_workload}_postgres_map.times",
        lambda: {query: postgres_time(query, workload_set_name) for query in queries},
    )
    bayes_query_times_map = load_or_calc(
        f"{eval_workload}_bayes_best_map.times",
        lambda: (
            {
                query: best_bayes_series(
                    query, workload_set_name, {InitType.bao}, cross_joins=False
                )[-1][1]
                for query in tqdm(queries, desc="Loading bayes series")
            }
        ),
    )
    random_query_times = load_or_calc(
        f"{eval_workload}_random_query_names_best.times",
        lambda: (
            {
                query: best_random_series(query)[-1][1]
                for query in tqdm(queries, desc="Loading random series")
            }
        ),
    )

    bayes_improvements = list(
        sorted(
            [
                -1
                + min(bayes_query_times_map[query] / postgres_query_times_map[query], 1)
                for query in queries
            ],
        )
    )
    random_improvements = list(
        sorted(
            [
                -1 + min(random_query_times[query] / postgres_query_times_map[query], 1)
                for query in queries
            ],
        )
    )
    # 2 vertically stacked subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.bar(list(range(len(queries))), bayes_improvements)
    ax1.set_title("Bayes improvement over Postgres")
    ax1.set_ylabel("Improvement %")
    ax1.set_xlabel("Query")
    ax1.set_xticks([])
    ax1.grid(axis="y")

    ax2.bar(list(range(len(queries))), random_improvements)
    ax2.set_title("Random improvement over Postgres")
    ax2.set_ylabel("Improvement %")
    ax2.set_xlabel("Query")
    ax2.set_xticks([])
    ax2.grid(axis="y")

    plt.tight_layout()
    fig.suptitle(f"Improvement over Postgres ({nice_title(eval_workload)})")
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"db_eval/figures/{eval_workload}_sorted_improvement.png", dpi=fig.dpi)


@app.command()
def best_plans_at_times(eval_workload: EvalWorkload = typer.Option()):
    times = [60 * 60, 3 * 60 * 60, 6 * 60 * 60]
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    query_times: dict[
        str,
        dict[str, dict[Literal["bayes", "balsa", "random", "bao"], float]],
    ] = {}
    for query in tqdm(queries):
        query_results = query_times.setdefault(query, {})
        bao_time = bao_optimal_time(query, workload_set_name)
        bayes_times = best_bayes_series(
            query, workload_set_name, {InitType.bao}, cross_joins=False
        )
        random_times = best_random_series(query)
        balsa_times = balsa_series(query)

        for time in times:
            query_results[pretty_time(time)] = {
                "bao": bao_time,
                "bayes": best_at_time(
                    [bayes_times],
                    time,
                )
                or -1,
                "balsa": best_at_time([balsa_times], time) or -1,
                "random": best_at_time([random_times], time) or -1,
            }
    print(json.dumps(query_times, indent=2))


@app.command()
def single_query(
    query_name: str = typer.Option(),
    workload_set_name: str = typer.Option(),
    top: Optional[float] = None,
    right: Optional[float] = None,
    legend: bool = False,
):
    bao_time = bao_optimal_time(query_name, workload_set_name)
    full_bayes_run = get_bayes_run_history(query_name, workload_set_name)
    if full_bayes_run is None:
        print("No Bayes run found")
        return
    full_bayes_run = full_bayes_run.dropna(
        subset=["non_parallel_runtime", "max_score_most_recent_batch"]
    )
    full_bayes_run["max_score_most_recent_batch"] *= -1
    full_bayes_run["non_parallel_runtime"] *= 60 * 60

    bayes_improvements = best_bayes_series(
        query_name, workload_set_name, {InitType.bao}, cross_joins=False
    )
    balsa_improvements = balsa_series(query_name)
    random_improvements = best_random_series(query_name)

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})
    plt.figure(figsize=(12, 6))
    plt.grid(axis="y", ls="--")
    plt.grid(axis="x", ls="--")

    plt.axhline(bao_time, label="Bao", linestyle="--", color="tab:red")
    plt.plot(
        full_bayes_run["non_parallel_runtime"],
        full_bayes_run["max_score_most_recent_batch"],
        label="Bayes (latest)",
        color="lightblue",
        zorder=0,
        # alpha=0.5,
    )
    plt.plot(
        [point[0] for point in bayes_improvements],
        [point[1] for point in bayes_improvements],
        label="Bayes (best)",
        color="tab:blue",
    )
    plt.plot(
        [point[0] for point in balsa_improvements],
        [point[1] for point in balsa_improvements],
        label="Balsa",
        color="tab:orange",
    )
    plt.plot(
        [point[0] for point in random_improvements],
        [point[1] for point in random_improvements],
        label="Random",
        color="tab:green",
    )

    if legend:
        plt.legend(fontsize=24)

    plt.xticks(
        ticks=list(
            range(
                0,
                (
                    int(right) + 1
                    if right is not None
                    else (
                        (ceil(full_bayes_run["non_parallel_runtime"].max() / 3600) + 1)
                        * 3600
                    )
                ),
                3600,
            )
        )
    )
    plt.xlim(left=0, right=right)
    plt.ylim(bottom=0, top=top)
    plt.gca().xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: compact_time(x))
    )

    # plt.grid(axis="y", zorder=0)
    # plt.grid(axis="x", zorder=0)
    # plt.title(f"{query_name} ({workload_set_name})")
    plt.xlabel("Optimization time")
    plt.ylabel("Runtime (s)")

    # plt.savefig(f"db_eval/figures/{query_name}.png")
    plt.savefig(f"db_eval/figures/{query_name}.pdf")


@app.command()
def plan_degradation():
    past_plans = BayesValidationRun.select().where(
        BayesValidationRun.workload_set == "SO_RETRO"
    )
    queries = [past_plan.query_name for past_plan in past_plans]

    def get_future_runtimes():
        future_runtimes: dict[str, float] = {}
        for query in tqdm(queries):
            bayes_run = best_bayes_series(
                query, "SO_FUTURE", {InitType.bao}, cross_joins=False
            )
            future_runtimes[query] = bayes_run[-1][1]
        return future_runtimes

    future_runtimes = load_or_calc("future_runtimes", get_future_runtimes)
    degradations = sorted(
        [
            past_plan.runtime_secs / future_runtimes[past_plan.query_name]
            for past_plan in past_plans
        ]
    )
    negative_bars = [(d - 1) * 100 for d in degradations if d < 1]
    positive_bars = [(d - 1) * 100 for d in degradations if d >= 1]

    # fig = plt.figure()
    # rects = plt.bar(list(range(len(degradations))), degradations)
    # ax = plt.gca()
    # ylim = ax.get_ylim()[1]
    # autolabel(ax, rects, ylim, fontsize=6)
    # fig = plt.figure()
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})
    plt.figure(figsize=(12, 6))

    neg_rects = plt.bar(
        list(range(len(negative_bars))), negative_bars, color="tab:blue"
    )
    pos_rects = plt.bar(
        list(range(len(negative_bars), len(degradations))),
        positive_bars,
        color="tab:orange",
    )
    ax = plt.gca()
    # ylim = ax.get_ylim()[1]
    ylim = 100
    plt.ylim(-100, ylim)
    plt.xlim(-0.5, len(degradations) - 0.5)
    autolabel(ax, neg_rects, ylim, fontsize=12, ymin=2)
    autolabel(ax, pos_rects, ylim, fontsize=12, ymin=2)

    # plt.title("Plan degradation")
    plt.xlabel("Query")
    plt.ylabel("Percentage Difference (%)")
    # plt.tight_layout()
    plt.grid(axis="y", ls="--")
    plt.xticks([])
    # plt.savefig("db_eval/figures/plan_degradation.png", dpi=300)
    plt.savefig("db_eval/figures/plan_degradation.pdf")


@app.command()
def reoptimize_comparison():
    FromScratchBayesRun = BayesRun.alias()
    comparisons = (
        BayesRun.select(
            BayesRun.query_name,
            BayesRun.log_path.alias("reoptimize_path"),
            FromScratchBayesRun.log_path.alias("from_scratch_path"),
        )
        .join(
            FromScratchBayesRun,
            on=(BayesRun.query_name == FromScratchBayesRun.query_name),
            attr="from_scratch",
        )
        .where(
            (BayesRun.workload_set == "SO_RETRO")
            & (FromScratchBayesRun.workload_set == "SO_FUTURE")
        )
    )
    reoptimize_times = []
    comparison_ratios = []
    from_scratch_times = []
    for comparison in tqdm(comparisons):
        reoptimize_run = pd.read_csv(comparison.reoptimize_path)
        from_scratch_run = pd.read_csv(comparison.from_scratch.from_scratch_path)
        reoptimize_end = reoptimize_run.dropna(
            subset=["non_parallel_runtime", "best_found"]
        ).iloc[-1]
        from_scratch_end = from_scratch_run.dropna(
            subset=["non_parallel_runtime", "best_found"]
        ).iloc[-1]

        best_bao_time = bao_optimal_time(comparison.query_name, "SO_FUTURE")

        reoptimize_times.append(reoptimize_end["non_parallel_runtime"])
        from_scratch_times.append(from_scratch_end["non_parallel_runtime"])
        comparison_ratios.append(
            (
                (
                    min(-1 * reoptimize_end["best_found"], best_bao_time)
                    / min(-1 * from_scratch_end["best_found"], best_bao_time)
                )
                - 1
            )
            * 100
        )

    print()
    print(f"Median reoptimize time: {round(np.median(reoptimize_times), 2)} hours")
    print(f"Average reoptimize time: {round(np.mean(reoptimize_times), 2)} hours")
    print(f"Stddev reoptimize time: {round(np.std(reoptimize_times), 2)}")
    print(f"Median from scratch time: {round(np.median(from_scratch_times), 2)}")
    print(f"Average from scratch time: {round(np.mean(from_scratch_times), 2)} hours")
    print(f"Stddev from scratch time: {round(np.std(from_scratch_times), 2)}")

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})
    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        list(range(len(comparison_ratios))),
        list(sorted(comparison_ratios)),
    )
    for bar in bars:
        if bar.get_height() < 0:
            bar.set_color("tab:blue")
        else:
            bar.set_color("tab:orange")
    # plt.title("Reoptimize comparison")
    plt.xlabel("Query")
    plt.ylabel("Percentage Difference (%)")
    # plt.tight_layout()
    plt.grid(axis="y", ls="--")
    plt.xlim(-0.5, len(comparison_ratios) - 0.5)
    plt.ylim(-100, 100)
    plt.xticks([])

    autolabel(plt.gca(), bars, 100, fontsize=12, ymin=2)
    # plt.savefig("db_eval/figures/reoptimize_comparison.png", dpi=300)
    plt.savefig("db_eval/figures/reoptimize_comparison.pdf")


@app.command()
def vae_drift_comparison():
    RetrainedBayesRun = BayesRun.alias()
    comparisons = (
        BayesRun.select(
            BayesRun.query_name,
            BayesRun.log_path.alias("drifted_vae_path"),
            RetrainedBayesRun.log_path.alias("retrained_vae_path"),
        )
        .join(
            RetrainedBayesRun,
            on=(BayesRun.query_name == RetrainedBayesRun.query_name),
            attr="retrained",
        )
        .where(
            (BayesRun.workload_set == "SO_DRIFT")
            & (RetrainedBayesRun.workload_set == "SO_FUTURE")
        )
    )
    drifted_times = []
    retrained_times = []
    comparison_ratios = []
    for comparison in tqdm(comparisons):
        drifted_run = pd.read_csv(comparison.drifted_vae_path)
        retrained_run = pd.read_csv(comparison.retrained.retrained_vae_path)
        drifted_end = drifted_run.dropna(
            subset=["non_parallel_runtime", "best_found"]
        ).iloc[-1]
        retrained_end = retrained_run.dropna(
            subset=["non_parallel_runtime", "best_found"]
        ).iloc[-1]

        best_bao_time = bao_optimal_time(comparison.query_name, "SO_FUTURE")

        drifted_times.append(drifted_end["non_parallel_runtime"])
        retrained_times.append(retrained_end["non_parallel_runtime"])
        comparison_ratios.append(
            (
                (
                    min(-1 * drifted_end["best_found"], best_bao_time)
                    / min(-1 * retrained_end["best_found"], best_bao_time)
                )
                - 1
            )
            * 100
        )

    print()
    print(f"Median drifted time: {round(np.median(drifted_times), 2)} hours")
    print(
        f"Average drifted optimization time: {round(np.mean(drifted_times), 2)} hours"
    )
    print(f"Stddev drifted time: {round(np.std(drifted_times), 2)}")
    print(f"Median retrained time: {round(np.median(retrained_times), 2)} hours")
    print(f"Average retrained time: {round(np.mean(retrained_times), 2)} hours")
    print(f"Stddev retrained time: {round(np.std(retrained_times), 2)}")

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})
    plt.figure(figsize=(12, 6))

    bars = plt.bar(
        list(range(len(comparison_ratios))),
        list(sorted(comparison_ratios)),
    )
    for bar in bars:
        if bar.get_height() < 0:
            bar.set_color("tab:blue")
        else:
            bar.set_color("tab:orange")
    plt.ylim(-100, 100)
    plt.xlim(-0.5, len(comparison_ratios) - 0.5)
    autolabel(plt.gca(), bars, 100, fontsize=12, ymin=2)
    # plt.title("VAE drift comparison")
    plt.xlabel("Query")
    plt.ylabel("Percentage Difference (%)")
    # plt.tight_layout()
    plt.grid(axis="y", ls="--")
    plt.xticks([])
    # plt.savefig("db_eval/figures/vae_drift_comparison.png", dpi=300)
    plt.savefig("db_eval/figures/vae_drift_comparison.pdf")


@app.command()
def compare_llm():
    llm_results = pd.read_csv("db_eval/llm_results.csv")
    comparisons = []
    for row in tqdm(llm_results.itertuples()):
        query_name = row.task
        llm_runtime = row.gpt4o_top_50
        try:
            bo_runtime = best_bayes_series(
                query_name, "SO_FUTURE", {InitType.bao}, cross_joins=False
            )[-1][1]
            comparisons.append(((llm_runtime / bo_runtime) - 1) * 100)
        except:
            tqdm.write("No Bayes run found for " + query_name)

    plt.bar(list(range(len(comparisons))), list(sorted(comparisons)))
    plt.show()


@app.command()
def drift_figure():
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})

    # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))

    # ====================
    # Aggregate bar charts
    # ====================
    def calc_reoptimize_series():
        past_plans = BayesValidationRun.select().where(
            BayesValidationRun.workload_set == "SO_RETRO"
        )
        queries = [past_plan.query_name for past_plan in past_plans]

        bao_times = [
            bao_optimal_time(query.query_name, "SO_FUTURE") for query in past_plans
        ]
        past_times = [past_plan.runtime_secs for past_plan in past_plans]
        # from_scratch_times = [
        #     min(
        #         best_bayes_series(
        #             query, "SO_FUTURE", {InitType.bao}, cross_joins=False
        #         )[-1][1],
        #         bao_optimal_time(query, "SO_FUTURE"),
        #     )
        #     for query in queries
        # ]
        FromScratchBayesRun = BayesRun.alias()
        comparisons = (
            BayesRun.select(
                BayesRun.query_name,
                BayesRun.log_path.alias("reoptimize_path"),
                FromScratchBayesRun.log_path.alias("from_scratch_path"),
            )
            .join(
                FromScratchBayesRun,
                on=(BayesRun.query_name == FromScratchBayesRun.query_name),
                attr="from_scratch",
            )
            .where(
                (BayesRun.workload_set == "SO_RETRO")
                & (FromScratchBayesRun.workload_set == "SO_FUTURE")
            )
        )
        reoptimize_times = []
        from_scratch_times = []
        for comparison in tqdm(comparisons):
            reoptimize_run = pd.read_csv(comparison.reoptimize_path)
            from_scratch_run = pd.read_csv(comparison.from_scratch.from_scratch_path)
            reoptimize_end = reoptimize_run.dropna(
                subset=["non_parallel_runtime", "best_found"]
            ).iloc[-1]
            from_scratch_end = from_scratch_run.dropna(
                subset=["non_parallel_runtime", "best_found"]
            ).iloc[-1]

            reoptimize_times.append(
                min(
                    -1 * reoptimize_end["best_found"],
                    bao_optimal_time(comparison.query_name, "SO_FUTURE"),
                )
            )
            from_scratch_times.append(
                min(
                    -1 * from_scratch_end["best_found"],
                    bao_optimal_time(comparison.query_name, "SO_FUTURE"),
                )
            )

        return bao_times, past_times, from_scratch_times, reoptimize_times

    bao_times, past_times, from_scratch_times, reopt_times = load_or_calc(
        "drift_figure_1", calc_reoptimize_series
    )

    aggs = [Aggregate.median, Aggregate.avg, Aggregate.p90]
    group_labels = ["Median", "Average", "P90"]
    series_labels = ["Bao", "Past Plan", "Bao-only BO", "Bao + Past Plan BO"]
    series_values = [bao_times, past_times, from_scratch_times, reopt_times]
    bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    method_bars = {
        method: [AGG_FUNCS[agg](values) for agg in aggs]
        for method, values in zip(series_labels, series_values)
    }

    xs = np.arange(len(group_labels))
    BAR_WIDTH = 0.2
    ax1_bars = []
    for i, (method, bars) in enumerate(method_bars.items()):
        rects = ax1.bar(
            xs + i * BAR_WIDTH,
            bars,
            label=method,
            width=BAR_WIDTH,
            zorder=3,
            edgecolor="black",
            color=bar_colors[i],
        )
        # ax1.bar_label(rects, [f"{x:.2f}" for x in bars], rotation=90, padding=3)
        ax1_bars.append(rects)
    ax1.legend()
    ax1.set_xticks(xs + 1.5 * BAR_WIDTH, group_labels)
    ax1.set_ylabel("Query runtime (s)")
    ax1.grid(axis="y", ls="--")
    ax1.set_title("Plan Drift & Reoptimization")

    # ====================
    # VAE Drift Bar Charts
    # ====================

    def calc_vae_series():
        RetrainedBayesRun = BayesRun.alias()
        comparisons = (
            BayesRun.select(
                BayesRun.query_name,
                BayesRun.log_path.alias("drifted_vae_path"),
                RetrainedBayesRun.log_path.alias("retrained_vae_path"),
            )
            .join(
                RetrainedBayesRun,
                on=(BayesRun.query_name == RetrainedBayesRun.query_name),
                attr="retrained",
            )
            .where(
                (BayesRun.workload_set == "SO_DRIFT")
                & (RetrainedBayesRun.workload_set == "SO_FUTURE")
            )
        )
        drifted_times = []
        retrained_times = []
        for comparison in tqdm(comparisons):
            drifted_run = pd.read_csv(comparison.drifted_vae_path)
            retrained_run = pd.read_csv(comparison.retrained.retrained_vae_path)
            drifted_end = drifted_run.dropna(
                subset=["non_parallel_runtime", "best_found"]
            ).iloc[-1]
            retrained_end = retrained_run.dropna(
                subset=["non_parallel_runtime", "best_found"]
            ).iloc[-1]

            drifted_times.append(
                min(
                    -1 * drifted_end["best_found"],
                    bao_optimal_time(comparison.query_name, "SO_FUTURE"),
                )
            )
            retrained_times.append(
                min(
                    -1 * retrained_end["best_found"],
                    bao_optimal_time(comparison.query_name, "SO_FUTURE"),
                )
            )
        return drifted_times, retrained_times

    drifted_vae_times, retrained_vae_times = load_or_calc(
        "drift_figure_2", calc_vae_series
    )

    aggs = [Aggregate.median, Aggregate.avg, Aggregate.p90]
    group_labels = ["Median", "Average", "P90"]
    series_labels = ["Past VAE", "Retrained VAE"]
    series_values = [drifted_vae_times, retrained_vae_times]
    bar_colors = ["tab:cyan", "tab:green"]

    method_bars = {
        method: [AGG_FUNCS[agg](values) for agg in aggs]
        for method, values in zip(series_labels, series_values)
    }

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})

    xs = np.arange(len(group_labels))
    BAR_WIDTH = 0.2
    ax2_bars = []
    for i, (method, bars) in enumerate(method_bars.items()):
        rects = ax2.bar(
            xs + i * BAR_WIDTH,
            bars,
            label=method,
            width=BAR_WIDTH,
            zorder=3,
            edgecolor="black",
            color=bar_colors[i],
        )
        # ax2.bar_label(rects, [f"{x:.2f}" for x in bars], rotation=90, padding=3)
        ax2_bars.append(rects)
    ax2.legend()
    ax2.set_xticks(xs + 0.5 * BAR_WIDTH, group_labels)
    ax2.grid(axis="y", ls="--")
    ax2.set_title("BO Using Old vs. Retrained VAE")

    # ==========================
    # Reoptimization Convergence
    # ==========================
    times = np.arange(5 * 60, 6 * 60 * 60, 5 * 60)

    def calc_convergence_series():
        FromScratchBayesRun = BayesRun.alias()
        comparisons = (
            BayesRun.select(
                BayesRun.query_name,
                BayesRun.log_path.alias("reoptimize_path"),
                FromScratchBayesRun.log_path.alias("from_scratch_path"),
            )
            .join(
                FromScratchBayesRun,
                on=(BayesRun.query_name == FromScratchBayesRun.query_name),
                attr="from_scratch",
            )
            .where(
                (BayesRun.workload_set == "SO_RETRO")
                & (FromScratchBayesRun.workload_set == "SO_FUTURE")
            )
        )
        reoptimize_runs: list[list[tuple[float, float]]] = []
        from_scratch_runs: list[list[tuple[float, float]]] = []
        for comparison in tqdm(comparisons):
            reoptimize_run = pd.read_csv(comparison.reoptimize_path)
            from_scratch_run = pd.read_csv(comparison.from_scratch.from_scratch_path)
            reoptimize_run["non_parallel_runtime"] *= 60 * 60
            from_scratch_run["non_parallel_runtime"] *= 60 * 60
            reoptimize_run["best_found"] *= -1
            from_scratch_run["best_found"] *= -1
            reoptimize_run = list(
                reoptimize_run.dropna(subset=["non_parallel_runtime", "best_found"])[
                    ["non_parallel_runtime", "best_found"]
                ].itertuples(index=False, name=None)
            )
            reoptimize_run = [
                (
                    opttime,
                    min(runtime, bao_optimal_time(comparison.query_name, "SO_FUTURE")),
                )
                for opttime, runtime in reoptimize_run
            ]
            from_scratch_run = list(
                from_scratch_run.dropna(subset=["non_parallel_runtime", "best_found"])[
                    ["non_parallel_runtime", "best_found"]
                ].itertuples(index=False, name=None)
            )
            from_scratch_run = [
                (
                    opttime,
                    min(runtime, bao_optimal_time(comparison.query_name, "SO_FUTURE")),
                )
                for opttime, runtime in from_scratch_run
            ]
            reoptimize_runs.append(reoptimize_run)
            from_scratch_runs.append(from_scratch_run)

        median_reoptimize_times = []
        median_from_scratch_times = []
        p90_reoptimize_times = []
        p90_from_scratch_times = []

        for time in tqdm(times):
            reoptimize_times = [
                max(point for point in run if point[0] <= time)[1]
                for run in reoptimize_runs
            ]
            from_scratch_times = [
                max(point for point in run if point[0] <= time)[1]
                for run in from_scratch_runs
            ]
            median_reoptimize_times.append(np.median(reoptimize_times))
            median_from_scratch_times.append(np.median(from_scratch_times))
            p90_reoptimize_times.append(np.percentile(reoptimize_times, 90))
            p90_from_scratch_times.append(np.percentile(from_scratch_times, 90))

        return (
            median_reoptimize_times,
            median_from_scratch_times,
            p90_reoptimize_times,
            p90_from_scratch_times,
        )

    (
        median_reoptimize_times,
        median_from_scratch_times,
        p90_reoptimize_times,
        p90_from_scratch_times,
    ) = load_or_calc("drift_figure_3", calc_convergence_series)
    ax3.plot(
        times,
        median_reoptimize_times,
        label="Bao + Past Plan Init (Median)",
        color="tab:red",
    )
    ax3.plot(
        times,
        median_from_scratch_times,
        label="Bao Init (Median)",
        color="tab:green",
    )
    ax3.plot(
        times,
        p90_reoptimize_times,
        label="Bao + Past Plan Init (P90)",
        color="tab:red",
        linestyle="--",
    )
    ax3.plot(
        times,
        p90_from_scratch_times,
        label="Bao Init (P90)",
        color="tab:green",
        linestyle="--",
    )
    ax3.grid(axis="y", ls="--")
    ax3.grid(axis="x", ls="--")
    ax3.set_xlabel("Optimization time")
    ax3.set_title("Reoptimization vs. Optimization Time")
    ax3.legend()
    ax3.set_xticks(list(range(0, 6 * 60 * 60 + 1, 3600)))
    ax3.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, _: compact_time(x))
    )

    # get ylim
    ylim = max(
        ax1.get_ylim()[1],
        ax2.get_ylim()[1],
    )
    ax1.set_ylim(0, ylim)
    ax2.set_ylim(0, ylim)
    ax3.set_ylim(0, ylim)
    ax3.set_xlim(0, 6 * 60 * 60)

    for bars in ax1_bars:
        autolabel(ax1, bars, ylim, fontsize=32)
    for bars in ax2_bars:
        autolabel(ax2, bars, ylim, fontsize=32)

    # plt.show()
    fig.savefig("db_eval/figures/drift_figure.pdf")


@app.command()
def stack_shift_summary():
    results = StackShiftedOptimizedPlan.select(
        StackShiftedOptimizedPlan.query_name,
        StackShiftedOptimizedPlan.plan_version,
        StackShiftedOptimizedPlan.shifted_to,
        fn.AVG(StackShiftedOptimizedPlan.runtime_secs),
    ).group_by(
        StackShiftedOptimizedPlan.query_name,
        StackShiftedOptimizedPlan.plan_version,
        StackShiftedOptimizedPlan.shifted_to,
    )
    dates = list(sorted(set(result.shifted_to for result in results)))
    past_medians = []
    past_p90s = []
    past_maxs = []
    future_medians = []
    future_p90s = []
    future_maxs = []
    for date in dates:
        date_results = [result for result in results if result.shifted_to == date]
        past_results = [
            result.runtime_secs
            for result in date_results
            if result.plan_version == "past"
        ]
        future_results = [
            result.runtime_secs
            for result in date_results
            if result.plan_version == "future"
        ]
        for result in date_results:
            print(
                f"{result.query_name},{result.plan_version},{result.runtime_secs:.2f}"
            )
        past_medians.append(np.median(past_results))
        past_p90s.append(np.percentile(past_results, 90))
        past_maxs.append(max(past_results))
        future_medians.append(np.median(future_results))
        future_p90s.append(np.percentile(future_results, 90))
        future_maxs.append(max(future_results))

    plt.subplot(131)
    plt.plot(dates, past_medians, label="Past", marker="o")
    plt.plot(dates, future_medians, label="Future", marker="o")
    plt.ylim(bottom=0)
    plt.legend()
    plt.title("Median runtime")

    plt.subplot(132)
    plt.plot(dates, past_p90s, label="Past", marker="o")
    plt.plot(dates, future_p90s, label="Future", marker="o")
    plt.ylim(bottom=0)
    plt.legend()
    plt.title("P90 runtime")

    plt.subplot(133)
    plt.plot(dates, past_maxs, label="Past", marker="o")
    plt.plot(dates, future_maxs, label="Future", marker="o")
    plt.ylim(bottom=0)
    plt.legend()
    plt.title("Max runtime")

    plt.show()
    # plt.savefig("db_eval/figures/stack_shift_figure.png")


@app.command()
def stack_shift_figure():
    results = StackShiftedOptimizedPlan.select(
        StackShiftedOptimizedPlan.query_name,
        StackShiftedOptimizedPlan.plan_version,
        StackShiftedOptimizedPlan.shifted_to,
        fn.AVG(StackShiftedOptimizedPlan.runtime_secs),
    ).group_by(
        StackShiftedOptimizedPlan.query_name,
        StackShiftedOptimizedPlan.plan_version,
        StackShiftedOptimizedPlan.shifted_to,
    )
    LOW_PERCENTILE = 25
    HIGH_PERCENTILE = 75
    TOP_N = 3

    dates = list(sorted(set(result.shifted_to for result in results)))
    past_low_percentile = []
    past_median = []
    past_high_percentile = []
    future_low_percentile = []
    future_median = []
    future_high_percentile = []
    past_top_n = [[] for _ in range(TOP_N)]
    future_top_n = [[] for _ in range(TOP_N)]
    for date in dates:
        date_results = [result for result in results if result.shifted_to == date]
        past_results = list(
            sorted(
                result.runtime_secs
                for result in date_results
                if result.plan_version == "past"
            )
        )
        future_results = list(
            sorted(
                result.runtime_secs
                for result in date_results
                if result.plan_version == "future"
            )
        )
        past_low_percentile.append(np.percentile(past_results, LOW_PERCENTILE))
        past_median.append(np.median(past_results))
        past_high_percentile.append(np.percentile(past_results, HIGH_PERCENTILE))
        future_low_percentile.append(np.percentile(future_results, LOW_PERCENTILE))
        future_median.append(np.median(future_results))
        future_high_percentile.append(np.percentile(future_results, HIGH_PERCENTILE))
        for i in range(TOP_N):
            past_top_n[i].append(past_results[-(i + 1)])
            future_top_n[i].append(future_results[-(i + 1)])

    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams.update({"font.size": 28})
    plt.figure(figsize=(14, 6))
    LINE_WIDTH = 3

    plt.subplot(121)
    # plt.plot(
    #     dates,
    #     past_low_percentile,
    #     marker=".",
    #     color="tab:orange",
    # )
    plt.plot(
        dates,
        past_median,
        label="Past plan",
        marker="o",
        linestyle="dashed",
        linewidth=LINE_WIDTH,
        color="tab:orange",
    )
    # plt.plot(
    #     dates,
    #     past_high_percentile,
    #     marker=".",
    #     color="tab:orange",
    # )
    plt.fill_between(
        dates,
        past_low_percentile,
        past_high_percentile,
        color="tab:orange",
        alpha=0.2,
    )

    # plt.plot(
    #     dates,
    #     future_low_percentile,
    #     marker=".",
    #     color="tab:blue",
    # )
    plt.plot(
        dates,
        future_median,
        label="Future plan",
        linewidth=LINE_WIDTH,
        marker="o",
        color="tab:blue",
    )
    # plt.plot(
    #     dates,
    #     future_high_percentile,
    #     marker=".",
    #     color="tab:blue",
    # )
    plt.fill_between(
        dates,
        future_low_percentile,
        future_high_percentile,
        color="tab:blue",
        alpha=0.2,
    )

    plt.ylim(bottom=0)
    plt.ylabel("Runtime (s)")
    plt.xlabel("Date")
    plt.xticks(fontsize=20)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
    plt.title("Plan Runtimes vs. Date")
    plt.legend(fontsize=20)

    plt.subplot(122)
    ALPHA_SCALE = 0.2
    colors = ["tab:purple", "tab:cyan", "tab:red"]
    for i in range(TOP_N):
        plt.plot(
            dates,
            past_top_n[i],
            marker="o",
            color=colors[i],
            linestyle="dashed",
            linewidth=LINE_WIDTH,
        )
        plt.plot(
            dates,
            future_top_n[i],
            marker="o",
            color=colors[i],
            linestyle="solid",
            linewidth=LINE_WIDTH,
        )
    # plt.yscale("log")
    # plt.ylim(bottom=1)
    plt.ylim(bottom=0)
    plt.xlabel("Date")
    plt.xticks(fontsize=20)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
    plt.title("Top 3 Runtimes vs. Date")

    # Custom legend:
    # Solid: Future plan
    # Dashed: Past plan
    # Purple: 1st
    # Cyan: 2nd
    # Red: 3rd
    legend_elements = [
        Line2D([0], [0], color="black", linestyle="solid", label="Future plan"),
        Line2D([0], [0], color="black", linestyle="dashed", label="Past plan"),
        Patch(facecolor="tab:purple", label="1st"),
        Patch(facecolor="tab:cyan", label="2nd"),
        Patch(facecolor="tab:red", label="3rd"),
    ]

    plt.legend(handles=legend_elements, fontsize=20)

    plt.savefig("db_eval/figures/stack_shift_figure.pdf")


@app.command()
def stack_shift_divergence():
    results = StackShiftedOptimizedPlan.select(
        StackShiftedOptimizedPlan.query_name,
        StackShiftedOptimizedPlan.plan_version,
        StackShiftedOptimizedPlan.shifted_to,
        fn.MIN(StackShiftedOptimizedPlan.runtime_secs),
    ).group_by(
        StackShiftedOptimizedPlan.query_name,
        StackShiftedOptimizedPlan.plan_version,
        StackShiftedOptimizedPlan.shifted_to,
    )
    dates = list(sorted(set(result.shifted_to for result in results)))
    for i, date in enumerate(tqdm(dates)):
        date_results = [result for result in results if result.shifted_to == date]
        past_results = [
            (result.query_name, result.runtime_secs)
            for result in date_results
            if result.plan_version == "past"
        ]
        future_runtime = {
            result.query_name: result.runtime_secs
            for result in date_results
            if result.plan_version == "future"
        }

        degradations = sorted(
            [
                (
                    past_runtime / future_runtime[query_name]
                    if query_name in future_runtime
                    else 1
                )
                for query_name, past_runtime in past_results
            ]
        )
        negative_bars = [(d - 1) * 100 for d in degradations if d < 1]
        positive_bars = [(d - 1) * 100 for d in degradations if d >= 1]

        named_degradations = sorted(
            [
                (
                    query_name,
                    (
                        past_runtime / future_runtime[query_name]
                        if query_name in future_runtime
                        else 1
                    ),
                )
                for query_name, past_runtime in past_results
            ],
            key=lambda x: x[1],
        )
        print(f"Worst degradations on {date}")
        for i in range(-3, 0):
            print(named_degradations[i])

        plt.rcParams["figure.constrained_layout.use"] = True
        plt.rcParams.update({"font.size": 28})
        plt.figure(figsize=(12, 6))

        neg_rects = plt.bar(
            list(range(len(negative_bars))), negative_bars, color="tab:blue"
        )
        pos_rects = plt.bar(
            list(range(len(negative_bars), len(degradations))),
            positive_bars,
            color="tab:orange",
        )
        ax = plt.gca()
        # ylim = ax.get_ylim()[1]
        ylim = 100
        plt.ylim(-100, ylim)
        plt.xlim(-0.5, len(degradations) - 0.5)
        autolabel(ax, neg_rects, ylim, fontsize=12, ymin=2)
        autolabel(ax, pos_rects, ylim, fontsize=12, ymin=2)

        plt.xlabel("Query")
        plt.ylabel("Percentage Difference (%)")
        plt.grid(axis="y", ls="--")
        plt.xticks([])
        plt.title(f"Degradation on {date}")
        plt.savefig(f"db_eval/figures/stack_shift_degradation_{i}.png")


@app.command()
def limeqo_comparison(
    query_names: list[str] = typer.Option(),
    workload_set: str = typer.Option(),
    max_time: int = 5 * 60,
):
    num_queries = len(query_names)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["figure.constrained_layout.h_pad"] = 0.1
    plt.rcParams.update({"font.size": 28})
    LINE_WIDTH = 3
    fig = plt.figure(figsize=(12, 4 * num_queries))

    for i, query_name in enumerate(query_names):
        hint_runtimes = {
            (row.join_hint, row.scan_hint): row.avg_runtime
            for row in (
                BaoPlan.select(
                    BaoPlan.join_hint,
                    BaoPlan.scan_hint,
                    fn.AVG(BaoPlan.runtime_secs).alias("avg_runtime"),
                )
                .where(
                    BaoPlan.query_name == query_name,
                    BaoPlan.workload_set == workload_set,
                )
                .group_by(BaoPlan.join_hint, BaoPlan.scan_hint)
            )
        }
        bayesqo_hints = ranked_hint_badness()
        limeqo_hints = limeqo_hint_order()

        limeqo_optimization_time = []
        limeqo_best_runtime = []
        limeqo_best_seen = float("inf")
        limeqo_cumulative_time = 0
        for join_hint, scan_hint in limeqo_hints:
            runtime = hint_runtimes[(join_hint, scan_hint)]
            if runtime < limeqo_best_seen:
                limeqo_best_seen = runtime
            limeqo_cumulative_time += limeqo_best_seen
            limeqo_optimization_time.append(limeqo_cumulative_time)
            limeqo_best_runtime.append(limeqo_best_seen)
        limeqo_optimization_time.append(max_time)
        limeqo_best_runtime.append(limeqo_best_seen)

        bayesqo_optimization_time = []
        bayesqo_best_runtime = []
        bayesqo_best_seen = float("inf")
        bayesqo_cumulative_time = 0
        for join_hint, scan_hint in bayesqo_hints:
            runtime = hint_runtimes[(join_hint, scan_hint)]
            if runtime < bayesqo_best_seen:
                bayesqo_best_seen = runtime
            bayesqo_cumulative_time += bayesqo_best_seen
            bayesqo_optimization_time.append(bayesqo_cumulative_time)
            bayesqo_best_runtime.append(bayesqo_best_seen)

        bo_run = best_bayes_series(
            query_name, workload_set, {InitType.bao}, cross_joins=False
        )
        for time, best_seen in bo_run:
            cumulative_time = time + bayesqo_cumulative_time
            if cumulative_time > max_time:
                break
            bayesqo_optimization_time.append(cumulative_time)
            bayesqo_best_runtime.append(best_seen)
        bayesqo_optimization_time.append(max_time)
        bayesqo_best_runtime.append(best_seen)

        plt.subplot(num_queries, 1, query_names.index(query_name) + 1)
        plt.plot(
            limeqo_optimization_time,
            limeqo_best_runtime,
            label="LimeQO",
            linewidth=LINE_WIDTH,
            color="tab:green",
        )
        plt.plot(
            bayesqo_optimization_time,
            bayesqo_best_runtime,
            label="BayesQO",
            linewidth=LINE_WIDTH,
            color="tab:blue",
        )
        plt.legend()
        if i == num_queries - 1:
            plt.xlabel("Optimization time")
        plt.xticks(fontsize=20)
        plt.gca().xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: compact_time(x))
        )
        plt.ylabel("Runtime (s)")
        plt.yticks(fontsize=20)
        plt.gca().set_ylim(bottom=0)
        plt.grid(axis="y", ls="--")
        plt.grid(axis="x", ls="--")
        plt.title(f"{query_name}")
    # plt.show()
    plt.savefig("db_eval/figures/limeqo_comparison.pdf")


@app.command()
def median_joins():
    for eval_workload in EvalWorkload:
        queries = resolve_eval_queries(eval_workload)
        workload_set_name = resolve_workload_set(eval_workload)
        workload_set = get_workload_set(workload_set_name)
        num_joins = []
        for query in queries:
            query_spec = workload_set.queries[query]
            num_joins.append(
                sum(num_aliases for _, num_aliases in query_spec.query_tables) - 1
            )
        print(f"{eval_workload.name}: {np.median(num_joins)}")


if __name__ == "__main__":
    app()
