import colorsys
import os
from math import ceil
from statistics import geometric_mean
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer  # type: ignore
from tqdm import tqdm
from workload.workloads import get_workload_set

from .bao import bao_optimal_time, postgres_time
from .bayes import InitType, bayes_series
from .random_plans import all_random_series
from .utils import AGG_FUNCS, Aggregate, compact_time, pretty_time

app = typer.Typer(no_args_is_help=True)


def only_improvements(series: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Given a series of optimizations, returns only the points where
    runtime improved.
    """
    filtered = []
    current_best = float("inf")
    for cumulative_time, plan_runtime in series:
        if plan_runtime < current_best:
            current_best = plan_runtime
            filtered.append((cumulative_time, plan_runtime))
    return filtered


def strictly_decreasing(series: list[tuple[float, float]]) -> list[tuple[float, float]]:
    corrected = []
    current_best = float("inf")
    for cumulative_time, plan_runtime in series:
        if plan_runtime < current_best:
            current_best = plan_runtime
        corrected.append((cumulative_time, current_best))
    return corrected


def best_overall(optimization_series: list[list[tuple[float, float]]]) -> float:
    """Given multiple series of optimizations, returns the best time
    achieved across all series.
    """
    return min(min(plan_runtime for _, plan_runtime in series) for series in optimization_series)


def best_at_time(optimization_series: list[list[tuple[float, float]]], time: float) -> Optional[float]:
    """Given a particular time, returns the best plan runtime achieved"""
    best_seen = None
    seen_any = False
    for series in optimization_series:
        for cumulative_time, plan_runtime in series:
            if cumulative_time > time:
                break
            if best_seen is None or plan_runtime < best_seen:
                seen_any = True
                best_seen = plan_runtime
    if not seen_any:
        return min(series[0][1] for series in optimization_series)
    return best_seen


# https://stackoverflow.com/a/60562502
def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def average_with_error(
    series: list[list[tuple[float, float]]], time_step_secs: int
) -> tuple[list[float], list[float], list[float]]:
    series = [s for s in series if s[0][0] < time_step_secs]
    last_time = max(run[-1][0] for run in series)
    times = list(float(t) for t in range(time_step_secs, int(ceil(last_time)), time_step_secs))
    means = []
    errors = []
    for time in times:
        best_times = []
        for run in series:
            before_time = [point for point in run if point[0] <= time]
            if len(before_time) == 0:
                continue
            last_point = max(
                before_time,
                key=lambda x: x[0],
            )
            best_times.append(last_point[1])
        means.append(np.mean(best_times))
        errors.append(np.std(best_times))
    return times, means, errors


@app.command()
def time_vs_best(
    workload_set: str = typer.Option(),
    query: str = typer.Option(str),
    cross_joins: bool = False,
    inits: List[InitType] = typer.Option([InitType.bao]),
    save: bool = False,
):
    postgres = postgres_time(query)
    plt.axhline(y=postgres, color="orange", label="PostgreSQL", linestyle="--")

    bao_optimal = bao_optimal_time(query, workload_set)
    plt.axhline(y=bao_optimal, color="green", label="Bao", linestyle="--")
    plt.grid(axis="both", which="both", zorder=0)

    # color = mpl.colors.ColorConverter.to_rgb("firebrick")
    # for run_num, run in enumerate(all_random_series(query)):
    #     run_points = strictly_decreasing(run)
    #     x, y = zip(*run_points)
    #     plt.plot(
    #         x,
    #         y,
    #         label="Random" if run_num == 0 else "",
    #         color=scale_lightness(color, (run_num * 0.2) + 1),
    #     )

    xs, ys, errors = average_with_error(
        [strictly_decreasing(run) for run in all_random_series(query)],
        time_step_secs=1 * 60,
    )
    plt.plot(xs, ys, label="Random", color="firebrick")
    plt.fill_between(
        xs,
        np.array(ys) - np.array(errors),
        np.array(ys) + np.array(errors),
        alpha=0.2,
        color="firebrick",
    )

    # color = mpl.colors.ColorConverter.to_rgb("blue")
    # set_label = False
    # for run_num, run in enumerate(bayes_series(query, inits, cross_joins)):
    #     x, y = zip(*run)
    #     if len(run) < 20:
    #         continue
    #     plt.plot(
    #         x,
    #         y,
    #         label="Our Method" if not set_label else "",
    #         color=scale_lightness(color, (run_num * 0.05) + 1),
    #     )
    #     set_label = True

    xs, ys, errors = average_with_error(
        [strictly_decreasing(run) for run in bayes_series(query, inits, cross_joins)],
        time_step_secs=1 * 60,
    )
    plt.plot(xs, ys, label="Our Method", color="blue")
    plt.fill_between(
        xs,
        np.array(ys) - np.array(errors),
        np.array(ys) + np.array(errors),
        alpha=0.2,
        color="blue",
    )

    balsa_log_path = os.path.join(os.path.dirname(__file__), "balsa", f"{query}.csv")
    if os.path.exists(balsa_log_path):
        balsa_data = pd.read_csv(balsa_log_path)
        x = balsa_data["Time (m)"] * 60
        y = balsa_data["Best latency (s)"]
        plt.plot(x, y, label="Balsa", color="darkviolet")

    plt.gca().set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
    plt.gca().yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
    plt.gca().set_xticks([0, 3600, 2 * 3600, 3 * 3600, 4 * 3600, 5 * 3600, 6 * 3600])
    plt.gca().set_xlim(0, 5 * 3600)
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: compact_time(x)))
    plt.title(
        f"{query} comparison of optimization strategies",
        fontfamily="Arvo",
    )
    plt.xlabel("Cumulative optimization time (hours)")
    plt.ylabel("Plan runtime (s)")
    plt.legend()
    plt.tight_layout()
    if save:
        target_dir = os.path.join(os.path.dirname(__file__), "plots/time_vs_best")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                target_dir,
                f"{query}_{'cross_joins' if cross_joins else 'no_cross_joins'}_{'_'.join(sorted(inits))}.svg",
            )
        )
        plt.close()
    else:
        plt.show()


@app.command()
def plot_all_time_vs_best(
    workload_set: str = typer.Option(),
    cross_joins: bool = False,
    inits: List[InitType] = typer.Option([InitType.bao]),
):
    workload_set_obj = get_workload_set(workload_set)
    for query in tqdm(workload_set_obj.queries):
        try:
            time_vs_best(
                workload_set,
                query,
                cross_joins=cross_joins,
                inits=inits,
                save=True,
            )
            tqdm.write(query)
        except ValueError:
            tqdm.write(f"Skipping {query}")


@app.command()
def plot_whole_benchmark(
    aggregate: Aggregate = typer.Option(),
    workload_set: str = typer.Option(),
    cross_joins: bool = False,
    inits: List[InitType] = typer.Option([InitType.bao]),
    save: bool = False,
):
    postgres = []
    bao = []
    # random = []
    bayes = []
    workload_set_obj = get_workload_set(workload_set)
    for query in tqdm(workload_set_obj.queries):
        try:
            query_postgres = postgres_time(query)
            query_bao = bao_optimal_time(query, workload_set)
            query_bayes = (
                best_overall(bayes_series(query, set(inits), cross_joins))
                if query != "JOB_20B"
                else best_overall(bayes_series(query, {InitType.random}, cross_joins))
            )
            query_bayes_bao = min(query_bayes, query_bao)
            # query_random = best_overall(all_random_series(query))

            tqdm.write(query)
            postgres.append(query_postgres)
            bao.append(query_bao)
            bayes.append(query_bayes_bao)
            # random.append(query_random)
        except ValueError as e:
            tqdm.write(f"Skipping {query}: {e}")

    # Bar chart of best time with each technique for each query
    agg_func = AGG_FUNCS[aggregate]
    postgres_agg = agg_func(postgres)
    bao_agg = agg_func(bao)
    # random_agg = agg_func(random)
    bayes_agg = agg_func(bayes)

    if aggregate == Aggregate.sum:
        balsa_agg = 52
    elif aggregate == Aggregate.p90:
        balsa_agg = 2.7
    else:
        balsa_agg = 0

    # strategies = ["PostgreSQL", "Bao", "Random", "Our Method"]
    # times = [postgres_agg, bao_agg, random_agg, bayes_agg]
    strategies = ["PostgreSQL", "Bao", "Balsa", "BOUTT"]
    times = [postgres_agg, bao_agg, balsa_agg, bayes_agg]

    plt.rcParams["figure.figsize"] = (3.5, 2.8)
    plt.grid(axis="y", zorder=0)
    # plt.bar(strategies, times, color=["orange", "green", "firebrick", "navy"], zorder=3)
    plt.bar(
        strategies,
        times,
        color=["black", "red", "green", "magenta"],
        zorder=3,
    )
    if aggregate == Aggregate.sum:
        plt.ylabel("Whole JOB runtime (s)")
    elif aggregate == Aggregate.median:
        plt.ylabel("Median plan runtime (s)")
    elif aggregate == Aggregate.p90:
        plt.ylabel("90th percentile plan runtime (s)")
    plt.tight_layout()
    if save:
        target_dir = os.path.join(os.path.dirname(__file__), "plots/whole_benchmark")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                target_dir,
                f"{aggregate.name}_{'cross_joins' if cross_joins else 'no_cross_joins'}_{'_'.join(sorted(inits))}_{workload_set}.pdf",
            )
        )
        plt.close()
    else:
        plt.show()


@app.command()
def plot_best_overall(
    workload_set: str = typer.Option(),
    cross_joins: bool = False,
    inits: List[InitType] = typer.Option([InitType.bao]),
):
    # Bar chart of best time with each technique for each query
    # strategies = ["Postgres", "Bao", "Random", InitType.bao]
    strategies = ["Postgres", "Bao", InitType.bao]
    workload_set_obj = get_workload_set(workload_set)
    queries = workload_set_obj.queries
    times = {s: [] for s in strategies}
    idx = []
    for query in tqdm(queries):
        try:
            bayes = bayes_series(query, inits, cross_joins)
            postgres = postgres_time(query)
            bao = bao_optimal_time(query, workload_set)
            if len(bayes) == 0:
                raise ValueError

            times["Postgres"].append(postgres)
            times["Bao"].append(bao)
            times["Bayes"].append(best_overall(bayes))
            idx.append(query)
            # times["Random"].append(best_overall(all_random_series(query)))
        except ValueError:
            tqdm.write(f"Skipping {query}")

    df = pd.DataFrame(times, index=idx)
    df.plot.bar()
    plt.title("Best plan runtime for each query")
    plt.ylabel("Plan runtime (s)")
    plt.xlabel("Query")
    plt.show()


@app.command()
def plot_geomean_per_time(
    workload_set: str = typer.Option(),
    cross_joins: bool = False,
    inits: List[InitType] = typer.Option([InitType.bao]),
    save: bool = False,
):
    max_time = 16 * 60 * 60
    time_incr_mins = 5
    tick_incr_mins = 60
    times = list(range(time_incr_mins * 60, max_time + 1, time_incr_mins * 60))
    tick_times = list(range(tick_incr_mins * 60, max_time + 1, tick_incr_mins * 60))
    # labels = [pretty_time(tick) for tick in tick_times]
    labels = [f"{tick // 3600}" for tick in tick_times]

    bayes_bao_improvement = [[] for _ in range(len(times))]
    workload_set_obj = get_workload_set(workload_set)
    observed_queries = []
    for query in tqdm(workload_set_obj.queries):
        try:
            bayes = bayes_series(query, inits, cross_joins)
            for i, time in enumerate(times):
                # TODO: THIS IS CHEATING
                # We're taking the best time discovered by Bayes across all runs
                # so the time spent optimizing a particular query is actually multiplied
                # by the number of runs
                bayes_best = best_at_time(bayes, time)
                postgres = postgres_time(query)
                if bayes_best is None:
                    bayes_best = postgres
                bayes_bao_improvement[i].append(min(bayes_best, bao_optimal_time(query, workload_set)) / postgres)
            observed_queries.append(query)
        except ValueError:
            tqdm.write(f"Skipping {query}")
    tqdm.write(f"Saw {len(observed_queries)} / {len(workload_set_obj.queries)} queries")

    bao_geo = geometric_mean(
        [(bao_optimal_time(query, workload_set) / postgres_time(query)) for query in observed_queries]
    )
    plt.axhline(y=bao_geo, label="Bao", color="green", linestyle="--", zorder=4, linewidth=1)

    bayes_bao_means = [geometric_mean(points) for points in bayes_bao_improvement]
    plt.plot(
        times,
        bayes_bao_means,
        label="Our Method",
        color="blue",
        zorder=3,
    )

    if workload_set == "JOB":
        # balsa_geo = [1.043315, 0.726004, 0.583407, 0.580955, 0.567157, 0.559430, 0.554451]
        balsa_data = pd.read_json("balsa_geomeans.json")
        balsa_geo = balsa_data["Geomean"].tolist()[1:]
        plt.plot(times, balsa_geo, label="Balsa", color="darkviolet", zorder=3)

    plt.xticks(tick_times, labels)

    plt.grid(axis="y", zorder=0)
    plt.grid(axis="x", zorder=0)
    plt.legend()

    plt.title("Geometric mean improvement over PostgreSQL", fontfamily="Arvo", fontsize=16)
    plt.ylabel("Geometric mean proportional runtime")
    plt.xlabel("Wall clock time (hours)")
    plt.tight_layout()
    plt.gca().set_ylim(0, 1)
    plt.gca().set_xlim(0, 10 * 60 * 60)
    if save:
        target_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(target_dir, exist_ok=True)

        plt.savefig(
            os.path.join(
                target_dir,
                f"geomean_per_time_{'cross_joins' if cross_joins else 'no_cross_joins'}_{'_'.join(sorted(inits))}_{workload_set}.svg",
            ),
            # dpi=300,
        )
        plt.close()
    else:
        plt.show()


@app.command()
def plot_improvement_histogram(
    workload_set: str,
    time: int,
    cross_joins: bool = False,
    inits: List[InitType] = typer.Option([InitType.bao]),
    save: bool = False,
):
    workload_set_obj = get_workload_set(workload_set)
    bayes_improvement = []
    for query in tqdm(workload_set_obj.queries, desc="Queries", position=1, leave=False):
        try:
            bayes = bayes_series(query, inits, cross_joins)
            bayes_best = best_at_time(bayes, time)
            postgres = postgres_time(query)
            if bayes_best is None:
                bayes_best = postgres
            bayes_improvement.append((query, bayes_best / postgres))
        except ValueError:
            tqdm.write(f"Skipping {query}")
    bayes_improvement = sorted(bayes_improvement, key=lambda x: -1 * x[1])
    xs, ys = zip(*bayes_improvement)

    plt.bar(xs, ys, width=1, edgecolor="black", linewidth=0.5)

    text_size = 6
    tick_label_size = 3

    # label y values that go past the top of the chart
    for x, y in zip(xs, ys):
        if y > 2:
            plt.text(
                x,
                2.01,
                round(y, 2),
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=tick_label_size,
            )

    # add bao optimal marker to each bar
    bao_improvement = []
    for x, query in enumerate(xs):
        bao = bao_optimal_time(query, workload_set)
        bao_improvement.append(bao / postgres_time(query))
    plt.scatter(
        xs,
        bao_improvement,
        color="green",
        label="Bao",
        edgecolor="white",
        s=2,
        linewidth=0.5,
    )

    # horizontal line at 1 for parity with Postgres
    plt.axhline(y=1, color="orange", label="Postgres", linewidth=1, alpha=0.5)
    plt.gca().set_aspect(((9 / 16) * len(xs)) / 2, adjustable="box")
    # make y axis stable across all plots
    plt.ylim(0, 2)
    for x in plt.gca().spines.values():
        x.set_linewidth(0.5)
    plt.gca().tick_params(width=0.5)
    plt.xticks(rotation=90, fontsize=tick_label_size)
    plt.xlim(-1, len(xs))
    plt.xlabel("Query", fontsize=text_size)
    plt.yticks(fontsize=text_size)
    plt.ylabel("Fraction of Postgres time (smaller is better)", fontsize=text_size)
    plt.title(
        f"Runtime vs. Postgres at {pretty_time(time)}",
        fontsize=text_size,
    )
    if save:
        target_dir = os.path.join(
            os.path.dirname(__file__),
            f"plots/improvement_histogram/{'cross_joins' if cross_joins else 'no_cross_joins'}/{'_'.join(sorted(inits))}",
        )
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(os.path.join(target_dir, f"{time}.png"), dpi=400, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


@app.command()
def plot_all_improvement_histograms(workload_set: str = typer.Option(), cross_joins: bool = False):
    for time in tqdm(
        [5 * i * 60 for i in range(1, (16 * 60) // 5)],
        desc="Histograms",
        position=0,
    ):
        plot_improvement_histogram(workload_set, time, cross_joins=cross_joins, save=True)


@app.command()
def compare_workloads():
    from statistics import median

    workload_sets = ["JOB", "CEB_3K"]
    for workload_set in workload_sets:
        workload_set_obj = get_workload_set(workload_set)
        runs = 0
        total_oracle_calls = 0
        oracle_call_points = []
        for query in tqdm(workload_set_obj.queries):
            try:
                bayes = bayes_series(query, False, True)
                total_oracle_calls += sum(len(run) for run in bayes)
                runs += len(bayes)
                oracle_call_points += [len(run) for run in bayes]
            except ValueError:
                # tqdm.write(f"Skipping {query}")
                pass
        tqdm.write(f"Average oracle calls for {workload_set}: {total_oracle_calls / runs}")
        tqdm.write(f"Median oracle calls for {workload_set}: {median(oracle_call_points)}")


if __name__ == "__main__":
    app()
