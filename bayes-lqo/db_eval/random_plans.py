import os
import pdb
import re
from itertools import combinations
from math import floor
from random import choice, shuffle
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import typer
from peewee import JOIN, AsIs, fn
from tqdm import tqdm

from codec.codec import (
    AliasesCodec,
    AliasSymbolTable,
    JoinOperator,
    JoinTree,
    JoinTreeBranch,
    expand_counts,
)
from logger.log import l
from oracle.oracle import WorkloadInput, oracle_for_workload_cluster
from oracle.structures import CompletedQuery, FailedQuery, TimedOutQuery
from workload.schema import build_alias_join_graph, build_join_graph
from workload.workloads import (
    OracleCodec,
    WorkloadSpec,
    WorkloadSpecDefinition,
    get_workload_set,
)

from .bao import bao_optimal_time, postgres_time, top_bot_improvement, top_pg_runtime
from .eval_workloads import EvalWorkload, resolve_eval_queries, resolve_workload_set
from .stack import workload_queries as stack_workload_queries
from .storage import (
    ExecutionResult,
    PlanType,
    PostgresPlan,
    RandomPlan,
    SaturatedRandomRun,
)
from .utils import AGG_FUNCS, JOB_QUERIES_SORTED, Aggregate, pretty_time

app = typer.Typer(no_args_is_help=True)


def generate_random_plan(
    workload: WorkloadSpecDefinition, plan_type: PlanType
) -> JoinTree:
    query_join_graph = workload.schema.query_join_graph
    # Sample edges until the graph is re-connected
    partial_joins = AliasSymbolTable(
        workload.all_tables, expand_counts(workload.all_tables, workload.query_tables)
    )
    if plan_type == PlanType.NonCrossUniformOp:
        random_edges = list(query_join_graph.edges)
        shuffle(random_edges)
    elif plan_type == PlanType.UniformRandom:
        random_edges = list(combinations(query_join_graph.nodes, 2))
        shuffle(random_edges)

    while len(partial_joins.tree_to_symbols) > 1:
        if len(random_edges) == 0:
            pdb.set_trace()
        (left_table, left_alias), (right_table, right_alias) = random_edges.pop()
        left_table_symol = workload.all_tables.index(left_table)
        right_table_symbol = workload.all_tables.index(right_table)
        left_subtree = partial_joins.get(left_table_symol, left_alias)
        right_subtree = partial_joins.get(right_table_symbol, right_alias)

        if left_subtree is None or right_subtree is None:
            raise ValueError(
                f"Couldn't find {left_table} {left_alias} or {right_table} {right_alias}"
            )

        # Sometimes edges will cause cycles
        if left_subtree is right_subtree:
            continue

        new_tree = JoinTreeBranch(
            left_subtree, right_subtree, choice(list(JoinOperator))
        )
        partial_joins = partial_joins.with_join(left_subtree, right_subtree, new_tree)

    join_tree = next(iter(partial_joins.tree_to_symbols.keys()))
    # return f"/*+\n{join_tree.to_operator_hint()}\n*/\n{workload.query_template.format(join_tree.to_join_clause())}"
    return join_tree


def build_query_join_graph(
    aliased_join_graph: nx.Graph, workload: WorkloadSpecDefinition
) -> nx.Graph:
    query_table_aliases = set(
        (table, alias_num)
        for (table, num_aliases) in workload.query_tables
        for alias_num in range(1, num_aliases + 1)
    )

    filtered_graph = nx.subgraph_view(
        aliased_join_graph,
        filter_node=lambda n: n in query_table_aliases,
        filter_edge=lambda u, v: u in query_table_aliases and v in query_table_aliases,
    )

    if not nx.is_connected(filtered_graph):
        # nx.draw_networkx(filtered_graph)
        # plt.show()
        # plt.clf()
        # nx.draw_networkx(workload.schema.query_join_graph)
        # plt.show()
        # pdb.set_trace()
        raise ValueError("Somehow the query graph is not connected")

    return filtered_graph


def try_random_plans(
    workload_set: str,
    query: str,
    run_id: int,
    plan_type: PlanType,
    max_plans: Optional[int] = None,
    max_time: Optional[int] = None,
    max_plans_without_improvement: Optional[int] = None,
    initial_timeout: Union[
        Literal["pg"], Literal["bao"], Literal["constant"]
    ] = "constant",
):
    workload_def_set = get_workload_set(workload_set)
    workload = workload_def_set.queries[query]

    # if workload_set in ["JOB", "CEB_3K"]:
    #     join_graph = build_join_graph(f"./schema_fk.sql")
    # else:
    #     join_graph = build_join_graph(f"./workload/stack/schema.sql")
    # aliased_join_graph = build_alias_join_graph(join_graph, 4)
    # query_join_graph = build_query_join_graph(aliased_join_graph, workload)

    workload_spec = WorkloadSpec.from_definition(workload, OracleCodec.Aliases)
    codec = AliasesCodec(workload.all_tables)

    if (
        SaturatedRandomRun.select()
        .where(
            (SaturatedRandomRun.plan_type == plan_type)
            & (SaturatedRandomRun.workload_set == workload_set)
            & (SaturatedRandomRun.query_name == query)
            & (SaturatedRandomRun.run_id == run_id)
        )
        .count()
        > 0
    ):
        l.info(f"Run {run_id} for {query} fully saturated, skipping")
        return

    plans_without_improvement = 0
    if max_plans_without_improvement:
        all_plans = (
            RandomPlan.select()
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.workload_set == workload_set)
                & (RandomPlan.query_name == query)
                & (RandomPlan.run_id == run_id)
            )
            .order_by(RandomPlan.plan_id)
        )
        best_seen = float("inf")
        last_improvement = 0
        for i, plan in enumerate(all_plans):
            if plan.runtime_secs < best_seen:
                best_seen = plan.runtime_secs
                last_improvement = i

        plans_without_improvement = all_plans.count() - last_improvement
        l.info(f"{plans_without_improvement} plans without improvement")

    past_initial_timeout = False
    best_seen = 5
    run_best_seen = (
        RandomPlan.select(fn.Min(RandomPlan.runtime_secs))
        .where(
            (RandomPlan.plan_type == plan_type)
            & (RandomPlan.workload_set == workload_set)
            & (RandomPlan.query_name == query)
            & (RandomPlan.run_id == run_id)
            & (RandomPlan.result == ExecutionResult.Success)
        )
        .first()
    )
    if run_best_seen.runtime_secs:
        past_initial_timeout = (
            RandomPlan.select()
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.workload_set == workload_set)
                & (RandomPlan.query_name == query)
                & (RandomPlan.run_id == run_id)
                & (RandomPlan.result == ExecutionResult.Success)
            )
            .count()
            > 0
        )
        if past_initial_timeout:
            best_seen = run_best_seen.runtime_secs
            l.info(f"Best seen from previous run: {pretty_time(best_seen)}")
        else:
            best_seen = run_best_seen.runtime_secs * 2
            l.info(
                f"No successful plans in previous run, doubling initial best to {pretty_time(best_seen)}"
            )
    else:
        match initial_timeout:
            case "pg":
                postgres_avg = postgres_time(query, workload_set)
                if postgres_avg:
                    l.info(
                        f"Using average postgres time of {postgres_avg} as initial best"
                    )

                    run_previous_timeout = (
                        RandomPlan.select(fn.Min(RandomPlan.runtime_secs))
                        .where(
                            (RandomPlan.plan_type == plan_type)
                            & (RandomPlan.workload_set == workload_set)
                            & (RandomPlan.query_name == query)
                            & (RandomPlan.run_id == run_id)
                        )
                        .first()
                    )

                    if (
                        run_previous_timeout.runtime_secs
                        and run_previous_timeout.runtime_secs < postgres_avg
                    ):
                        best_seen = run_previous_timeout.runtime_secs
                        l.warning(
                            f"Somehow previous run's best of {best_seen} is better than postgres time of {postgres_avg}"
                        )
                        l.info(
                            f"Using previous run's best of {best_seen} as initial best"
                        )
                    else:
                        best_seen = postgres_avg
                    past_initial_timeout = True
                else:
                    raise ValueError("No postgres time found")
            case "bao":
                bao_avg = bao_optimal_time(query, workload_set)
                if bao_avg:
                    l.info(f"Using average bao time of {bao_avg} as initial best")
                    best_seen = bao_avg
                    past_initial_timeout = True
                else:
                    raise ValueError("No bao time found")
            case "constant":
                l.info("Using constant initial best of 5s")
            case _:
                raise ValueError(f"Invalid timeout initialization {initial_timeout}")

    run_plan_count = (
        RandomPlan.select(fn.Min(RandomPlan.runtime_secs))
        .where(
            (RandomPlan.plan_type == plan_type)
            & (RandomPlan.workload_set == workload_set)
            & (RandomPlan.query_name == query)
            & (RandomPlan.run_id == run_id)
        )
        .count()
    ) or 0
    run_time_count = (
        RandomPlan.select(fn.Sum(RandomPlan.runtime_secs))
        .where(
            (RandomPlan.plan_type == plan_type)
            & (RandomPlan.query_name == query)
            & (RandomPlan.run_id == run_id)
        )
        .scalar()
    ) or 0

    if max_plans is not None:
        l.info(
            f"Starting run {run_id} for {query}, {run_plan_count} plans already evaluated"
        )
    elif max_time is not None:
        l.info(
            f"Starting run {run_id} for {query}, {run_plan_count} plans already evaluated for {pretty_time(run_time_count)}"
        )

    continuous_skips = 0
    while True:
        if max_plans and run_plan_count >= max_plans:
            l.info(f"Reached max plans of {max_plans} for run")
            break

        if (
            max_time
            and (
                RandomPlan.select(fn.Sum(RandomPlan.runtime_secs))
                .where(
                    (RandomPlan.plan_type == plan_type)
                    & (RandomPlan.query_name == query)
                    & (RandomPlan.run_id == run_id)
                )
                .scalar()
                or 0
            )
            > max_time
        ):
            l.info(f"Reached max time of {max_time} for run")
            break

        if (
            max_plans_without_improvement is not None
            and plans_without_improvement >= max_plans_without_improvement
        ):
            l.info(
                f"Run {run_id} for {query} has not improved in {max_plans_without_improvement} plans, stopping"
            )
            return

        plan = generate_random_plan(workload, plan_type)
        encoded = codec.encode(plan)

        existing = (
            RandomPlan.select(fn.Min(RandomPlan.runtime_secs))
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.query_name == query)
                & (RandomPlan.run_id == run_id)
                & (RandomPlan.encoded_plan == AsIs(encoded))
            )
            .count()
        )
        if existing > 0:
            l.info(f"Skipping existing plan")
            continuous_skips += 1
            if continuous_skips > 1000:
                l.info(f"Too many continuous skips, stopping")
                SaturatedRandomRun.create(
                    plan_type=plan_type,
                    workload_set=workload_set,
                    query_name=query,
                    run_id=run_id,
                )
                break
            continue
        continuous_skips = 0

        result = oracle_for_workload_cluster(
            workload_spec,
            [WorkloadInput("random_test", encoded, best_seen)],
        )[0]

        run_plan_count += 1
        num_plans = (
            RandomPlan.select()
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.query_name == query)
                & (RandomPlan.run_id == run_id)
            )
            .count()
        )
        if run_plan_count != num_plans + 1:
            # Happens when multiple parallel runs end up on the same query
            return

        match result:
            case CompletedQuery(_, elapsed_secs):
                if elapsed_secs < best_seen:
                    best_seen = elapsed_secs
                    past_initial_timeout = True
                    plans_without_improvement = 0
                l.info(
                    f"{run_plan_count}: Finished query in {pretty_time(elapsed_secs)}, current best {pretty_time(best_seen)}"
                )
                RandomPlan.create(
                    plan_type=plan_type,
                    workload_set=workload_set,
                    query_name=query,
                    run_id=run_id,
                    encoded_plan=encoded,
                    runtime_secs=elapsed_secs,
                    result=ExecutionResult.Success,
                )
            case TimedOutQuery(_, elapsed_secs):
                if not past_initial_timeout:
                    best_seen *= 2
                    best_seen = min(best_seen, 300)
                plans_without_improvement += 1

                l.info(f"{run_plan_count}: Timed out after {pretty_time(elapsed_secs)}")
                RandomPlan.create(
                    plan_type=plan_type,
                    workload_set=workload_set,
                    query_name=query,
                    run_id=run_id,
                    encoded_plan=encoded,
                    runtime_secs=elapsed_secs,
                    result=ExecutionResult.TimedOut,
                )
            case FailedQuery(_, elapsed_secs, error):
                l.info(
                    f"{run_plan_count}: Failed query after {pretty_time(elapsed_secs)}: {error}"
                )
                if isinstance(error, str) and (
                    "no space" in error or "invalid DSA memory alloc" in error
                ):
                    RandomPlan.create(
                        plan_type=plan_type,
                        workload_set=workload_set,
                        query_name=query,
                        run_id=run_id,
                        encoded_plan=encoded,
                        runtime_secs=elapsed_secs,
                        result=ExecutionResult.Error,
                    )


def all_random_series(
    query: str, max_samples: Optional[int] = None
) -> list[list[tuple[float, float]]]:
    """Produces a list of all salient optimization points in a given
    optimization run, i.e. each point where runtime improved.

    Each tuple is (cumulative optimization time, plan runtime)
    """
    highest_run_number = (
        RandomPlan.select(fn.MAX(RandomPlan.run_id))
        .where(RandomPlan.query_name == query)
        .scalar()
    )
    runs = []
    for run_id in range(1, highest_run_number + 1):
        run_points = []
        run = (
            RandomPlan.select()
            .where((RandomPlan.query_name == query) & (RandomPlan.run_id == run_id))
            .order_by(RandomPlan.plan_id)
        )
        cumulative_time = 0
        for plan in run:
            cumulative_time += plan.runtime_secs
            run_points.append((cumulative_time, plan.runtime_secs))

        if max_samples:
            run_points = run_points[:max_samples]
        runs.append(run_points)

    return runs


def best_random_series(
    query: str, max_samples: Optional[int] = None
) -> list[tuple[float, float]]:
    """
    Produces a list of all salient optimization points for the run ending
    with the best plan for the query.
    """
    all_series = all_random_series(query, max_samples)
    best_series = min(all_series, key=lambda series: series[-1][1])
    return best_series


@app.command()
def summarize(plan_type: PlanType = PlanType.NonCrossUniformOp):
    """List benchmarking stats for all runs across all queries"""
    print("Query    Runs  Plans  Total Time   Best Time  Postgres Time")
    print("=====    ====  =====  ==========   =========  =============")
    for query in JOB_QUERIES_SORTED:
        run_summary = (
            RandomPlan.select(
                fn.Sum(RandomPlan.runtime_secs).alias("total_time"),
                fn.Count(RandomPlan.plan_id).alias("total_plans"),
            )
            .where(
                (RandomPlan.plan_type == plan_type) & (RandomPlan.query_name == query)
            )
            .first()
        )
        best_time = (
            RandomPlan.select(fn.Min(RandomPlan.runtime_secs))
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.query_name == query)
                & (RandomPlan.result == ExecutionResult.Success)
            )
            .scalar()
        )
        runs = (
            RandomPlan.select(RandomPlan.run_id)
            .where(
                (RandomPlan.plan_type == plan_type) & (RandomPlan.query_name == query)
            )
            .distinct()
            .count()
        )

        postgres_time = (
            PostgresPlan.select(fn.AVG(PostgresPlan.runtime_secs))
            .where(PostgresPlan.query_name == query)
            .scalar()
        )

        print(
            f"{query:9}"
            f"{runs if runs else 'N/A':<6}"
            f"{run_summary.total_plans if run_summary.total_plans else 'N/A':<7}"
            f"{pretty_time(run_summary.total_time) if run_summary.total_time else 'N/A':<13}"
            f"{pretty_time(best_time) if best_time else 'N/A' :<11}"
            f"{pretty_time(postgres_time) if postgres_time else 'N/A'}"
        )


@app.command()
def list_runs(query: str, plan_type: PlanType = PlanType.NonCrossUniformOp):
    """List all runs for the given query"""
    runs = (
        RandomPlan.select(
            RandomPlan.run_id,
            fn.Sum(RandomPlan.runtime_secs).alias("total_time"),
            fn.Count(RandomPlan.plan_id).alias("total_plans"),
            fn.Min(RandomPlan.runtime_secs).alias("best_time"),
        )
        .where((RandomPlan.plan_type == plan_type) & (RandomPlan.query_name == query))
        .group_by(RandomPlan.run_id)
        .order_by(RandomPlan.run_id)
    )

    for run in runs:
        print(
            f"Run {run.run_id}: {run.total_plans} plans, "
            f"executed for {pretty_time(run.total_time)}, "
            f"best time {pretty_time(run.best_time)}"
        )

    summary = (
        RandomPlan.select(
            fn.Sum(RandomPlan.runtime_secs).alias("total_time"),
            fn.Count(RandomPlan.plan_id).alias("total_plans"),
            fn.Min(RandomPlan.runtime_secs).alias("best_time"),
        )
        .where((RandomPlan.plan_type == plan_type) & (RandomPlan.query_name == query))
        .first()
    )

    print(
        f"Overall: {summary.total_plans or 0} plans, "
        f"executed for {pretty_time(summary.total_time) if summary.total_time else 'N/A'}, "
        f"best time {pretty_time(summary.best_time) if summary.best_time else 'N/A'}"
    )


@app.command()
def begin(
    workload_set: str = typer.Option(),
    query: str = typer.Option(),
    max_plans: Optional[int] = None,
    plan_type: PlanType = PlanType.NonCrossUniformOp,
):
    """Start a new run for the given query"""
    # Get the next run_id
    highest_run = (
        RandomPlan.select(fn.Max(RandomPlan.run_id))
        .where(
            (RandomPlan.plan_type == plan_type)
            & (RandomPlan.workload_set == workload_set)
            & (RandomPlan.query_name == query)
        )
        .first()
    )
    run_id = (highest_run.run_id + 1) if highest_run.run_id else 1
    try_random_plans(workload_set, query, run_id, plan_type, max_plans=max_plans)


@app.command()
def resume(
    workload_set: str = typer.Option(),
    query: str = typer.Option(),
    run_id: int = typer.Option(),
    max_plans: Optional[int] = None,
    plan_type: PlanType = PlanType.NonCrossUniformOp,
):
    """Resume a run for the given query"""
    run_plans = (
        RandomPlan.select()
        .where(
            (RandomPlan.plan_type == plan_type)
            & (RandomPlan.workload_set == workload_set)
            & (RandomPlan.query_name == query)
            & (RandomPlan.plan_type == plan_type)
            & (RandomPlan.run_id == run_id)
        )
        .count()
    )
    if run_plans == 0:
        raise ValueError(f"No runs found for {query} {run_id}")
    else:
        try_random_plans(workload_set, query, run_id, plan_type, max_plans=max_plans)


@app.command()
def fill(
    workload_set: str = typer.Option(),
    query: str = typer.Option(),
    runs: int = 1,
    max_plans: int = 200,
    plan_type: PlanType = PlanType.NonCrossUniformOp,
):
    """Ensure there are at least `runs` runs with `max_plans` plans each for the given query"""
    for run_id in range(1, runs + 1):
        try_random_plans(workload_set, query, run_id, plan_type, max_plans=max_plans)


@app.command()
def fill_workload_set(
    workload_set: str = typer.Option(),
    runs: int = 1,
    max_plans: int = 4000,
    plan_type: PlanType = PlanType.NonCrossUniformOp,
    initial_timeout: str = "pg",
):
    """Ensure there are at least `runs` runs with `max_plans` plans each for all queries in the workload set"""
    workload_def_set = get_workload_set(workload_set)
    all_queries = list(workload_def_set.queries.keys())
    shuffle(all_queries)
    for query in all_queries:
        for run_id in range(1, runs + 1):
            try_random_plans(
                workload_set,
                query,
                run_id,
                plan_type,
                max_plans=max_plans,
                initial_timeout=initial_timeout,
            )


@app.command()
def optimize_until(
    workload_set: str = typer.Option(),
    query: str = typer.Option(),
    time: int = typer.Option(),
    initial_timeout: str = typer.Option(),
    plan_type: PlanType = PlanType.NonCrossUniformOp,
    max_plans_without_improvement: Optional[int] = None,
):
    """Optimize the given query until the cumulative time for all runs exceeds the given time"""
    try_random_plans(
        workload_set,
        query,
        1,
        plan_type,
        max_time=time,
        initial_timeout=initial_timeout,
        max_plans_without_improvement=max_plans_without_improvement,
    )


@app.command()
def fill_until(
    workload_set: str = typer.Option(),
    time: int = typer.Option(),
    initial_timeout: str = typer.Option(),
):
    workload_def_set = get_workload_set(workload_set)
    all_queries = list(workload_def_set.queries.keys())
    shuffle(all_queries)
    for query in all_queries:
        try_random_plans(
            workload_set,
            query,
            1,
            PlanType.NonCrossUniformOp,
            max_time=time,
            initial_timeout=initial_timeout,
        )


@app.command()
def fill_job():
    work = list(JOB_QUERIES_SORTED)
    shuffle(work)
    for query in work:
        try_random_plans(
            workload_set="JOB",
            query=query,
            run_id=1,
            plan_type=PlanType.NonCrossUniformOp,
            max_plans=4000,
            initial_timeout="pg",
        )
        # optimize_until(
        #     workload_set="JOB",
        #     query=query,
        #     time=60 * 60,
        #     initial_timeout="pg",
        #     # max_plans_without_improvement=500,
        # )


@app.command()
def fill_ceb():
    top, bot = top_bot_improvement("CEB_3K", 100)
    top_100_runtime = top_pg_runtime("CEB_3K", 100)
    work = list(set(top + bot + top_100_runtime))
    shuffle(work)
    for query in work:
        try_random_plans(
            workload_set="CEB_3K",
            query=query,
            run_id=1,
            plan_type=PlanType.NonCrossUniformOp,
            max_plans=4000,
            initial_timeout="pg",
        )
        # optimize_until(
        #     workload_set="CEB_3K",
        #     query=query,
        #     time=6 * 60 * 60,
        #     initial_timeout="pg",
        #     max_plans_without_improvement=500,
        # )


@app.command()
def fill_stack(version: str = "SO_FUTURE"):
    queries = stack_workload_queries(200, 1.0)
    shuffle(queries)

    for query in queries:
        try_random_plans(
            workload_set=version,
            query=query,
            run_id=1,
            plan_type=PlanType.NonCrossUniformOp,
            max_plans=4000,
            initial_timeout="pg",
        )
        # optimize_until(
        #     workload_set=version,
        #     query=query,
        #     time=6 * 60 * 60,
        #     initial_timeout="pg",
        #     max_plans_without_improvement=500,
        # )


@app.command()
def eval_status(eval_workload: EvalWorkload = typer.Option(), verbose: bool = False):
    for query in resolve_eval_queries(eval_workload):
        saturated = (
            SaturatedRandomRun.select()
            .where(SaturatedRandomRun.query_name == query)
            .count()
            > 0
        )
        if saturated:
            if verbose:
                print(f"{query} is saturated")
            continue

        total_runtime = (
            RandomPlan.select(fn.Sum(RandomPlan.runtime_secs))
            .where(RandomPlan.query_name == query)
            .scalar()
        )
        if total_runtime > 6 * 60 * 60:
            if verbose:
                print(f"{query} has exceeded 6 hours of runtime")
            continue

        run = (
            RandomPlan.select()
            .where(
                (RandomPlan.query_name == query)
                & (RandomPlan.plan_type == PlanType.NonCrossUniformOp)
            )
            .order_by(RandomPlan.plan_id)
        )
        last_improvement_index = 0
        current_best = float("inf")
        for i, plan in enumerate(run):
            if plan.runtime_secs < current_best:
                last_improvement_index = i
                current_best = plan.runtime_secs
        total_time = sum(plan.runtime_secs for plan in run)
        since_last_improvement = run.count() - last_improvement_index
        if last_improvement_index == 0:
            if since_last_improvement < 500 or verbose:
                print(
                    f"{query} has not improved at all in {run.count()} plans, {pretty_time(total_time)}"
                )
        else:
            if since_last_improvement < 500 or verbose:
                print(
                    f"{query} last improved {since_last_improvement}/{run.count()} plans ago, total time {pretty_time(total_time)}"
                )


@app.command()
def check_num_plans(
    eval_workload: EvalWorkload = typer.Option(), threshold: int = 4000
):
    workload_set_name = resolve_workload_set(eval_workload)
    needed = 0
    for query in resolve_eval_queries(eval_workload):
        saturated = (
            SaturatedRandomRun.select()
            .where(
                (SaturatedRandomRun.query_name == query)
                & (SaturatedRandomRun.workload_set == workload_set_name)
            )
            .exists()
        )
        if saturated:
            continue

        num_plans = (
            RandomPlan.select()
            .where(
                (RandomPlan.query_name == query)
                & (RandomPlan.workload_set == workload_set_name)
            )
            .count()
        )
        if num_plans < threshold:
            print(f"{query}: {num_plans} plans")
            needed += threshold - num_plans
    print(f"Need {needed} more plans")


@app.command()
def check_workload_remaining(
    eval_workload: EvalWorkload = typer.Option(), threshold: int = 4000
):
    queries = resolve_eval_queries(eval_workload)
    workload_set_name = resolve_workload_set(eval_workload)
    needed = (
        RandomPlan.select(
            RandomPlan.query_name, fn.Count(RandomPlan.plan_id).alias("plans")
        )
        .join(
            SaturatedRandomRun,
            on=(RandomPlan.query_name == SaturatedRandomRun.query_name),
            join_type=JOIN.LEFT_OUTER,
        )
        .where(
            (RandomPlan.query_name << queries)
            & (RandomPlan.workload_set == workload_set_name)
            & (SaturatedRandomRun.query_name.is_null())
        )
        .group_by(RandomPlan.query_name)
        .having(fn.Count(RandomPlan.query_name) < threshold)
    )
    total_needed = sum(threshold - query.plans for query in needed)
    print(f"Need {total_needed} more plans")


@app.command()
def clear_ceb():
    count = (
        RandomPlan.select()
        .where(RandomPlan.workload_set == "CEB_3K")
        .group_by(RandomPlan.query_name)
        .count()
    )
    print(f"Before: {count} queries")
    input("Are you sure?")
    top, _ = top_bot_improvement("CEB_3K", 100)
    RandomPlan.delete().where(
        (RandomPlan.query_name << top) & (RandomPlan.workload_set == "CEB_3K")
    ).execute()
    count = (
        RandomPlan.select()
        .where(RandomPlan.workload_set == "CEB_3K")
        .group_by(RandomPlan.query_name)
        .count()
    )
    print(f"After: {count} queries")


@app.command()
def plot(
    query: str,
    save_to: Optional[str] = None,
    plan_type: PlanType = PlanType.NonCrossUniformOp,
):
    """Plot the best time for each run of the given query"""
    max_run = (
        RandomPlan.select(fn.Max(RandomPlan.run_id))
        .where((RandomPlan.plan_type == plan_type) & (RandomPlan.query_name == query))
        .first()
    )

    series: list[tuple[list[float], list[float]]] = []
    for run_id in range(1, max_run.run_id + 1):
        runs = (
            RandomPlan.select(
                RandomPlan.run_id,
                RandomPlan.runtime_secs,
                RandomPlan.result,
                fn.Sum(RandomPlan.runtime_secs)
                .over(order_by=[RandomPlan.plan_id])
                .alias("cumulative_time"),
            )
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.query_name == query)
                & (RandomPlan.run_id == run_id)
            )
            .order_by(RandomPlan.run_id)
        )

        xs = []
        ys = []
        for run in runs:
            if run.result == ExecutionResult.Success:
                xs.append(run.cumulative_time)
                ys.append(run.runtime_secs)

        series.append((xs, ys))

    for run_id, (xs, ys) in enumerate(series):
        plt.plot(xs, ys, label=f"Run {run_id + 1}", marker="o")

    plt.axhline(
        y=postgres_time(query), color="orange", linestyle="--", label="Postgres"
    )
    plt.axhline(y=bao_optimal_time(query), color="green", linestyle="--", label="Bao")

    plt.title(f"Random plans, {query}")
    plt.xlabel("Cumulative Time (s)")
    plt.ylabel("Run time (s)")
    plt.legend()

    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()

    plt.clf()


@app.command()
def plot_all(plan_type: PlanType = PlanType.NonCrossUniformOp):
    OUT_DIR = os.path.join(os.path.dirname(__file__), f"plots/random_plans/{plan_type}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for query in tqdm(JOB_QUERIES_SORTED):
        if (
            RandomPlan.select()
            .where(
                (RandomPlan.plan_type == plan_type) & (RandomPlan.query_name == query)
            )
            .count()
            > 0
        ):
            plot(query, save_to=os.path.join(OUT_DIR, f"{query}.png"))


@app.command()
def export_best_runs(plan_type: PlanType = PlanType.NonCrossUniformOp):
    print("query_name;result;runtime_secs;plan")
    for query in JOB_QUERIES_SORTED:
        best_run = (
            RandomPlan.select(
                RandomPlan.run_id,
            )
            .where(
                (RandomPlan.plan_type == plan_type)
                & (RandomPlan.query_name == query)
                & (RandomPlan.result == ExecutionResult.Success)
            )
            .order_by(RandomPlan.runtime_secs)
            .limit(1)
        ).first()

        if best_run is not None:
            plans_in_run = (
                RandomPlan.select()
                .where(
                    (RandomPlan.plan_type == plan_type)
                    & (RandomPlan.query_name == query)
                    & (RandomPlan.run_id == best_run.run_id)
                )
                .order_by(RandomPlan.plan_id)
            )

            for plan in plans_in_run:
                print(
                    f"{query};{plan.result};{plan.runtime_secs};{','.join(str(c) for c in plan.encoded_plan)}"
                )


@app.command()
def ceb_top_vs_bao():
    workload_set = get_workload_set("CEB_3K")
    by_improvement = sorted(
        [
            (query, bao_optimal_time(query, "CEB_3K") / postgres_time(query))
            for query in workload_set.queries
        ],
        key=lambda x: x[1],
    )
    top_50 = by_improvement[:50]
    bot_50 = by_improvement[-50:]

    random_times = []
    failed_to_plan = 0

    for category, queries in [("Top 50", top_50), ("Bottom 50", bot_50)]:
        for query, _ in queries:
            random_run = (
                RandomPlan.select(RandomPlan.runtime_secs, RandomPlan.result)
                .where(
                    (RandomPlan.query_name == query)
                    & (RandomPlan.plan_type == PlanType.NonCrossUniformOp)
                )
                .order_by(RandomPlan.plan_id)
            )
            best_seen = float("inf")
            had_success = False
            cumulative_time = 0
            for plan in random_run:
                best_seen = min(best_seen, plan.runtime_secs)
                if plan.result == ExecutionResult.Success:
                    had_success = True
                cumulative_time += plan.runtime_secs
                if cumulative_time > 3600:
                    break
            random_times.append(best_seen)
            if not had_success:
                failed_to_plan += 1

        print(f"Failed to plan {failed_to_plan} queries")
        plt.suptitle(f"{category} queries")
        for i, agg in enumerate(Aggregate):
            plt.subplot(1, 3, i + 1)
            agg_func = AGG_FUNCS[agg]
            bao_times = [bao_optimal_time(query, "CEB_3K") for query, _ in queries]
            agg_bao = agg_func(bao_times)
            agg_random = agg_func(random_times)
            print(agg_random)
            plt.bar(["Bao", "Random"], [agg_bao, agg_random])
            plt.title(f"{agg}")
        plt.show()
        # plt.text(0.5, 0.5, f"Failed to plan {failed_to_plan} queries")
        # plt.savefig(
        #     f"db_eval/plots/random_plans/CEB_3K {category}.png".replace(" ", "_")
        # )


gpt4o_queries = [
    "CEB_2A682",
    "CEB_2A794",
    "CEB_11B156",
    "CEB_2A854",
    "CEB_9A116",
    "CEB_2B212",
    "CEB_9A31",
    "CEB_9B47",
    "CEB_2B411",
    "CEB_9A88",
    "CEB_3B47",
    "CEB_1A2661",
    "CEB_8A203",
    "CEB_10A63",
    "CEB_11B151",
    "CEB_1A2754",
    "CEB_11B143",
    "CEB_2A589",
    "CEB_6A376",
    "CEB_5A68",
    "CEB_1A1482",
    "CEB_2C255",
    "CEB_2A323",
    "CEB_1A1487",
    "CEB_11B157",
    "CEB_9B3",
    "CEB_2C187",
    "CEB_1A2946",
    "CEB_2A730",
    "CEB_9A128",
    "CEB_9B106",
    "CEB_1A2639",
    "CEB_1A376",
    "CEB_2A683",
    "CEB_11B12",
    "CEB_2C66",
    "CEB_5A691",
    "CEB_2C266",
    "CEB_3A1",
    "CEB_1A2556",
    "CEB_4A434",
    "CEB_3B56",
    "CEB_2B436",
    "CEB_6A247",
    "CEB_8A264",
    "CEB_2B278",
    "CEB_1A1521",
    "CEB_7A48",
    "CEB_9B113",
    "CEB_2B72",
    "CEB_11B61",
    "CEB_9A49",
    "CEB_2B441",
    "CEB_2C35",
    "CEB_2A302",
    "CEB_9A239",
    "CEB_11A22",
    "CEB_2B458",
    "CEB_9A121",
    "CEB_2B424",
    "CEB_9A10",
    "CEB_11A20",
    "CEB_10A29",
    "CEB_11A70",
    "CEB_5A701",
    "CEB_8A300",
    "CEB_9A87",
    "CEB_3B12",
    "CEB_1A2101",
    "CEB_1A1908",
    "CEB_2B304",
    "CEB_11A134",
    "CEB_5A1012",
    "CEB_11B106",
    "CEB_2A782",
    "CEB_1A2719",
    "CEB_1A2124",
    "CEB_2B179",
    "CEB_8A472",
    "CEB_11B119",
    "CEB_6A370",
    "CEB_10A0",
    "CEB_1A2247",
    "CEB_2A210",
    "CEB_2B320",
    "CEB_6A404",
    "CEB_2A578",
    "CEB_1A1423",
    "CEB_2A332",
    "CEB_2A240",
    "CEB_9A162",
    "CEB_9B12",
    "CEB_6A326",
    "CEB_1A84",
    "CEB_2A612",
    "CEB_2A517",
    "CEB_2B323",
    "CEB_2A591",
    "CEB_1A208",
    "CEB_9B51",
]


@app.command()
def fill_gpt4o():
    queries = list(gpt4o_queries)
    shuffle(queries)
    for query in queries:
        try_random_plans(
            workload_set="CEB_3K",
            query=query,
            run_id=1,
            plan_type=PlanType.NonCrossUniformOp,
            max_plans=50,
            initial_timeout="constant",
        )


@app.command()
def gpt4o_status():
    print("query_name;best_time_secs")
    for query in gpt4o_queries:
        best = (
            RandomPlan.select(fn.Min(RandomPlan.runtime_secs))
            .where(
                (RandomPlan.query_name == query)
                & (RandomPlan.plan_type == PlanType.NonCrossUniformOp)
                & (RandomPlan.result == ExecutionResult.Success)
            )
            .scalar()
        )
        print(f"{query};{best}")


@app.command()
def check_decreasing():
    for query in tqdm(RandomPlan.select(RandomPlan.query_name).distinct()):
        # for query in ["STACK_Q2-025"]:
        run = (
            RandomPlan.select()
            .where((RandomPlan.query_name == query.query_name))
            .order_by(RandomPlan.plan_id)
        )
        last_time = float("inf")
        cumulative_time = 0
        nondecreasing_run = 0
        nondecreasing_start_id = None
        nondecreasing_end_id = None
        past_initial_timeout = False
        for plan in run:
            if plan.result == "success" or plan.runtime_secs % 5 != 0:
                past_initial_timeout = True
            if not past_initial_timeout:
                continue

            cumulative_time += plan.runtime_secs
            if plan.runtime_secs <= last_time:
                last_time = plan.runtime_secs

                if nondecreasing_run > 0:
                    nondecreasing_end_id = plan.plan_id
                    if nondecreasing_run > 1:
                        affected_plans = (
                            RandomPlan.select()
                            .where(
                                (RandomPlan.plan_id >= nondecreasing_start_id)
                                & (RandomPlan.plan_id < nondecreasing_end_id)
                                & (RandomPlan.query_name == query.query_name)
                            )
                            .count()
                        )
                        tqdm.write(
                            f"Found nondecreasing run for {query.query_name} starting at {nondecreasing_start_id} ending at {nondecreasing_start_id}, {affected_plans} plans affected"
                        )
                        # input(f"\nWill update {affected_plans} plans:")
                    else:
                        tqdm.write(
                            f"Fixing single nondecreasing plan for {query.query_name} at {plan.plan_id}"
                        )

                    # Update the nondecreasing run with the last known good time
                    RandomPlan.update(runtime_secs=last_time).where(
                        (RandomPlan.plan_id >= nondecreasing_start_id)
                        & (RandomPlan.plan_id < nondecreasing_end_id)
                        & (RandomPlan.query_name == query.query_name)
                    ).execute()

                    nondecreasing_run = 0
                    nondecreasing_start_id = None
                    nondecreasing_end_id = None
            else:
                nondecreasing_run += 1
                if nondecreasing_start_id is None:
                    nondecreasing_start_id = plan.plan_id

        if nondecreasing_run > 1:
            affected_plans = (
                RandomPlan.select()
                .where(
                    (RandomPlan.plan_id >= nondecreasing_start_id)
                    & (RandomPlan.query_name == query.query_name)
                )
                .count()
            )
            tqdm.write(
                f"Found nondecreasing run for {query.query_name} starting at {nondecreasing_start_id}, {affected_plans} plans affected"
            )
            # input(f"Will update {affected_plans} plans:")

            # Update the nondecreasing run with the last known good time
            RandomPlan.update(runtime_secs=last_time).where(
                (RandomPlan.plan_id >= nondecreasing_start_id)
                & (RandomPlan.query_name == query.query_name)
            ).execute()


if __name__ == "__main__":
    app()
