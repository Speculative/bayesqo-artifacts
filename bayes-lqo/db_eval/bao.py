import pdb
import re
from itertools import product
from random import shuffle
from statistics import median

import typer
from peewee import fn
from psycopg import sql

from logger.log import l
from oracle.oracle import QueryTask, _default_plan, _resolve_codec
from training_data.codec import build_join_tree
from workload.workloads import OracleCodec, WorkloadSpec, get_workload_set

from .storage import BaoInitialization, BaoJoinHint, BaoPlan, BaoScanHint, PostgresPlan
from .utils import JOB_QUERIES_SORTED

app = typer.Typer(no_args_is_help=True)


def get_hints(join_hint: BaoJoinHint, scan_hint: BaoScanHint) -> list[str]:
    hint_statements = []

    match join_hint:
        case BaoJoinHint.NoHash:
            hint_statements.append("SET enable_hashjoin = off")
        case BaoJoinHint.NoMerge:
            hint_statements.append("SET enable_mergejoin = off")
        case BaoJoinHint.NoNestedLoops:
            hint_statements.append("SET enable_nestloop = off")
        case BaoJoinHint.NoHashNoMerge:
            hint_statements.append("SET enable_hashjoin = off")
            hint_statements.append("SET enable_mergejoin = off")
        case BaoJoinHint.NoHashNoNestedLoops:
            hint_statements.append("SET enable_hashjoin = off")
            hint_statements.append("SET enable_nestloop = off")
        case BaoJoinHint.NoMergeNoNestedLoops:
            hint_statements.append("SET enable_mergejoin = off")
            hint_statements.append("SET enable_nestloop = off")

    match scan_hint:
        case BaoScanHint.NoIndex:
            hint_statements.append("SET enable_indexscan = off")
        case BaoScanHint.NoSeq:
            hint_statements.append("SET enable_seqscan = off")
        case BaoScanHint.NoIndexOnly:
            hint_statements.append("SET enable_indexonlyscan = off")
        case BaoScanHint.NoIndexNoSeq:
            hint_statements.append("SET enable_indexscan = off")
            hint_statements.append("SET enable_seqscan = off")
        case BaoScanHint.NoIndexNoIndexOnly:
            hint_statements.append("SET enable_indexscan = off")
            hint_statements.append("SET enable_indexonlyscan = off")
        case BaoScanHint.NoSeqNoIndexOnly:
            hint_statements.append("SET enable_seqscan = off")
            hint_statements.append("SET enable_indexonlyscan = off")

    return hint_statements


@app.command()
def sample(
    query: str,
    join_hint: BaoJoinHint,
    scan_hint: BaoScanHint,
    workload_set: str = typer.Option("JOB"),
):
    workload_def_set = get_workload_set(workload_set)
    spec = workload_def_set.queries[query]
    workload = WorkloadSpec.from_definition(spec, OracleCodec.Aliases)

    hint_statements = get_hints(join_hint, scan_hint)
    query_statements = hint_statements + [
        f"EXPLAIN (ANALYZE, FORMAT JSON, SETTINGS ON, TIMING OFF) {_default_plan(spec)}",
    ]
    db = "imdb"
    if workload_set == "SO_FUTURE":
        db = "so_future"
    elif workload_set == "SO_PAST":
        db = "so_past"
    db_user = "imdb"
    if workload_set in ["SO_FUTURE", "SO_PAST"]:
        db_user = "so"

    sql = "; ".join(query_statements)
    task = QueryTask(
        sql,
        10 * 60 * 1000,
        db=db,
        user=db_user,
    )
    task.submit()
    result = task.result()
    if result is None:
        raise ValueError("Somehow got a None result from the task!")

    if result["status"] == "timeout":
        l.warning(f"Query for {query} timed out")
        return
    elif result["status"] == "failed":
        l.error(f"Query for {query} failed: {result['message']}")
        return

    if not "result" in result:
        l.error("Worker did not return a result!")
    join_tree = build_join_tree(result["result"][0][0][0]["Plan"])
    codec_inst = _resolve_codec(workload)
    encoded = codec_inst.encode(join_tree)

    elapsed = result["duration (ns)"] / 1_000_000_000
    l.info(f"Query for {query} ({join_hint}, {scan_hint}) in {elapsed} secs")
    BaoPlan.create(
        workload_set=workload_set,
        query_name=query,
        join_hint=join_hint,
        scan_hint=scan_hint,
        runtime_secs=elapsed,
        encoded_plan=encoded,
    )


@app.command()
def fill(samples: int = 5, workload_set: str = typer.Option("JOB")):
    workload_def_set = get_workload_set(workload_set)
    all_queries = list(workload_def_set.queries.keys())
    shuffle(all_queries)
    for query in all_queries:
        for join_hint, scan_hint in product(list(BaoJoinHint), list(BaoScanHint)):
            existing = (
                BaoPlan.select()
                .where(
                    (BaoPlan.query_name == query)
                    & (BaoPlan.join_hint == join_hint)
                    & (BaoPlan.scan_hint == scan_hint)
                )
                .count()
            )
            # Temporary, for when we have parallel runs
            if existing >= samples:
                continue
            # l.info(f"Sampling {query}: {existing} existing samples")
            for _ in range(samples - existing):
                existing = (
                    BaoPlan.select()
                    .where(
                        (BaoPlan.query_name == query)
                        & (BaoPlan.join_hint == join_hint)
                        & (BaoPlan.scan_hint == scan_hint)
                    )
                    .count()
                )
                if existing >= samples:
                    break
                sample(query, join_hint, scan_hint, workload_set=workload_set)


@app.command()
def summarize(workload_set: str = typer.Option("JOB")):
    workload_def_set = get_workload_set(workload_set)
    for query in workload_def_set.queries:
        print(query)
        for join_hint, scan_hint in product(list(BaoJoinHint), list(BaoScanHint)):
            count = (
                BaoPlan.select()
                .where(
                    (BaoPlan.query_name == query)
                    & (BaoPlan.join_hint == join_hint)
                    & (BaoPlan.scan_hint == scan_hint)
                )
                .count()
            )
            average = (
                BaoPlan.select(fn.AVG(BaoPlan.runtime_secs))
                .where(
                    (BaoPlan.query_name == query)
                    & (BaoPlan.join_hint == join_hint)
                    & (BaoPlan.scan_hint == scan_hint)
                )
                .scalar()
            ) or 0
            print(
                f"\t{join_hint}, {scan_hint}: {count} samples, avg {average:.2f} secs"
            )


def bao_optimal_time(query: str, workload_set: str) -> float:
    average_runtimes = (
        BaoPlan.select(BaoPlan, fn.AVG(BaoPlan.runtime_secs).alias("runtime_avg"))
        .where((BaoPlan.query_name == query) & (BaoPlan.workload_set == workload_set))
        .group_by(BaoPlan.join_hint, BaoPlan.scan_hint)
    )
    result = (
        BaoPlan.select(
            average_runtimes.c.query_name,
            average_runtimes.c.join_hint,
            average_runtimes.c.scan_hint,
            fn.MIN(average_runtimes.c.runtime_avg).alias("min_runtime"),
        )
        .from_(average_runtimes)
        .first()
    )
    if result.min_runtime is not None:
        return result.min_runtime

    result = (
        BaoInitialization.select(fn.MIN(BaoInitialization.runtime_secs))
        .where(
            (BaoInitialization.query_name == query)
            & (BaoInitialization.workload_set == workload_set)
        )
        .scalar()
    )
    if result is not None:
        return result

    raise ValueError(f"No Bao optimal time found for {query} in {workload_set}")


def hint_performance(
    query: str, join_hint: BaoJoinHint, scan_hint: BaoScanHint
) -> float:
    return (
        BaoPlan.select(fn.AVG(BaoPlan.runtime_secs))
        .where(
            (BaoPlan.query_name == query)
            & (BaoPlan.join_hint == join_hint)
            & (BaoPlan.scan_hint == scan_hint)
        )
        .scalar()
    )


def postgres_time(query: str, workload_set: str) -> float:
    pg_time = (
        BaoPlan.select(fn.AVG(BaoPlan.runtime_secs))
        .where(
            (BaoPlan.query_name == query)
            & (BaoPlan.workload_set == workload_set)
            & (BaoPlan.join_hint == BaoJoinHint.NoHint)
            & (BaoPlan.scan_hint == BaoScanHint.NoHint)
        )
        .scalar()
    )
    if pg_time is not None:
        return pg_time

    maybe_censored_pg_time = (
        BaoInitialization.select(BaoInitialization.runtime_secs)
        .where(
            (BaoInitialization.query_name == query)
            & (BaoInitialization.workload_set == workload_set)
            & (BaoInitialization.join_hint == BaoJoinHint.NoHint)
            & (BaoInitialization.scan_hint == BaoScanHint.NoHint)
        )
        .scalar()
    )
    if maybe_censored_pg_time is not None:
        return maybe_censored_pg_time

    return None
    # return (
    #     PostgresPlan.select(fn.AVG(PostgresPlan.runtime_secs))
    #     .where(PostgresPlan.query_name == query)
    #     .scalar()
    # )


@app.command()
def show_optimal(workload_set: str = typer.Option()):
    workload_set_obj = get_workload_set(workload_set)
    print("query_name,bao_optimal_time")
    for query in workload_set_obj.queries:
        print(f"{query},{bao_optimal_time(query, workload_set)}")


@app.command()
def query_hint_ranks(query: str):
    result = (
        BaoPlan.select(
            BaoPlan.join_hint,
            BaoPlan.scan_hint,
            fn.AVG(BaoPlan.runtime_secs).alias("runtime_avg"),
        )
        .where(BaoPlan.query_name == query)
        .group_by(BaoPlan.join_hint, BaoPlan.scan_hint)
        .order_by(fn.AVG(BaoPlan.runtime_secs))
    )

    return [(row.join_hint, row.scan_hint, rank + 1) for rank, row in enumerate(result)]


def ranked_hint_badness():
    """Least bad hints first, ordered by worst case rank, breaking ties with median rank."""
    query_ranks: dict[tuple[BaoJoinHint, BaoScanHint], list[int]] = {}
    for query in JOB_QUERIES_SORTED:
        ranks = query_hint_ranks(query)
        for join_hint, scan_hint, rank in ranks:
            query_ranks[(join_hint, scan_hint)] = query_ranks.get(
                (join_hint, scan_hint), []
            )
            query_ranks[(join_hint, scan_hint)].append(rank)
    worst_rankings = list(
        sorted(query_ranks.items(), key=lambda x: (max(x[1]), median(x[1])))
    )
    return [x[0] for x in worst_rankings]


@app.command()
def show_hint_badness():
    for join_hint, scan_hint in ranked_hint_badness():
        print(join_hint, scan_hint)


@app.command()
def plot_runtimes_hist():
    import matplotlib.pyplot as plt

    runtimes = [
        p.runtime_avg
        for p in BaoPlan.select(
            fn.AVG(BaoPlan.runtime_secs).alias("runtime_avg")
        ).group_by(BaoPlan.query_name, BaoPlan.join_hint, BaoPlan.scan_hint)
    ]
    values, bins, bars = plt.hist(runtimes)
    plt.bar_label(bars)
    plt.title("Number of plans by average runtime")
    plt.ylabel("Frequency")
    plt.xlabel("Average runtime (s)")
    plt.show()


def top_bot_improvement(
    workload_set: str = "CEB_3K",
    n: int = 50,
):
    workload_set_obj = get_workload_set(workload_set)
    improvements = sorted(
        [
            (
                query,
                bao_optimal_time(query, workload_set)
                / postgres_time(query, workload_set),
            )
            for query in workload_set_obj.queries
        ],
        key=lambda x: x[1],
    )
    top = [query for query, _ in improvements[:n]]
    bot = [query for query, _ in improvements[-n:]]
    return top, bot


def top_pg_runtime(
    workload_set: str = "CEB_3K",
    n: int = 50,
):
    workload_set_obj = get_workload_set(workload_set)
    runtimes = sorted(
        [
            (query, postgres_time(query, workload_set))
            for query in workload_set_obj.queries
        ],
        key=lambda x: x[1],
    )
    return [query for query, _ in runtimes[-n:]]


@app.command()
def show_top_improvement(workload_set: str = "CEB_3K", n: int = 50):
    top, _ = top_bot_improvement(workload_set=workload_set, n=n)
    for query in top:
        print(query)


@app.command()
def show_bot_improvement(workload_set: str = "CEB_3K", n: int = 50):
    _, bot = top_bot_improvement(workload_set=workload_set, n=n)
    for query in bot:
        print(query)


def show_top_pg_runtime(workload_set: str = "CEB_3K", n: int = 50):
    top = top_pg_runtime(workload_set=workload_set, n=n)
    for query in top:
        print(query)


@app.command()
def show_ceb_workload():
    top, bot = top_bot_improvement(workload_set="CEB_3K", n=100)
    top_100 = top_pg_runtime("CEB_3K", 100)
    all_queries = set(top + bot + top_100)
    template_counts = {}
    for query in all_queries:
        print(query)
        match = re.match(r"CEB_(\d+[ABC]).*", query)
        template = match.group(1)
        template_counts.setdefault(template, 0)
        template_counts[template] += 1
    for template, count in sorted(
        template_counts.items(), key=lambda x: (int(x[0][:-1]), x[0][-1])
    ):
        print(f"{template}: {count}")


@app.command()
def write_ceb_workload_queries():
    top, bot = top_bot_improvement(workload_set="CEB_3K", n=100)
    top_100 = top_pg_runtime("CEB_3K", 100)
    with open("ceb_top_100_queries.txt", "w") as f:
        for query in top:
            f.write(query + "\n")
    with open("ceb_bot_100_queries.txt", "w") as f:
        for query in bot:
            f.write(query + "\n")
    with open("ceb_pg_runtime_100_queries.txt", "w") as f:
        for query in top_100:
            if query == "CEB_11B82":
                continue
            f.write(query + "\n")


@app.command()
def check_workload_pg(workload_set: str = typer.Option()):
    workload_set_obj = get_workload_set(workload_set)
    for query in workload_set_obj.queries:
        print(f"{query}: {postgres_time(query, workload_set)}")


def invert_hint(hint: str) -> tuple[set[BaoJoinHint], set[BaoScanHint]]:
    all_join_hints = set(BaoJoinHint)
    all_scan_hints = set(BaoScanHint)
    match hint:
        case "hashjoin":
            return {
                BaoJoinHint.NoMerge,
                BaoJoinHint.NoNestedLoops,
                BaoJoinHint.NoMergeNoNestedLoops,
            }, all_scan_hints
        case "mergejoin":
            return {
                BaoJoinHint.NoHash,
                BaoJoinHint.NoNestedLoops,
                BaoJoinHint.NoHashNoNestedLoops,
            }, all_scan_hints
        case "nestloop":
            return {
                BaoJoinHint.NoHash,
                BaoJoinHint.NoMerge,
                BaoJoinHint.NoHashNoMerge,
            }, all_scan_hints
        case "indexonlyscan":
            return all_join_hints, {
                BaoScanHint.NoIndex,
                BaoScanHint.NoSeq,
                BaoScanHint.NoIndexNoSeq,
            }
        case "indexscan":
            return all_join_hints, {
                BaoScanHint.NoSeq,
                BaoScanHint.NoIndexOnly,
                BaoScanHint.NoSeqNoIndexOnly,
            }
        case "seqscan":
            return all_join_hints, {
                BaoScanHint.NoIndex,
                BaoScanHint.NoIndexOnly,
                BaoScanHint.NoIndexNoIndexOnly,
            }
        case _:
            raise ValueError(f"Unknown hint: {hint}")


def combine_join_hints(join_hints: set[BaoJoinHint]) -> BaoJoinHint:
    if len(join_hints) == 1:
        return join_hints.pop()
    elif join_hints in [
        {BaoJoinHint.NoHash, BaoJoinHint.NoMerge, BaoJoinHint.NoHashNoMerge},
        {BaoJoinHint.NoHash, BaoJoinHint.NoMerge},
    ]:
        return BaoJoinHint.NoHashNoMerge
    elif join_hints in [
        {
            BaoJoinHint.NoHash,
            BaoJoinHint.NoNestedLoops,
            BaoJoinHint.NoHashNoNestedLoops,
        },
        {BaoJoinHint.NoHash, BaoJoinHint.NoNestedLoops},
    ]:
        return BaoJoinHint.NoHashNoNestedLoops
    elif join_hints in [
        {
            BaoJoinHint.NoMerge,
            BaoJoinHint.NoNestedLoops,
            BaoJoinHint.NoMergeNoNestedLoops,
        },
        {BaoJoinHint.NoMerge, BaoJoinHint.NoNestedLoops},
    ]:
        return BaoJoinHint.NoMergeNoNestedLoops
    elif len(join_hints) == 0:
        return BaoJoinHint.NoHint
    else:
        raise ValueError(f"Can't combine join hints: {join_hints}")


def combine_scan_hints(scan_hints: set[BaoScanHint]) -> BaoScanHint:
    if len(scan_hints) == 1:
        return scan_hints.pop()
    elif scan_hints in [
        {BaoScanHint.NoIndex, BaoScanHint.NoSeq, BaoScanHint.NoIndexNoSeq},
        {BaoScanHint.NoIndex, BaoScanHint.NoSeq},
    ]:
        return BaoScanHint.NoIndexNoSeq
    elif scan_hints in [
        {
            BaoScanHint.NoIndex,
            BaoScanHint.NoIndexOnly,
            BaoScanHint.NoIndexNoIndexOnly,
        },
        {BaoScanHint.NoIndex, BaoScanHint.NoIndexOnly},
    ]:
        return BaoScanHint.NoIndexNoIndexOnly
    elif scan_hints in [
        {
            BaoScanHint.NoSeq,
            BaoScanHint.NoIndexOnly,
            BaoScanHint.NoSeqNoIndexOnly,
        },
        {BaoScanHint.NoSeq, BaoScanHint.NoIndexOnly},
    ]:
        return BaoScanHint.NoSeqNoIndexOnly
    elif len(scan_hints) == 0:
        return BaoScanHint.NoHint
    else:
        raise ValueError(f"Can't combine scan hints: {scan_hints}")


def limeqo_hint_order() -> list[tuple[BaoJoinHint, BaoScanHint]]:
    limeqo_hints = [(BaoJoinHint.NoHint, BaoScanHint.NoHint)]
    for hint_str in [
        "hashjoin,indexonlyscan",
        "hashjoin,indexonlyscan,indexscan",
        "hashjoin,indexonlyscan,indexscan,mergejoin",
        "hashjoin,indexonlyscan,indexscan,mergejoin,nestloop",
        "hashjoin,indexonlyscan,indexscan,mergejoin,seqscan",
        "hashjoin,indexonlyscan,indexscan,nestloop",
        "hashjoin,indexonlyscan,indexscan,nestloop,seqscan",
        "hashjoin,indexonlyscan,indexscan,seqscan",
        "hashjoin,indexonlyscan,mergejoin",
        "hashjoin,indexonlyscan,mergejoin,nestloop",
        "hashjoin,indexonlyscan,mergejoin,nestloop,seqscan",
        "hashjoin,indexonlyscan,mergejoin,seqscan",
        "hashjoin,indexonlyscan,nestloop",
        "hashjoin,indexonlyscan,nestloop,seqscan",
        "hashjoin,indexonlyscan,seqscan",
        "hashjoin,indexscan",
        "hashjoin,indexscan,mergejoin",
        "hashjoin,indexscan,mergejoin,nestloop",
        "hashjoin,indexscan,mergejoin,nestloop,seqscan",
        "hashjoin,indexscan,mergejoin,seqscan",
        "hashjoin,indexscan,nestloop",
        "hashjoin,indexscan,nestloop,seqscan",
        "hashjoin,indexscan,seqscan",
        "hashjoin,mergejoin,nestloop,seqscan",
        "hashjoin,mergejoin,seqscan",
        "hashjoin,nestloop,seqscan",
        "hashjoin,seqscan",
        "indexonlyscan,indexscan,mergejoin",
        "indexonlyscan,indexscan,mergejoin,nestloop",
        "indexonlyscan,indexscan,mergejoin,nestloop,seqscan",
        "indexonlyscan,indexscan,mergejoin,seqscan",
        "indexonlyscan,indexscan,nestloop",
        "indexonlyscan,indexscan,nestloop,seqscan",
        "indexonlyscan,mergejoin",
        "indexonlyscan,mergejoin,nestloop",
        "indexonlyscan,mergejoin,nestloop,seqscan",
        "indexonlyscan,mergejoin,seqscan",
        "indexonlyscan,nestloop",
        "indexonlyscan,nestloop,seqscan",
        "indexscan,mergejoin",
        "indexscan,mergejoin,nestloop",
        "indexscan,mergejoin,nestloop,seqscan",
        "indexscan,mergejoin,seqscan",
        "indexscan,nestloop",
        "indexscan,nestloop,seqscan",
        "mergejoin,nestloop,seqscan",
        "mergejoin,seqscan",
        "nestloop,seqscan",
    ]:
        hints = hint_str.split(",")
        join_hints, scan_hints = zip(*[invert_hint(hint) for hint in hints])
        unified_join_hints = set.intersection(*join_hints)
        unified_scan_hints = set.intersection(*scan_hints)
        limeqo_hints.append(
            (
                combine_join_hints(unified_join_hints),
                combine_scan_hints(unified_scan_hints),
            )
        )
    return limeqo_hints


if __name__ == "__main__":
    app()
    # hints = limeqo_hint_order()
    # for join_hint, scan_hint in product(list(BaoJoinHint), list(BaoScanHint)):
    #     if (join_hint, scan_hint) not in hints:
    #         print("❌", join_hint, scan_hint)
    #     else:
    #         print("✅", join_hint, scan_hint)
