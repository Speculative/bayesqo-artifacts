import pdb
import re
import time
from multiprocessing import Pool, Queue
from random import shuffle

import typer
from peewee import fn

from logger.log import l
from oracle.oracle import _default_plan
from oracle.pg_celery_worker.pg_worker import tasks
from workload.workloads import IMDB_WORKLOAD_SET, get_workload_set

from .storage import PostgresPlan
from .utils import JOB_QUERIES_SORTED

app = typer.Typer(no_args_is_help=True)


@app.command()
def sample(query: str):
    spec = IMDB_WORKLOAD_SET.queries[query]
    query_sql = _default_plan(spec)
    result = tasks.pg_execute_query_high.delay(query_sql, 10 * 60 * 1000).get()

    if result["status"] == "timeout":
        l.warning(f"Query for {query} timed out")
        return
    elif result["status"] == "failed":
        l.error(f"Query for {query} failed: {result['message']}")
        return

    elapsed = result["duration (ns)"] / 1_000_000_000
    l.info(f"Query for {query} in {elapsed} secs")
    PostgresPlan.create(query_name=query, runtime_secs=elapsed)


@app.command()
def fill(samples: int = 5):
    for query in JOB_QUERIES_SORTED:
        existing = PostgresPlan.select().where(PostgresPlan.query_name == query).count()
        l.info(f"Sampling {query}: {existing} existing samples")
        for _ in range(samples - existing):
            sample(query)


@app.command()
def summarize():
    for query in JOB_QUERIES_SORTED:
        count = PostgresPlan.select().where(PostgresPlan.query_name == query).count()
        average = (
            PostgresPlan.select(fn.AVG(PostgresPlan.runtime_secs))
            .where(PostgresPlan.query_name == query)
            .scalar()
        ) or 0
        print(f"{query}: {count} samples, avg {average:.2f} secs")


def postgres_time(query: str):
    postgres_avg = (
        PostgresPlan.select(fn.AVG(PostgresPlan.runtime_secs))
        .where(PostgresPlan.query_name == query)
        .scalar()
    )

    return postgres_avg


if __name__ == "__main__":
    app()
