import asyncio
import json
import pdb
import time
from itertools import repeat
from multiprocessing import Pool, cpu_count
from subprocess import run
from typing import Any, Iterable, Optional

import psycopg2
import psycopg2.pool

from logger.log import l

from .storage import PlanType

conn: Any = None


def init_worker():
    global conn
    # conn = psycopg2.connect(host="172.17.0.2", database="so", user="so", password="so")
    conn = psycopg2.connect(
        host="172.17.0.2", database="postgres", user="imdb", password="imdb"
    )


def get_explain(args: tuple[str, PlanType]) -> Optional[str]:
    query, plan_type = args
    setting = ""
    # it's very important that these end with semicolons
    match (plan_type):
        case PlanType.NO_HASH_JOIN:
            setting = "SET enable_hashjoin TO off;"
        case PlanType.NO_LOOP_JOIN:
            setting = "SET enable_nestloop TO off;"
        case PlanType.NO_SEQ_SCAN:
            setting = "SET enable_seqscan TO off;"
        case PlanType.NO_INDEX_SCAN:
            setting = "SET enable_indexscan TO off;"

    """Produces JSON output for EXPLAIN {`query`}"""
    if conn is None:
        l.error("EXPLAIN worker has no connection!")
        return None
    try:
        cur = conn.cursor()
        cur.execute(f"{setting} EXPLAIN (FORMAT JSON, SETTINGS ON) {query}")
        explain_out = cur.fetchall()
        if len(explain_out) == 0:
            l.error("EXPLAIN output empty")
            return None

        # Result is one column in one tuple
        return explain_out[0][0]
    except psycopg2.errors.UndefinedColumn as e:
        l.error(f"EXPLAIN failed: {e} on {query}")
        return None


def explain_batch(work: Iterable[tuple[str, PlanType]]):
    return pool.map(get_explain, work)


if __name__ == "__main__":
    size = 1000
    query = "SELECT * FROM link_type, cast_info, keyword, char_name, complete_cast, company_name, kind_type, comp_cast_type, company_type, movie_keyword, movie_companies, title, movie_link WHERE ((((((((((movie_link.link_type_id = link_type.id AND cast_info.person_role_id = char_name.id) AND cast_info.movie_id = title.id) AND movie_keyword.keyword_id = keyword.id) AND complete_cast.status_id = comp_cast_type.id) AND complete_cast.movie_id = title.id) AND movie_companies.company_id = company_name.id) AND title.kind_id = kind_type.id) AND movie_companies.company_type_id = company_type.id) AND movie_keyword.movie_id = title.id) AND movie_companies.movie_id = title.id) AND movie_link.linked_movie_id = title.id"

    start = time.time()
    with Pool(cpu_count(), initializer=init_worker) as p:
        p.map(
            get_explain,
            [(query, PlanType.NO_HASH_JOIN) for _ in range(size)],
        )
        end = time.time()
        # print("Parallel time:", end - start)
        # print("Parallel time per result:", (end - start) / size)

# This has to go on the bottom
# Pool workers can't see functions defined below the pool initialization
pool = Pool(cpu_count(), initializer=init_worker)
