import re
from enum import StrEnum, auto

import numpy as np
from workload.workloads import IMDB_WORKLOAD_SET


def query_key(query: str) -> tuple[int, str]:
    num, letter = re.match(r"JOB_(\d+)(.+)", query).groups()
    return int(num), letter


JOB_QUERIES_SORTED = list(
    sorted(
        (q for q in IMDB_WORKLOAD_SET.queries),
        key=query_key,
    )
)


def pretty_time(seconds: float) -> str:
    if seconds == 0:
        return "0"
    elif seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        if seconds % 60 == 0:
            return f"{seconds // 60:.0f}m"
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        if seconds % 3600 == 0:
            return f"{seconds // 3600:.0f}h"
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


def compact_time(seconds: float) -> str:
    if seconds == 0:
        return "0"
    elif seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m"
    else:
        minutes = (seconds % 3600) // 60
        minute_str = f"{minutes:.0f}m" if minutes else ""
        return f"{seconds // 3600:.0f}h{minute_str}"


class Aggregate(StrEnum):
    sum = auto()
    median = auto()
    p90 = auto()


AGG_FUNCS = {
    Aggregate.sum: np.sum,
    Aggregate.median: np.median,
    Aggregate.p90: lambda x: np.percentile(x, 90),
}
