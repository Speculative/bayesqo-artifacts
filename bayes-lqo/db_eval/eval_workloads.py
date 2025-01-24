from enum import StrEnum
from typing import assert_never

from workload.workloads import get_workload_set

from .bao import top_bot_improvement, top_pg_runtime
from .stack import workload_queries as stack_queries


class EvalWorkload(StrEnum):
    JOB = "JOB"
    CEB = "CEB"
    STACK_50 = "STACK-50"
    STACK_200 = "STACK-200"
    DSB = "DSB"


def resolve_eval_queries(eval_workload: EvalWorkload) -> list[str]:
    if eval_workload == EvalWorkload.JOB:
        return list(query for query in get_workload_set("JOB").queries.keys())
    elif eval_workload == EvalWorkload.CEB:
        queries = set()
        queries.update(top_pg_runtime("CEB_3K", 100))
        top_improved, bot_improved = top_bot_improvement("CEB_3K", 100)
        queries.update(top_improved)
        queries.update(bot_improved)
        return list(queries)
    elif eval_workload == EvalWorkload.STACK_50:
        return list(
            query
            for query in stack_queries(target=50, min_runtime=1.0)
            if query != "STACK_Q16-0043"
        )
    elif eval_workload == EvalWorkload.STACK_200:
        return list(query for query in stack_queries(target=200, min_runtime=1.0))
    elif eval_workload == EvalWorkload.DSB:
        return list(query for query in get_workload_set("DSB").queries.keys())
    else:
        raise ValueError(f"Unknown workload: {eval_workload}")


def resolve_workload_set(eval_workload: EvalWorkload) -> str:
    match eval_workload:
        case EvalWorkload.JOB:
            return "JOB"
        case EvalWorkload.CEB:
            return "CEB_3K"
        case EvalWorkload.STACK_50 | EvalWorkload.STACK_200:
            return "SO_FUTURE"
        case EvalWorkload.DSB:
            return "DSB"
        case _:
            raise ValueError(f"Unknown workload: {eval_workload}")


def nice_title(eval_workload: EvalWorkload) -> str:
    match eval_workload:
        case EvalWorkload.JOB:
            return "JOB"
        case EvalWorkload.CEB:
            return "CEB-3K"
        case EvalWorkload.STACK_50:
            return "StackOverflow 50"
        case EvalWorkload.STACK_200:
            return "StackOverflow 200"
        case EvalWorkload.DSB:
            return "DSB"
        case _:
            raise ValueError(f"Unknown workload: {eval_workload}")
