import csv
import io
import json
import os
import pdb
from abc import ABC, abstractmethod
from typing import Any, Optional

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore

from codec.codec import (
    AliasesCodec,
    Codec,
    JoinOperator,
    JoinTree,
    JoinTreeBranch,
    JoinTreeLeaf,
    StackMachineCodec,
    StackMachineWithOperatorsCodec,
)
from logger.log import l
from workload.schema import build_table_order

from .storage import (
    AliasWorkloadPlan,
    AliasWorkloadQuery,
    PlanType,
    WorkloadPlan,
    WorkloadQuery,
)


def resolve_join_operator(join_type: str) -> JoinOperator:
    match join_type:
        case "Nested Loop":
            return JoinOperator.NestLoop
        case "Merge Join":
            return JoinOperator.MergeJoin
        case "Hash Join":
            return JoinOperator.HashJoin
        case _:
            raise ValueError(f"Unknown join type {join_type}")


def _build_join_tree(explain: dict) -> JoinTree | None:
    if not "Node Type" in explain:
        raise ValueError("Node without type?")
    match explain["Node Type"]:
        case "Nested Loop" | "Merge Join" | "Hash Join":
            sub_plans = explain["Plans"]
            left = _build_join_tree(sub_plans[0])
            right = _build_join_tree(sub_plans[1])
            match (left, right):
                case (None, None):
                    return None
                case (None, r):
                    return r
                case (l, None):
                    return l
                case (l, r):
                    return JoinTreeBranch(
                        l,
                        r,
                        op=resolve_join_operator(explain["Node Type"]),
                    )
        case "Seq Scan" | "Index Scan" | "Bitmap Heap Scan" | "Index Only Scan":
            if "Alias" in explain:
                full_alias = explain["Alias"]
                table = explain["Relation Name"]
                try:
                    if "." in full_alias:
                        full_alias = full_alias.split(".", 1)[1]
                    alias = int(full_alias[len(table) :])
                    return JoinTreeLeaf(table, alias)
                except:
                    # pdb.set_trace()
                    # This is caused by subqueries, which should be explicitly excluded from the JoinTree
                    return None
            return JoinTreeLeaf(explain["Relation Name"])
        case _:
            if "Plans" in explain:
                return _build_join_tree(explain["Plans"][0])
            raise ValueError(f"Unknown node type {explain['Node Type']}")


def build_join_tree(explain: dict) -> JoinTree:
    res = _build_join_tree(explain)
    if res is None:
        pdb.set_trace()
        raise ValueError("Could not build join tree")
    return res


def generate_perturbations(join_tree: JoinTreeBranch):
    perturbations = [(join_tree, "postgres")]
    if join_tree.is_left_deep() or join_tree.is_right_deep():
        zigzag = join_tree.clone()
        zigzag.left, zigzag.right = zigzag.right, zigzag.left
        perturbations.append((zigzag, "zigzag"))

        zigzigzag = join_tree.clone()
        target: Any = zigzigzag.left if zigzigzag.is_left_deep() else zigzigzag.right
        if isinstance(target, JoinTreeBranch):
            target.left, target.right = target.right, target.left
            perturbations.append((zigzigzag, "zigzigzag"))

    if join_tree.is_zigzag() or join_tree.is_right_deep():
        left_deep = join_tree.clone()
        curr = left_deep
        while isinstance(curr, JoinTreeBranch):
            if isinstance(curr.right, JoinTreeBranch):
                curr.left, curr.right = curr.right, curr.left
            curr = curr.left  # type: ignore
        perturbations.append((left_deep, "left_deep"))
    return perturbations


def encode_plans(codec: Codec):
    total_seen = 0

    os.makedirs("encoded", exist_ok=True)
    for join_size in range(2, 18):
        plans_of_size = 0
        left_deep = 0
        right_deep = 0
        zigzag = 0
        bushy = 0
        variants = 0

        with open(f"encoded/{join_size}.csv", "w+") as out_file:
            writer = csv.writer(out_file, delimiter=";")

            for plan in (
                WorkloadPlan.select(WorkloadPlan, WorkloadQuery)
                .join(WorkloadQuery)
                .where(WorkloadQuery.num_joins == join_size)
                .order_by(WorkloadPlan.plan_type)
            ):
                explain = json.loads(plan.plan_json)
                if isinstance(explain, list):
                    explain = explain[0]
                join_tree = build_join_tree(explain["Plan"])

                # Accounting, just for interesting statistics
                plans_of_size += 1
                if join_tree.is_left_deep():
                    left_deep += 1
                elif join_tree.is_right_deep():
                    right_deep += 1
                elif join_tree.is_zigzag():
                    zigzag += 1
                else:
                    bushy += 1

                if isinstance(join_tree, JoinTreeBranch):
                    perturbations_for_plan = generate_perturbations(join_tree)

                    for plan_tree, perturbation_type in perturbations_for_plan:
                        encoded = codec.encode(plan_tree)
                        out = ",".join([str(symbol) for symbol in encoded])
                        writer.writerow(
                            (plan.query_id, plan.plan_type, perturbation_type, out)
                        )

                    variants += len(perturbations_for_plan)
                    total_seen += len(perturbations_for_plan)
                else:
                    print("Somehow got a leaf plan?")
            # print(join_size, total_seen, end="\r")

        print(
            f"Size {join_size}: {plans_of_size} postgres, {variants - plans_of_size} variants. "
        )
        print(
            f"\t{' ' if join_size > 9 else ''}{left_deep} left deep, {right_deep} right deep, {zigzag} zigzag, {bushy} bushy"
        )
    # print(end="\x1b[1K")
    # print(total_seen, "total plans (with variants)")


def encode_alias_plans(codec: Codec, schema: str, directory: str = "encoded"):
    total_seen = 0

    os.makedirs(directory, exist_ok=True)
    for plan_type in PlanType:
        plans_of_size = 0
        left_deep = 0
        right_deep = 0
        zigzag = 0
        bushy = 0
        variants = 0

        total_plans = (
            AliasWorkloadPlan.select()
            .where(
                (AliasWorkloadPlan.plan_type == plan_type)
                & (AliasWorkloadPlan.schema == schema)
            )
            .count()
        )

        num_encoded = 0
        with open(f"{directory}/{plan_type}.csv", "w+") as out_file:
            writer = csv.writer(out_file, delimiter=";")

            for plan in (
                AliasWorkloadPlan.select()
                .join(AliasWorkloadQuery)
                .where(
                    (AliasWorkloadPlan.plan_type == plan_type)
                    & (AliasWorkloadPlan.schema == schema)
                )
            ):
                explain = json.loads(plan.plan_json)
                if isinstance(explain, list):
                    explain = explain[0]
                join_tree = build_join_tree(explain["Plan"])

                # Round trip encoding test
                try:
                    encoded = codec.encode(join_tree)
                    decoded = codec.decode(plan.query_id.join_key, encoded)
                    assert join_tree == decoded
                except:
                    pdb.set_trace()

                # Accounting, just for interesting statistics
                plans_of_size += 1
                if join_tree.is_left_deep():
                    left_deep += 1
                elif join_tree.is_right_deep():
                    right_deep += 1
                elif join_tree.is_zigzag():
                    zigzag += 1
                else:
                    bushy += 1

                if isinstance(join_tree, JoinTreeBranch):
                    # perturbations_for_plan = generate_perturbations(join_tree)

                    # for plan_tree, perturbation_type in perturbations_for_plan:
                    #     encoded = codec.encode(plan_tree)
                    #     out = ",".join([str(symbol) for symbol in encoded])
                    #     writer.writerow(
                    #         (plan.query_id, plan.plan_type, perturbation_type, out)
                    #     )
                    encoded = codec.encode(join_tree)
                    out = ",".join([str(symbol) for symbol in encoded])
                    writer.writerow((plan.query_id, plan.plan_type, out))

                    # variants += len(perturbations_for_plan)
                    # total_seen += len(perturbations_for_plan)
                else:
                    print("Somehow got a leaf plan?")

                num_encoded += 1
                if num_encoded % 1000 == 0:
                    l.info(f"{num_encoded}/{total_plans} encoded")
            # print(join_size, total_seen, end="\r")
        l.info(
            f"{left_deep} left deep, {right_deep} right deep, {zigzag} zigzag, {bushy} bushy"
        )


if __name__ == "__main__":
    codec = AliasesCodec(build_table_order("workload/dsb/schema.sql"))
    encode_alias_plans(codec, schema="DSB", directory="dsb_encoded")
