import json
import os

import pandas as pd


def balsa_series(query_name: str) -> list[tuple[float, float]]:
    if query_name.startswith("DSB"):
        df = pd.read_csv("db_eval/balsa/balsa_6h_dsb.csv")
        row = df.loc[df["query_name"] == query_name]
        return [(6 * 60 * 60, row["Balsa"].iloc[0])]

    with open(f"db_eval/balsa/balsa_runs/{query_name}.json") as f:
        data = json.load(f)
        opt_time_secs = [hours * 3600 for hours in data["time_hours"]]
        best_runtime_secs = data["best_perf_secs"]
        return list(zip(opt_time_secs, best_runtime_secs))
