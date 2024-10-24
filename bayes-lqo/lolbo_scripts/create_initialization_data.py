import sys

sys.path.append("../")
import argparse

import numpy as np
import pandas as pd
import torch
from utils.set_seed import set_seed
from vae.model import VAEModule
from your_tasks.your_objective_functions import OBJECTIVE_FUNCTIONS_DICT


def create_init_data_aliases(
    workload_name,
    timeout=50,
    which_language="aliases",
    N=200,
    vae_latent_space_dim=64,
    path_to_vae="../vae/64.ckpt",
):
    # always use seed 0
    set_seed(seed=0)
    oracle = OBJECTIVE_FUNCTIONS_DICT["db"](
        workload_name=workload_name,
        worst_runtime_observed=timeout * 2,  # worst runtime from init data, used to set score for OOM query plans
        timeout=timeout,
        which_language=which_language,
    )
    vae = VAEModule.load_from_checkpoint(path_to_vae)
    vae.cuda()
    samples = vae.sample(torch.randn(N, vae_latent_space_dim).cuda())
    scores_list, censoring_list = oracle.query_black_box(samples)
    xs = [str(samp)[1:-1].replace(" ", "") for samp in samples]
    df = {}
    df["x"] = np.array(xs)
    df["y"] = np.array(scores_list)
    df["censoring"] = np.array(censoring_list)
    df = pd.DataFrame.from_dict(df)
    df.to_csv(f"../initialization_data/{workload_name}_init_data.csv", index=None)


def test_load_init_data(
    workload_name="JOB_16F",
    num_initialization_points=200,
):
    # TODO: fix...
    init_data_path = f"../initialization_data/{workload_name}_init_data.csv"
    df = pd.read_csv(init_data_path)
    x = df["x"].values.tolist()
    x = x[0:num_initialization_points]

    # Preprocessing for databases stuff:
    x = [xi.split(",") for xi in x]
    x = [[int(xj) for xj in xi] for xi in x]

    y = torch.from_numpy(df["y"].values).float()
    y = y[0:num_initialization_points]
    y = y.unsqueeze(-1)

    cen = torch.from_numpy(df["censoring"].values)
    cen = cen[0:num_initialization_points]
    cen = cen.unsqueeze(-1)
    print(len(x), y.shape, cen.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload_name",
        help=" Name of workload to run.",
        type=str,
        default="JOB_6F",
        required=False,
    )
    parser.add_argument(
        "--timeout",
        help=" Timeout to use when generating init data (s).",
        type=int,
        default=200,
        required=False,
    )
    parser.add_argument(
        "--which_language",
        help=" Which langauge to use for queries.",
        type=str,
        default="aliases",
        required=False,
    )
    parser.add_argument(
        "--N",
        help=" N init data points to generate.",
        type=int,
        default=200,
        required=False,
    )
    parser.add_argument(
        "--vae_latent_space_dim",
        help=" Dim of VAE latent space.",
        type=int,
        default=64,
        required=False,
    )
    parser.add_argument(
        "--path_to_vae",
        help=" Path to state dict for VAE to use.",
        type=str,
        default="../vae/64.ckpt",
        required=False,
    )
    args = parser.parse_args()
    create_init_data_aliases(
        workload_name=args.workload_name,
        timeout=args.timeout,
        which_language=args.which_language,
        N=args.N,
        vae_latent_space_dim=args.vae_latent_space_dim,
        path_to_vae=args.path_to_vae,
    )
