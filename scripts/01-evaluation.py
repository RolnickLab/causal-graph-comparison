import argparse
import torch
import numpy as np

from gadjid import  ancestor_aid, oset_aid, parent_aid, shd, sid

from causal_graph_comparison import *
from causal_graph_comparison.psd import power_spectral_density

# ---- CMD LINE ARGS

args = argparse.ArgumentParser()
args.add_argument(
    "--difficulty",
    type=str,
    choices=["easy", "med_easy", "med_hard", "hard"],
    help="Difficulty level: easy, med_easy, med_hard, hard",
)
args.add_argument("--num_modes", type=int, choices=[4, 16, 64], help="Number of modes: 4, 25, 100")
args.add_argument("--seed", type=int, default=1, choices=[1, 42, 99])
args.add_argument("--dataset_type", type=str, default="savar", help="Dataset type: savar")
args.add_argument("--resolution", type=int, default=10, help="Resolution per mode is 10x10")
args.add_argument(
    "--model",
    type=str,
    default="picabu",
    help="Model type: picabu, vae, mlp, lstm, cnn",
    choices=["picabu", "vae", "mlp", "lstm", "cnn"],
)

args = args.parse_args()

print(f"======= Training model: {args.model} on SAVAR data with difficulty: {args.difficulty}, num_modes: {args.num_modes}, seed: {args.seed} =======")



torch.manual_seed(args.seed)
np.random.seed(args.seed)


experiment_name = f"{args.model}-modes_{args.num_modes}-diff_{args.difficulty}-seed_{args.seed}"
gt_name = f"savar-modes_{args.num_modes}-diff_{args.difficulty}-seed_{args.seed}"

crl_graph_path = OUTPUTS_DIR / f"{experiment_name}-flat_graph_binary-crl.npz"
cd_graph_path = OUTPUTS_DIR / f"{experiment_name}-flat_graph_binary-cd.npz"

gt_crl_graph_path = OUTPUTS_DIR / f"{gt_name}-flat_graph_binary-crl.npz"
gt_cd_graph_path = OUTPUTS_DIR / f"{gt_name}-flat_graph_binary-cd.npz"

next_step_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_1steps.npz"
next_step_data = np.load(next_step_path)
print("next_step_path", next_step_data["targets"].shape)


# 1) Load data

print("Loading data...")

crl_graph = np.load(crl_graph_path)["array"]
cd_graph = np.load(cd_graph_path)["array"]
gt_crl_graph = np.load(gt_crl_graph_path)["array"]
gt_cd_graph = np.load(gt_cd_graph_path)["array"]

print("crl_graph.shape", crl_graph.shape)
print("cd_graph.shape", cd_graph.shape)
print("gt_crl_graph.shape", gt_crl_graph.shape)
print("gt_cd_graph.shape", gt_cd_graph.shape)


# 3) Apply causal & structural comparison metrics to mlp (picabu, causal_discovery)
# TODO: add scripts/11-graph-eval_mlp.py to pipeline
# TODO: implement Distance Average Causal Effect: https://www.nature.com/articles/s41467-024-50813-z

# crl
# -- 4. Calculate metrics --

print("Calculating causal & structural metrics...")

crl_parent_aid = parent_aid(gt_crl_graph, crl_graph, edge_direction="from row to column")
crl_sid = sid(gt_crl_graph, crl_graph, edge_direction="from row to column")
crl_shd = shd(gt_crl_graph, crl_graph)

print("crl_parent_aid", crl_parent_aid)
print("crl_sid", crl_sid)
print("crl_shd", crl_shd)
# TODO: implement f1 score

# cd
cd_parent_aid = parent_aid(gt_cd_graph, cd_graph, edge_direction="from row to column")
cd_sid = sid(gt_cd_graph, cd_graph, edge_direction="from row to column")
cd_shd = shd(gt_cd_graph, cd_graph)

print("cd_parent_aid", cd_parent_aid)
print("cd_sid", cd_sid)
print("cd_shd", cd_shd)

# 7) Run RMSE, statistical metrics on all trained models

print("Running statistical metrics on next step...")

next_step_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_1steps.npz"
next_step = np.load(next_step_path)["outputs"]
print("next_step", next_step.shape)

next_step_targets = np.load(next_step_path)["targets"]
print("next_step_targets", next_step_targets.shape)

# RMSE
next_step_rmse = np.sqrt(np.mean((next_step - next_step_targets) ** 2))
print("next_step_rmse", next_step_rmse)

# MSE
next_step_mse = np.mean((next_step - next_step_targets) ** 2)
print("next_step_mse", next_step_mse)

# MAE
next_step_mae = np.mean(np.abs(next_step - next_step_targets))
print("next_step_mae", next_step_mae)

# R2
next_step_r2 = 1 - (np.sum((next_step - next_step_targets) ** 2) / np.sum((next_step_targets - np.mean(next_step_targets)) ** 2))
print("next_step_r2", next_step_r2)

# Variance
next_step_variance = np.var(next_step)
print("next_step_variance", next_step_variance)

next_step_targets_variance = np.var(next_step_targets)
print("next_step_targets_variance", next_step_targets_variance)

# Bias
next_step_bias = np.mean(np.mean(next_step - next_step_targets) ** 2) - next_step_targets_variance
print("next_step_bias", next_step_bias)

# stdev
next_step_stdev = np.std(next_step)
print("next_step_stdev", next_step_stdev)

next_step_targets_stdev = np.std(next_step_targets)
print("next_step_targets_stdev", next_step_targets_stdev)

# range
next_step_range = np.max(next_step) - np.min(next_step)
print("next_step_range", next_step_range)

next_step_targets_range = np.max(next_step_targets) - np.min(next_step_targets)
print("next_step_targets_range", next_step_targets_range)



# 8) Apply power spectral density script from climatem module to: inference from models, targets 

print("Running power spectral density on rollouts...")

rollouts_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_20steps.npz"
LSD, fft_coeffs_rollouts, fft_coeffs_savar, fft_coeffs_savar_all = power_spectral_density(rollouts_path, args.num_modes)
print("LSD", LSD)

# TODO: use coeffs for plotting
# TODO: add metrics from climatem graph eval metrics
## graph_evaluation_ilija.py, climatem/model/metrics.py

metrics = {
        "shd": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "train_mse": 0.0,
        "val_mse": 0.0,
        "mcc": 0.0,
    }