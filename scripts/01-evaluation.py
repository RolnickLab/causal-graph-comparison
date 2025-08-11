import argparse
import torch
import numpy as np

from gadjid import  ancestor_aid, oset_aid, parent_aid, shd, sid

from causal_graph_comparison import *
from causal_graph_comparison.psd import power_spectral_density
from climatem.model.metrics import *

# ---- CMD LINE ARGS

model = "mlp"
difficulty = "easy"
num_modes = 4
seed = 1
dataset = "savar"

print(f"======= Training model: {model} on SAVAR data with difficulty: {difficulty}, num_modes: {num_modes}, seed: {seed} =======")



torch.manual_seed(seed)
np.random.seed(seed)


experiment_name = f"{model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}"
gt_name = f"{dataset}-modes_{num_modes}-diff_{difficulty}-seed_{seed}"

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
cd_graph = np.load(cd_graph_path)["graph"]
gt_crl_graph = np.load(gt_crl_graph_path)["array"]
gt_cd_graph = np.load(gt_cd_graph_path)["graph"]

print("crl_graph.shape", crl_graph.shape)
print("cd_graph.shape", cd_graph.shape)
print("gt_crl_graph.shape", gt_crl_graph.shape)
print("gt_cd_graph.shape", gt_cd_graph.shape)


# 3) Apply causal & structural comparison metrics to mlp (picabu, causal_discovery)
# TODO: add scripts/11-graph-eval_mlp.py to pipeline
# TODO: implement Distance Average Causal Effect: https://www.nature.com/articles/s41467-024-50813-z
print("==============================")

# -- 4. Calculate metrics --

print("Structural & causal metrics on CRL...")

crl_parent_aid = parent_aid(gt_crl_graph, crl_graph, edge_direction="from row to column")
crl_sid = sid(gt_crl_graph, crl_graph, edge_direction="from row to column")
crl_shd = shd(gt_crl_graph, crl_graph)
crl_f1 = f1_score(crl_graph, gt_crl_graph)
crl_precision, crl_recall = precision_recall(crl_graph, gt_crl_graph)

print("crl_parent_aid: ", crl_parent_aid)
print("crl_sid: ", crl_sid)
print("crl_shd: ", crl_shd)
print("crl_f1: ", crl_f1)
print("crl_precision: ", crl_precision)
print("crl_recall: ", crl_recall)
print("-----------------------------")

print("Structural & causal metrics on CD...")

# cd
cd_parent_aid = parent_aid(gt_cd_graph, cd_graph, edge_direction="from row to column")
cd_sid = sid(gt_cd_graph, cd_graph, edge_direction="from row to column")
cd_shd = shd(gt_cd_graph, cd_graph)
cd_f1 = f1_score(cd_graph, gt_cd_graph)
cd_precision, cd_recall = precision_recall(cd_graph, gt_cd_graph)


print("cd_parent_aid", cd_parent_aid)
print("cd_sid", cd_sid)
print("cd_shd", cd_shd)
print("cd_f1", cd_f1)
print("cd_precision", cd_precision)
print("cd_recall", cd_recall)

print("==============================")

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

# proportion of False positive / False negative


print("==============================")


# 8) Apply power spectral density script from climatem module to: inference from models, targets 

print("Running power spectral density on rollouts...")

rollouts_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_20steps.npz"
LSD, fft_coeffs_rollouts, fft_coeffs_savar = power_spectral_density(rollouts_path, num_modes)

# TODO: use coeffs for plotting