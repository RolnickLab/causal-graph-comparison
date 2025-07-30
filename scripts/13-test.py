import argparse
import os
import shutil
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
import numpy as np
import torch
import wandb

from causal_graph_comparison import CONFIGS_DIR
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison.picabu_helpers import load_picabu_config
from causal_graph_comparison.savar import generate_savar_data

# ---- CMD LINE ARGS

# input: difficulty, me/m-e/m-h/h
# input: dimensions (inferred from number of modes)
# input: number of modes, 4, 25, 100
# input: seeds


args = argparse.ArgumentParser()
args.add_argument("--difficulty", type=str, choices=["e", "me", "mh", "h"], help="Difficulty level: easy (e), medium-easy (me), medium-hard (mh), hard (h)")
args.add_argument("--num_modes", type=int, choices=[4, 25, 100], help="Number of modes: 4, 25, 100")
args.add_argument("--seed", type=int, default=42)

args = args.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

wandb.init(project="climatem")

# ---- DESIDERATA

# - log everything to wandb (including what steps we're on)
# - IDK, maybe copy config file somewhere? or just log the config to wandb which imho is better
# - IDK, maybe some checks in the beginning that all paths are good?
# - make sure input data to model is same size for all models / all models accept same input size
# - apply one more causal discovery method
# - use configs
# - train only on time window of 5, not 30

# --- PART 1: TRAINING

# Generate SAVAR data

# load savar+picabu config file
experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params = load_picabu_config()

device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

datamodule = generate_savar_data(experiment_params, data_params, savar_params, train_params)

train_dataset = datamodule._data_train
test_dataset = datamodule._data_val

print("train_dataset length: ", len(train_dataset))
print("test_dataset length: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_params.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_params.eval_batch_size, shuffle=False)

# STEP train picabu --> saves picabu.pth (learn "gt" causal graph)
train_picabu(experiment_params, data_params, gt_params, train_params, model_params, optim_params, plot_params, savar_params, datamodule)

# STEP train vae --> saves vae.pth
### get config from savar folder

# STEP train mlp --> saves mlp.pth
### get config from savar folder

# STEP train lstm --> saves lstm.pth
### get config from savar folder

# STEP train cnn --> saves cnn.pth
### get config from savar folder

# Load evaluated model to pass to picabu

# Load evaluated_model to learn causal graph on mlp, lstm, etc.
num_modes = experiment_params.d_z
dataset_type = data_params.in_var_ids[0]
difficulty = savar_params.difficulty
seed = experiment_params.random_seed

if trained_model is not None:
    save_name = f"{trained_model.name}-{dataset_type}-modes_{num_modes}-diff_{diff_mapping[difficulty]}-seed_{seed}"
    EVALUATED_MODEL = torch.load(
        f"{SCRATCH_DIR}/cgc/models/{trained_model.name}.pt",
        map_location=device,
        weights_only=False,
    )


# STEP train picabu on mlp --> saves picabu_mlp.npy (graph)
### get config from mlp folder

# STEP train picabu on lstm --> saves picabu_lstm.npy (graph)
### get config from lstm folder

# STEP train picabu on cnn --> saves picabu_cnn.npy (graph)
### get config from cnn folder

# STEP train picabu on vae --> saves picabu_vae.npy (graph)
### get config from vae folder

# STEP run causal discovery on mlp --> saves causal_discovery_mlp.npy (graph)

# STEP run causal discovery on lstm --> saves causal_discovery_lstm.npy (graph)

# STEP run causal discovery on cnn --> saves causal_discovery_cnn.npy (graph)

# STEP run causal discovery on vae --> saves causal_discovery_vae.npy (graph)

# --- PART 2: EVALUATION

# STEP apply causal & structural comparison metrics to mlp (picabu, causal_discovery)

# STEP apply causal & structural comparison metrics to lstm (picabu, causal_discovery)

# STEP apply causal & structural comparison metrics to cnn (picabu, causal_discovery)

# STEP apply causal & structural comparison metrics to vae (picabu, causal_discovery)

# STEP run RMSE, statistical metrics & psd on all trained models

# --- PART 3: INTERVENTIONS

# STEP perturb data

# STEP run picabu on all trained models with perturbed data

# STEP run inference on all trained models w/ perturbed data

# STEP get "gt" target from savar model for perturbed data

# STEP apply RMSE on inference results

# --------- PART 0: GRAPH EVALUATION ---------

# create permuted dataset of graphs

# apply comparison metrics & do qualitative analysis on graphs
