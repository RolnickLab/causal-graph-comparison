import argparse
import os
import shutil
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
import numpy as np
import torch
import wandb
from math import sqrt

from causal_graph_comparison import CONFIGS_DIR
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison.picabu_helpers import load_picabu_config
from causal_graph_comparison.savar import generate_savar_data
from causal_graph_comparison.trainer import run_trainer
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison.vae import train_vae
from causal_graph_comparison.mlp import MLP
from causal_graph_comparison.lstm import LSTM
from causal_graph_comparison.cnn import CNN

# ---- CMD LINE ARGS

# input: difficulty, me/m-e/m-h/h
# input: dimensions (inferred from number of modes)
# input: number of modes, 4, 25, 100
# input: seeds


args = argparse.ArgumentParser()
args.add_argument("--difficulty", type=str, choices=["e", "me", "mh", "h"], help="Difficulty level: easy (e), medium-easy (me), medium-hard (mh), hard (h)")
args.add_argument("--num_modes", type=int, choices=[4, 25, 100], help="Number of modes: 4, 25, 100")
args.add_argument("--seed", type=int, default=42)
args.add_argument("--dataset_type", type=str, default="savar", help="Dataset type: savar")
args.add_argument("--resolution", type=int, default=10, help="Resolution: 10")

args = args.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

wandb_dict = {
    "difficulty": args.difficulty,
    "num_modes": args.num_modes,
    "seed": args.seed,
    "dataset_type": args.dataset_type,
    "resolution": args.resolution,
}

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
experiment_params, data_params, gt_params, train_params, picabu_params, optim_params, plot_params, savar_params = load_picabu_config()

# set difficulty, num_modes, seed, size of input
experiment_params.difficulty = args.difficulty
experiment_params.d_z = args.num_modes
savar_params.comp_size = args.resolution
savar_params.n_per_col = int(sqrt(args.num_modes))
experiment_params.lon = savar_params.comp_size * savar_params.n_per_col
experiment_params.lat = savar_params.comp_size * savar_params.n_per_col
experiment_params.seed = args.seed
experiment_params.d_x = experiment_params.lon * experiment_params.lat

device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

datamodule = generate_savar_data(experiment_params, data_params, savar_params, train_params)

train_dataset = datamodule._data_train
test_dataset = datamodule._data_val

print("train_dataset length: ", len(train_dataset))
print("test_dataset length: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_params.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_params.eval_batch_size, shuffle=False)

# STEP train picabu --> saves picabu.pth (learn "gt" causal graph)
# TODO: train picabu only if it doesn't already exist

print ("=== training picabu on savar")
wandb.init(project="climatem", config={"model": "picabu", **wandb_dict})
train_picabu(datamodule, experiment_params, data_params, gt_params, train_params, picabu_params, optim_params, plot_params, savar_params)
wandb.finish()

# load model params 
trained_model_params = get_json_config(CONFIGS_DIR / "models.json")

# STEP train vae --> saves vae.pth
print ("=== training picabu as vae on savar")
wandb.init(project="climatem", config={"model": "vae", **wandb_dict})
train_vae(datamodule, experiment_params, data_params, gt_params, train_params, picabu_params, optim_params, plot_params, savar_params, trained_model_params=trained_model_params)
wandb.finish()

# STEP train mlp --> saves mlp.pth
mlp_input_size = experiment_params.d_x * experiment_params.tau
mlp_output_size = experiment_params.d_x * experiment_params.future_timesteps
mlp_layers = trained_model_params["mlp"]["model_params"]["layers"]


mlp = MLP(input_size=mlp_input_size, output_size=mlp_output_size, layers=mlp_layers).to(device)
print ("=== training mlp on savar")
wandb.init(project="climatem", config={"model": "mlp", **wandb_dict})
run_trainer(datamodule, mlp, trained_model_params, args.dataset_type, args.num_modes, args.difficulty, args.seed, device)
wandb.finish()

# STEP train lstm --> saves lstm.pth
input_size = experiment_params.d_x
num_layers = trained_model_params["lstm"]["model_params"]["num_layers"]

lstm = LSTM(input_size=input_size, hidden_size=int(input_size/2), num_layers=num_layers).to(device)
print ("=== training lstm on savar")
wandb.init(project="climatem", config={"model": "lstm", **wandb_dict})
run_trainer(datamodule, lstm, trained_model_params, args.dataset_type, args.num_modes, args.difficulty, args.seed, device)
wandb.finish()

# STEP train cnn --> saves cnn.pth
cnn_input_channels = experiment_params.tau
cnn_output_channels = experiment_params.future_timesteps
cnn_image_size = experiment_params.lon

cnn = CNN(input_channels=cnn_input_channels, output_channels=cnn_output_channels, image_size=cnn_image_size).to(device)
print ("=== training cnn on savar")
wandb.init(project="climatem", config={"model": "cnn", **wandb_dict})
run_trainer(datamodule, cnn, trained_model_params, args.dataset_type, args.num_modes, args.difficulty, args.seed, device)
wandb.finish()
# Load evaluated model to pass to picabu

# Load evaluated_model to learn causal graph on mlp, lstm, etc.
# num_modes = experiment_params.d_z
# dataset_type = data_params.in_var_ids[0]
# difficulty = savar_params.difficulty
# seed = experiment_params.random_seed

# if trained_model is not None:
#     save_name = f"{trained_model.name}-{dataset_type}-modes_{num_modes}-diff_{diff_mapping[difficulty]}-seed_{seed}"
#     EVALUATED_MODEL = torch.load(
#         f"{SCRATCH_DIR}/cgc/models/{trained_model.name}.pt",
#         map_location=device,
#         weights_only=False,
#     )


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
