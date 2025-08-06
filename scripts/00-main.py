import argparse
from climatem.model.tsdcd_latent import LatentTSDCD
import numpy as np
import torch
import wandb
from math import sqrt

from causal_graph_comparison import * # Directory paths
from causal_graph_comparison.causal_discovery import causal_discovery
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison.picabu_helpers import load_picabu_config
from causal_graph_comparison.savar import generate_savar_data
from causal_graph_comparison.trainer import run_trainer
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison.vae import train_vae
from causal_graph_comparison.mlp import MLP
from causal_graph_comparison.lstm import LSTM
from causal_graph_comparison.cnn import CNN
from causal_graph_comparison.graph_utils import binarize_array, flatten_temporal_adjacency_graph, permute_graph
from causal_graph_comparison.rollouts import run_rollouts

# ---- CMD LINE ARGS

args = argparse.ArgumentParser()
args.add_argument(
    "--difficulty",
    type=str,
    choices=["easy", "med_easy", "med_hard", "hard"],
    help="Difficulty level: easy, med_easy, med_hard, hard",
)
args.add_argument(
    "--num_modes", type=int, choices=[4, 16, 64], help="Number of modes: 4, 25, 100"
)
args.add_argument("--seed", type=int, default=1, choices=[1, 42, 99])
args.add_argument(
    "--dataset_type", type=str, default="savar", help="Dataset type: savar"
)
args.add_argument(
    "--resolution", type=int, default=10, help="Resolution per mode is 10x10"
)

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

# ====== PART 1: TRAINING ======

# ------ Generate SAVAR data ------

# load savar+picabu config file
(
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
) = load_picabu_config()

# load model params from config file
trained_model_params = get_json_config(CONFIGS_DIR / "models.json")

# set difficulty, num_modes, seed, size of input
experiment_params.d_z = args.num_modes
savar_params.difficulty = args.difficulty
savar_params.comp_size = args.resolution
savar_params.n_per_col = int(sqrt(args.num_modes))
experiment_params.lon = savar_params.comp_size * savar_params.n_per_col
experiment_params.lat = savar_params.comp_size * savar_params.n_per_col
experiment_params.random_seed = args.seed
experiment_params.d_x = experiment_params.lon * experiment_params.lat

# set variable sparsity threshold based on difficulty
N = args.num_modes
tau = experiment_params.tau

if N == 4:
    denom = 1
else:
    denom = 2

sparsity_thresholds = {
    "easy": N / (N**2 * tau), # for N = 4, tau = 5, prob = 4 / 80 = 0.05
    "med_easy": 2 * N / (N**2 * tau), # for N = 4, tau = 5, prob = 8 / 80 = 0.1
    "med_hard": 3 * N / (N**2 * tau), # for N = 4, tau = 5, prob = 12 / 80 = 0.15
    "hard": (N + N*(N-1)/denom) / (N**2 * tau) # for N = 4, tau = 5, prob = 10 / 80 = 0.125
}

optim_params.sparsity_upper_threshold = sparsity_thresholds[args.difficulty]

# set device
device = torch.device(
    "cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu"
)

# generate savar data
datamodule = generate_savar_data(
    experiment_params, data_params, savar_params, train_params, reload_data=False
)

# print datamodule 
print("datamodule.savar_gt_adj shape: ", datamodule.savar_gt_adj.shape)
print("datamodule.savar_gt_modes_weights shape: ", datamodule.savar_gt_modes_weights.shape)
print("datamodule.savar_links_coeffs (ground truth causal links): ", datamodule.savar_links_coeffs)
print("datamodule.savar_gt_modes shape: ", datamodule.savar_gt_modes.shape)
if datamodule.savar_gt_modes_weights is None:
    raise ValueError("datamodule.savar_gt_modes_weights is None, try setting reload_climate_set_data in data_params to True")

# generate train and test dataloaders
train_dataset = datamodule._data_train
test_dataset = datamodule._data_val

print("train_dataset length: ", len(train_dataset))
print("test_dataset length: ", len(test_dataset))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=data_params.batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=data_params.eval_batch_size, shuffle=False
)

# ------ Train picabu on savar for "ground truth" causal graph ------
# saves results to scratch/results/SAVAR_DATA_TEST/picabu-modes_{modes}-diff_{difficulty}-seed_{seed}/ (graph)

print("=== training picabu on savar")
wandb.init(project="climatem", config={"model": "picabu", **wandb_dict})
train_picabu(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
)
wandb.finish()

# ------ Train emulators on savar ------

# 1) Train vae --> saves scratch/cgc/models/vae-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
print("=== training picabu as vae on savar")
wandb.init(project="climatem", config={"model": "vae", **wandb_dict})
vae_path = train_vae(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
    trained_model_params=trained_model_params,
)
wandb.finish()

# 2) Train mlp --> saves scratch/cgc/models/mlp-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
mlp_input_size = experiment_params.d_x * experiment_params.tau
mlp_output_size = experiment_params.d_x * experiment_params.future_timesteps
mlp_layers = trained_model_params["mlp"]["model_params"]["num_layers"]

mlp = MLP(
    input_size=mlp_input_size, output_size=mlp_output_size, num_layers=mlp_layers
).to(device)
print("=== training mlp on savar")
wandb.init(project="climatem", config={"model": "mlp", **wandb_dict})
mlp_path = run_trainer(
    datamodule,
    mlp,
    trained_model_params,
    args.dataset_type,
    args.num_modes,
    args.difficulty,
    args.seed,
    device,
)
wandb.finish()

# 3) Train lstm --> saves scratch/cgc/models/lstm-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
input_size = experiment_params.d_x
num_layers = trained_model_params["lstm"]["model_params"]["num_layers"]

lstm = LSTM(
    input_size=input_size, hidden_size=int(input_size / 2), num_layers=num_layers
).to(device)
print("=== training lstm on savar")
wandb.init(project="climatem", config={"model": "lstm", **wandb_dict})
lstm_path = run_trainer(
    datamodule,
    lstm,
    trained_model_params,
    args.dataset_type,
    args.num_modes,
    args.difficulty,
    args.seed,
    device,
)
wandb.finish()

# 4) Train cnn --> saves scratch/cgc/models/cnn-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
cnn_input_channels = experiment_params.tau
cnn_output_channels = experiment_params.future_timesteps
cnn_image_size = experiment_params.lon

cnn = CNN(
    input_channels=cnn_input_channels,
    output_channels=cnn_output_channels,
    image_size=cnn_image_size,
).to(device)
print("=== training cnn on savar")
wandb.init(project="climatem", config={"model": "cnn", **wandb_dict})
cnn_path = run_trainer(
    datamodule,
    cnn,
    trained_model_params,
    args.dataset_type,
    args.num_modes,
    args.difficulty,
    args.seed,
    device,
)
wandb.finish()

# ------ Run causal representation learning (picabu) on emulators to learn their causal graphs ------

# 1) Train picabu on mlp --> saves scratch/results/SAVAR_DATA_TEST/picabu-mlp-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)

mlp_model = torch.load(mlp_path, map_location=device, weights_only=False)

wandb.init(project="climatem", config={"model": "picabu-mlp", **wandb_dict})
train_picabu(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
    trained_model=mlp_model,
    trained_model_params=trained_model_params,
)
wandb.finish()

# 2) Train picabu on lstm --> saves scratch/results/SAVAR_DATA_TEST/picabu-lstm-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
lstm_model = torch.load(lstm_path, map_location=device, weights_only=False)

wandb.init(project="climatem", config={"model": "picabu-lstm", **wandb_dict})
train_picabu(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
    trained_model=lstm_model,
    trained_model_params=trained_model_params,
)
wandb.finish()

# 3) Train picabu on cnn --> saves scratch/results/SAVAR_DATA_TEST/picabu-cnn-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
cnn_model = torch.load(cnn_path, map_location=device, weights_only=False)

wandb.init(project="climatem", config={"model": "picabu-cnn", **wandb_dict})
train_picabu(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
    trained_model=cnn_model,
    trained_model_params=trained_model_params,
)
wandb.finish()

# 4) Train picabu on vae --> saves scratch/results/SAVAR_DATA_TEST/picabu-vae-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
vae_state_dict = torch.load(vae_path, map_location=device)

# Since vae is just picabu with constraints turned off, we use picabu training pipeline
# which saves model as a state_dict, so we need to recreate the model

vae_model = LatentTSDCD(
    num_layers=picabu_params.num_layers,
    num_hidden=picabu_params.num_hidden,
    num_input=experiment_params.tau,
    num_output=experiment_params.future_timesteps, 
    num_layers_mixing=picabu_params.num_layers_mixing,
    num_hidden_mixing=picabu_params.num_hidden_mixing,
    position_embedding_dim=picabu_params.position_embedding_dim,
    reduce_encoding_pos_dim=picabu_params.reduce_encoding_pos_dim,
    coeff_kl=optim_params.coeff_kl,
    d=1, # number of datasets
    # Here, everything hardcoded to gaussian because GEV leads to Nan... TBD
    distr_z0="gaussian",
    distr_encoder="gaussian",
    distr_transition="gaussian",
    distr_decoder="gaussian",
    d_x=experiment_params.d_x,
    d_z=experiment_params.d_z,
    tau=experiment_params.tau,
    instantaneous=picabu_params.instantaneous,
    nonlinear_dynamics=picabu_params.nonlinear_dynamics,
    nonlinear_mixing=picabu_params.nonlinear_mixing,
    hard_gumbel=picabu_params.hard_gumbel,
    no_gt=gt_params.no_gt,
    debug_gt_graph=gt_params.debug_gt_graph,
    debug_gt_z=gt_params.debug_gt_z,
    debug_gt_w=gt_params.debug_gt_w,
    tied_w=picabu_params.tied_w,
    fixed=picabu_params.fixed,
    fixed_output_fraction=picabu_params.fixed_output_fraction,
    vae_mode=True,
)

vae_model.load_state_dict({k.replace("module.", ""): v for k, v in vae_state_dict.items()})
vae_model.to(device)

wandb.init(project="climatem", config={"model": "picabu-vae", **wandb_dict})
train_picabu(
    datamodule,
    experiment_params,
    data_params,
    gt_params,
    train_params,
    picabu_params,
    optim_params,
    plot_params,
    savar_params,
    trained_model=vae_model,
    trained_model_params=trained_model_params,
)
wandb.finish()

# ------ Run causal discovery on emulators to learn their causal graphs ------

# load inference params from config file
n_samples = trained_model_params["common"]["test_params"]["num_samples"]
rollouts = trained_model_params["common"]["test_params"]["rollout_timesteps"]
inference_batch_size = trained_model_params["common"]["test_params"]["inference_batch_size"]

# create inference loader from test dataset with batch size of 1000 (i.e. pass all samples at once)
inference_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=inference_batch_size, shuffle=False
)

# 1) run causal discovery on mlp --> saves scratch/cgc/outputs/mlp-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz

## 1.1) run inference - create 1000 samples 20 timesteps
print(f"Running inference on mlp: {n_samples} samples, {rollouts} timesteps...")

mlp_rollouts_path = run_rollouts(mlp_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

causal_discovery_params = {
    "num_modes": args.num_modes,
    "links_coeffs": datamodule.savar_links_coeffs,
    "tau_max": experiment_params.tau,
    "difficulty": args.difficulty,
    "seed": args.seed,
}

## 1.2) run causal discovery
mlp_graph, mlp_val_matrix, mlp_p_matrix, mlp_corr_matrix, mlp_var_names = causal_discovery(timeseries=mlp_rollouts_path, model_name=mlp_model.name, subsample="mean", **causal_discovery_params)

# 2) Run causal discovery on lstm 
## 2.1) run inference --> saves scratch/cgc/outputs/lstm-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
print(f"Running inference on lstm: {n_samples} samples, {rollouts} timesteps...")
lstm_rollouts_path = run_rollouts(lstm_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

## 2.2) run causal discovery
lstm_graph, lstm_val_matrix, lstm_p_matrix, lstm_corr_matrix, lstm_var_names = causal_discovery(timeseries=lstm_rollouts_path, model_name=lstm_model.name, subsample="mean", **causal_discovery_params)

# 3) Run causal discovery on cnn 
## 3.1) run inference --> saves scratch/cgc/outputs/cnn-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
print(f"Running inference on cnn: {n_samples} samples, {rollouts} timesteps...")
cnn_rollouts_path = run_rollouts(cnn_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

## 3.2) run causal discovery
cnn_graph, cnn_val_matrix, cnn_p_matrix, cnn_corr_matrix, cnn_var_names = causal_discovery(timeseries=cnn_rollouts_path, model_name=cnn_model.name, subsample="mean", **causal_discovery_params)

# 4) Run causal discovery on vae 
## 4.1) run inference --> saves scratch/cgc/outputs/vae-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
print(f"Running inference on vae: {n_samples} samples, {rollouts} timesteps...")
vae_rollouts_path = run_rollouts(vae_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

## 4.2) run causal discovery
vae_graph, vae_val_matrix, vae_p_matrix, vae_corr_matrix, vae_var_names = causal_discovery(timeseries=vae_rollouts_path, model_name=vae_model.name, subsample="mean", **causal_discovery_params)

# 5) Run causal discovery on savar ground truth
## 5.1) get targets saved from rollout (1000 samples, 20 timesteps)
rollout_targets = np.load(mlp_rollouts_path)['targets']
## 5.2) run causal discovery
savar_graph, savar_val_matrix, savar_p_matrix, savar_corr_matrix, savar_var_names = causal_discovery(timeseries=rollout_targets, model_name="savar", subsample="mean", **causal_discovery_params)

# ===== Run rollouts with 1 step for each model <-- for rmse, statistical metrics
rollouts = 1
# 1) mlp
mlp_rollouts_path = run_rollouts(mlp_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

# 2) lstm
lstm_rollouts_path = run_rollouts(lstm_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

# 3) cnn
cnn_rollouts_path = run_rollouts(cnn_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

# 4) vae
vae_rollouts_path = run_rollouts(vae_model, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)



# --- PART 2: EVALUATION

# 1) Permute learned adjacency graph outputs using ground truth so modes are in same order as in ground truth

picabu_permuted_crl_graph = permute_graph(datamodule, experiment_params, savar_params, "picabu")
mlp_permuted_crl_graph = permute_graph(datamodule, experiment_params, savar_params, "picabu_mlp")
lstm_permuted_crl_graph = permute_graph(datamodule, experiment_params, savar_params, "picabu_lstm")
cnn_permuted_crl_graph = permute_graph(datamodule, experiment_params, savar_params, "picabu_cnn")
vae_permuted_crl_graph = permute_graph(datamodule, experiment_params, savar_params, "picabu_vae")

# 2a) Flatten temporal adjacency graphs: causal representation learning
gt_flat_crl_graph = flatten_temporal_adjacency_graph(shape="time_child_parent", causal_method="crl", graph=picabu_permuted_crl_graph)
mlp_flat_crl_graph = flatten_temporal_adjacency_graph(shape="time_child_parent", causal_method="crl", graph=mlp_permuted_crl_graph)
lstm_flat_crl_graph = flatten_temporal_adjacency_graph(shape="time_child_parent", causal_method="crl", graph=lstm_permuted_crl_graph)
cnn_flat_crl_graph = flatten_temporal_adjacency_graph(shape="time_child_parent", causal_method="crl", graph=cnn_permuted_crl_graph)
vae_flat_crl_graph = flatten_temporal_adjacency_graph(shape="time_child_parent", causal_method="crl", graph=vae_permuted_crl_graph)

# 2b) Flatten temporal adjacency graphs: causal discovery
gt_flat_cd_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", graph=savar_val_matrix)
mlp_flat_cd_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", graph=mlp_val_matrix)
lstm_flat_cd_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", graph=lstm_val_matrix)
cnn_flat_cd_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", graph=cnn_val_matrix)
vae_flat_cd_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", graph=vae_val_matrix)

# Binarize causal discovery graphs
gt_flat_cd_graph = binarize_array(gt_flat_cd_graph)
mlp_flat_cd_graph = binarize_array(mlp_flat_cd_graph)
lstm_flat_cd_graph = binarize_array(lstm_flat_cd_graph)
cnn_flat_cd_graph = binarize_array(cnn_flat_cd_graph)
vae_flat_cd_graph = binarize_array(vae_flat_cd_graph)

# 3) Apply causal & structural comparison metrics to mlp (picabu, causal_discovery)
# TODO: add scripts/11-graph-eval_mlp.py to pipeline
# TODO: implement Distance Average Causal Effect: https://www.nature.com/articles/s41467-024-50813-z

# 4) Apply causal & structural comparison metrics to lstm (picabu, causal_discovery)

# 5) Apply causal & structural comparison metrics to cnn (picabu, causal_discovery)

# 6) Apply causal & structural comparison metrics to vae (picabu, causal_discovery)

# 7) Run RMSE, statistical metrics on all trained models

# 8) Apply power spectral density script from climatem module to: inference from models, targets 

# 9) Plotting...
    # explore-mlp-output.ipynb
    # causal-discovery-mlp.ipynb
    # causal-discovery-groundtruth.ipynb
    # causal-discovery-mlp-1000samples.ipynb

# --- PART 3: INTERVENTIONS

# 1) Perturb data

# 2) Run picabu on all trained models with perturbed data

# 3) Run inference on all trained models w/ perturbed data

# 4) Get "gt" target from savar model for perturbed data

# 5) Apply RMSE on inference results

# --------- PART 0: GRAPH EVALUATION ---------

# 1) Create permuted dataset of graphs
# TODO: use 01-dataset-builder.py to create permuted dataset of graphs

# 2) Apply comparison metrics 
# TODO: use graph-eval.py 

# 3) Do qualitative analysis on graphs


# --------- BASH

# 12 -16 cpus for 1 gpu a100
# 120 - 256 gb ram
# crank up num workers <-- 
