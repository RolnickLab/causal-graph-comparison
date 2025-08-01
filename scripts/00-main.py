import argparse
from climatem.model.tsdcd_latent import LatentTSDCD
import numpy as np
import torch
import wandb
from math import sqrt

from causal_graph_comparison import * # Directory paths
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison.picabu_helpers import load_picabu_config
from causal_graph_comparison.savar import generate_savar_data
from causal_graph_comparison.trainer import run_trainer
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison.vae import train_vae
from causal_graph_comparison.mlp import MLP
from causal_graph_comparison.lstm import LSTM
from causal_graph_comparison.cnn import CNN
from causal_graph_comparison.graph_permutation import permute_graph
from causal_graph_comparison.rollouts import run_rollouts

# ---- CMD LINE ARGS

args = argparse.ArgumentParser()
args.add_argument(
    "--difficulty",
    type=str,
    choices=["easy", "med_easy", "med_hard", "hard"],
    help="Difficulty level: easy (e), medium-easy (me), medium-hard (mh), hard (h)",
)
args.add_argument(
    "--num_modes", type=int, choices=[4, 25, 100], help="Number of modes: 4, 25, 100"
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

# set device
device = torch.device(
    "cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu"
)

# generate savar data
datamodule = generate_savar_data(
    experiment_params, data_params, savar_params, train_params, reload_data=True
)

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

# 1) run causal discovery on mlp --> saves scratch/cgc/outputs/mlp-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz

## 1.1) run inference - create 1000 samples 20 timesteps
print(f"Running inference on mlp: {n_samples} samples, {rollouts} timesteps...")

# create inference loader from test dataset with batch size of 1000 (i.e. pass all samples at once)
inference_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=inference_batch_size, shuffle=False
)
run_rollouts(mlp, device, inference_loader, n_samples, rollouts, args.num_modes, args.difficulty, args.seed)

## 1.2) subsample outputs
print("Spatial subsampling on mlp outputs...")
# TODO: add scripts/10-subsample-outputs.py to pipeline

## 1.3) run causal discovery
# TODO: add scripts/09-causal-discovery-1000samples.py to pipeline

# 2) Run causal discovery on lstm 
## 2.1) run inference --> saves scratch/cgc/outputs/lstm-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
## 2.2) subsample outputs
## 2.3) run causal discovery

# 3) Run causal discovery on cnn 
## 3.1) run inference --> saves scratch/cgc/outputs/cnn-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
## 3.2) subsample outputs
## 3.3) run causal discovery

# 4) Run causal discovery on vae 
## 4.1) run inference --> saves scratch/cgc/outputs/vae-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
## 4.2) subsample outputs
## 4.3) run causal discovery

# 5) Run causal discovery on picabu trained on savar (ground truth) 
## 5.1) create dataset from targets (1000 samples, 20 timesteps)
## 5.2) subsample outputs
## 5.3) run causal discovery

# --- PART 2: EVALUATION

# 1) Permute learned graph outputs using ground truth so modes are in same order as in ground truth

permuted_crl_graph_picabu = permute_graph(datamodule, experiment_params, savar_params, "picabu")
permuted_crl_graph_mlp = permute_graph(datamodule, experiment_params, savar_params, "picabu_mlp")
permuted_crl_graph_lstm = permute_graph(datamodule, experiment_params, savar_params, "picabu_lstm")
permuted_crl_graph_cnn = permute_graph(datamodule, experiment_params, savar_params, "picabu_cnn")
permuted_crl_graph_vae = permute_graph(datamodule, experiment_params, savar_params, "picabu_vae")

# 2) Apply causal & structural comparison metrics to mlp (picabu, causal_discovery)
# TODO: add scripts/11-graph-eval_mlp.py to pipeline
# TODO: implement Distance Average Causal Effect: https://www.nature.com/articles/s41467-024-50813-z

# 3) Apply causal & structural comparison metrics to lstm (picabu, causal_discovery)

# 4) Apply causal & structural comparison metrics to cnn (picabu, causal_discovery)

# 5) Apply causal & structural comparison metrics to vae (picabu, causal_discovery)

# 6) Run RMSE, statistical metrics on all trained models

# 7) Apply power spectral density script from climatem module to: inference from models, targets 

# 7) Plotting...

# --- PART 3: INTERVENTIONS

# 1) Perturb data

# 2) Run picabu on all trained models with perturbed data

# 3) Run inference on all trained models w/ perturbed data

# 4) Get "gt" target from savar model for perturbed data

# 5) Apply RMSE on inference results

# --------- PART 0: GRAPH EVALUATION ---------

# 1) Create permuted dataset of graphs
# TODO: use 01-dataset-builder.py to create permuted dataset of graphs

# 2) Apply comparison metrics & do qualitative analysis on graphs
# TODO: use graph-eval.py 
