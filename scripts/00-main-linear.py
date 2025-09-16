import argparse
from climatem.model.tsdcd_latent import LatentTSDCD
import numpy as np
import torch
import wandb
from math import sqrt


from causal_graph_comparison import *  # Directory paths
from causal_graph_comparison.causal_discovery import causal_discovery
from causal_graph_comparison.interventions import intervention
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison.picabu_helpers import load_picabu_config
from causal_graph_comparison.savar import generate_savar_data
from causal_graph_comparison.trainer import run_trainer
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison.vae import train_vae
from causal_graph_comparison.mlp import MLP
from causal_graph_comparison.lstm import LSTM
from causal_graph_comparison.cnn import CNN
from causal_graph_comparison.graph_utils import (
    binarize_array,
    flatten_temporal_adjacency_graph,
    permute_graph,
)
from causal_graph_comparison.rollouts import get_targets, run_rollouts

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

# ---- CMD LINE ARGS

args = argparse.ArgumentParser()
args.add_argument(
    "--difficulty",
    type=str,
    choices=["easy", "med_easy", "med_hard", "hard"],
    help="Difficulty level: easy, med_easy, med_hard, hard",
)
args.add_argument("--num_modes", type=int, choices=[4, 16, 36, 64], help="Number of modes: 4, 16, 36, 64")
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

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")


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
    rollout_params,
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
    "easy": N / (N**2 * tau),  # for N = 4, tau = 5, prob = 4 / 80 = 0.05
    "med_easy": 2 * N / (N**2 * tau),  # for N = 4, tau = 5, prob = 8 / 80 = 0.1
    "med_hard": 3 * N / (N**2 * tau),  # for N = 4, tau = 5, prob = 12 / 80 = 0.15
    "hard": (N + N * (N - 1) / denom) / (N**2 * tau),  # for N = 4, tau = 5, prob = 10 / 80 = 0.125
}

optim_params.sparsity_upper_threshold = sparsity_thresholds[args.difficulty]

# set device

device = torch.device("cpu")
if torch.cuda.is_available():
    print("CUDA is available.")
    if experiment_params.gpu:
        device = torch.device("cuda")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.set_default_device("cuda")
        # torch.set_default_dtype(torch.cuda.float32)
    else:
        print ("...but we decided not to use it.")

# Expected a 'cpu' device type for generator but found 'cuda


# generate savar data
datamodule = generate_savar_data(experiment_params, data_params, gt_params, train_params, picabu_params, optim_params, plot_params, savar_params, rollout_params, reload_data=True)

# print datamodule
print("datamodule.savar_gt_adj shape: ", datamodule.savar_gt_adj.shape)
print("datamodule.savar_gt_adj: ", datamodule.savar_gt_adj)

print("datamodule.savar_gt_modes_weights shape: ", datamodule.savar_gt_modes_weights.shape)
print(
    "datamodule.savar_links_coeffs (ground truth causal links): ",
    datamodule.savar_links_coeffs,
)
print("datamodule.savar_gt_modes shape: ", datamodule.savar_gt_modes.shape)
if datamodule.savar_gt_modes_weights is None:
    raise ValueError(
        "datamodule.savar_gt_modes_weights is None, try setting reload_climate_set_data in data_params to True"
    )

# generate train and test dataloaders
train_dataset = datamodule._data_train
test_dataset = datamodule._data_val

print("train_dataset length: ", len(train_dataset))
print("test_dataset length: ", len(test_dataset))

# assert train_dataset.device == torch.device("cuda"), "train_dataset must be on the same device as the model"
# assert test_dataset.device == torch.device("cuda"), "test_dataset must be on the same device as the model"

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=data_params.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=data_params.eval_batch_size, shuffle=False)
# train_loader = datamodule.train_dataloader(accelerator)
# test_loader = datamodule.val_dataloader()


# ------ param dictionaries ------

# load inference params from config file
n_samples = trained_model_params["common"]["test_params"]["num_samples"]
rollouts = trained_model_params["common"]["test_params"]["rollout_timesteps"]
inference_batch_size = trained_model_params["common"]["test_params"]["inference_batch_size"]

# create inference loader from test dataset with batch size of 1000 (i.e. pass all samples at once)
inference_loader = torch.utils.data.DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False)

wandb_dict = {
    "difficulty": args.difficulty,
    "num_modes": args.num_modes,
    "seed": args.seed,
    "dataset_type": args.dataset_type,
    "resolution": args.resolution,
}

# params for training picabu
picabu_train_args = {
    "datamodule": datamodule,
    "experiment_params": experiment_params,
    "data_params": data_params,
    "gt_params": gt_params,
    "train_params": train_params,
    "model_params": picabu_params,
    "optim_params": optim_params,
    "plot_params": plot_params,
    "savar_params": savar_params,
}

# params for rollouts
rollout_args = {
    "n_samples": n_samples,
    "test_loader": inference_loader,
    "device": device,
}

causal_discovery_args = {
    "num_modes": args.num_modes,
    "links_coeffs": datamodule.savar_links_coeffs,
    "tau_max": experiment_params.tau,
    "difficulty": args.difficulty,
    "seed": args.seed,
}

experiment_name = f"modes_{args.num_modes}-diff_{args.difficulty}-seed_{args.seed}"

model_train_args = {
    "datamodule": datamodule,
    "params": trained_model_params,
    "device": device,
}


# ====== PICABU ======
# 0.1) Get "ground truth" causal graph for savar------
# saves results to scratch/results/SAVAR_DATA_TEST/savar-modes_{modes}-diff_{difficulty}-seed_{seed}/ (graph)

if args.model == "picabu":

    experiment_name = f"savar-{experiment_name}-linear"

    # === Causal discovery ===
    # 0.1) Run causal discovery on savar ground truth
    ## 0.1a) get targets saved from rollout (1000 samples, 20 timesteps)

    # savar_targets_path_one_step = get_targets(inference_loader, n_samples, 1, experiment_name)

    # savar_targets_path = get_targets(inference_loader, n_samples, rollouts, experiment_name)
    # savar_targets = np.load(savar_targets_path)["targets"]
    # ## 0.1b) run causal discovery
    # savar_graph, savar_val_matrix, savar_p_matrix, savar_corr_matrix, savar_var_names = causal_discovery(
    #     timeseries=savar_targets,
    #     model_name="savar",
    #     subsample="mean",
    #     **causal_discovery_args,
    # )

    # # 0.2) Flatten temporal adjacency graphs: causal discovery
    # gt_flat_cd_graph = flatten_temporal_adjacency_graph(
    #     shape="parent_child_time", causal_method="cd", experiment_name=experiment_name
    # )

    # # 0.3) Binarize causal discovery & causal representation learning graphs
    # gt_flat_cd_graph = binarize_array(gt_flat_cd_graph)
    # np.savez(
    #     f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-cd.npz",
    #     graph=gt_flat_cd_graph,
    # )

    # === Causal representation learning ===

    # 0.4) Train picabu on savar --> saves scratch/results/SAVAR_DATA_TEST/picabu_savar-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
    print(f"=== training picabu on {experiment_name}")
    run = wandb.init(project="climatem", config={"model": "picabu", **wandb_dict})
    train_picabu(wandb=run, **picabu_train_args)
    run.finish()

    # 0.5) Permute learned adjacency graph outputs using ground truth so modes are in same order as in ground truth
    picabu_permuted_crl_graph = permute_graph(datamodule, f"{experiment_name}-linear")

    # 0.6) Flatten temporal adjacency graphs: causal representation learning
    gt_flat_crl_graph = flatten_temporal_adjacency_graph(
        shape="time_child_parent", causal_method="crl", graph=picabu_permuted_crl_graph, experiment_name=f"{experiment_name}-linear"
    )

    # 0.7) Binarize causal discovery & causal representation learning graphs
    gt_flat_crl_graph = binarize_array(gt_flat_crl_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-linear-flat_graph-binary-crl.npz",
        graph=gt_flat_crl_graph,
    )

elif args.model == "vae":

    # ====== VAE ======

    experiment_name = f"vae-{experiment_name}-linear"

    # 1.1) Train vae --> saves scratch/cgc/models/vae-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
    print("=== training picabu as vae on savar")
    run = wandb.init(project="climatem", config={"model": "vae", **wandb_dict})
    vae_path = train_vae(trained_model_params=trained_model_params, wandb=run, **picabu_train_args)
    wandb.finish()

    # 1.2) Train picabu on vae --> saves scratch/results/SAVAR_DATA_TEST/picabu-vae-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
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
        d=1,  # number of datasets
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

    # === Causal discovery ===

    # 1.2) Run causal discovery on vae
    ## 1.2a) run inference --> saves scratch/cgc/outputs/vae-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
    print(f"Running inference on vae: {n_samples} samples, {rollouts} timesteps...")
    vae_rollouts_path = run_rollouts(model=vae_model, experiment_name=experiment_name, rollouts=rollouts, **rollout_args)

    ## 1.2b) run causal discovery
    vae_graph, vae_val_matrix, vae_p_matrix, vae_corr_matrix, vae_var_names = causal_discovery(
        timeseries=vae_rollouts_path,
        model_name=vae_model.name,
        subsample="mean",
        experiment_name=experiment_name,
        **causal_discovery_args,
    )

    # 1.3) Run rollouts with 1 step for vae (for rmse, statistical metrics)
    vae_rollouts_path = run_rollouts(model=vae_model, experiment_name=experiment_name, rollouts=1, **rollout_args)

     # 1.7) Flatten temporal adjacency graphs: causal discovery
    vae_flat_cd_graph = flatten_temporal_adjacency_graph(
        shape="parent_child_time", causal_method="cd", experiment_name=experiment_name
    )

    # 1.8) Binarize causal discovery & causal representation learning graphs
    vae_flat_cd_graph = binarize_array(vae_flat_cd_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-cd.npz",
        graph=vae_flat_cd_graph,
    )

    if args.num_modes == 4:
        if args.difficulty == "med_easy" or args.difficulty == "med_hard":
            print("Skipping training linear picabu on vae formed_easy and med_hard for 4 modes")
            quit()


    # === Causal representation learning ===

    # 1.4) Train picabu on vae --> saves scratch/results/SAVAR_DATA_TEST/picabu-vae-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
    run = wandb.init(project="climatem", config={"model": "picabu-vae", **wandb_dict})
    train_picabu(trained_model=vae_model, trained_model_params=trained_model_params, wandb=run, **picabu_train_args)
    run.finish()

    # 1.5) Permute learned adjacency graph outputs using ground truth so modes are in same order as in ground truth
    vae_permuted_crl_graph = permute_graph(datamodule, experiment_name)

    # 1.6) Flatten temporal adjacency graphs: causal representation learning
    vae_flat_crl_graph = flatten_temporal_adjacency_graph(
        shape="time_child_parent", causal_method="crl", graph=vae_permuted_crl_graph, experiment_name=experiment_name
    )

    vae_flat_crl_graph = binarize_array(vae_flat_crl_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-crl.npz",
        graph=vae_flat_crl_graph,
    )

    #2.10) Run interventions

    print(f"Running intervention on {args.model}: generating next step + targets")

    vae_intervention_path = intervention(model=vae_model, experiment_name=experiment_name, test_loader=inference_loader, datamodule=datamodule, device=device)

# ====== MLP ======
elif args.model == "mlp":

    experiment_name = f"mlp-{experiment_name}"

    # 2.1) Train mlp --> saves scratch/cgc/models/mlp-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
    mlp_input_size = experiment_params.d_x * experiment_params.tau
    mlp_output_size = experiment_params.d_x * experiment_params.future_timesteps
    mlp_layers = trained_model_params["mlp"]["model_params"]["num_layers"]

    mlp = MLP(input_size=mlp_input_size, output_size=mlp_output_size, num_layers=mlp_layers).to(device)
    print("=== training mlp on savar")
    run = wandb.init(project="climatem", config={"model": "mlp", **wandb_dict})
    mlp_path, best_mlp_path = run_trainer(mlp, experiment_name, **model_train_args)
    wandb.finish()

    # load trained mlp model
    if best_mlp_path.exists():
        mlp_model = torch.load(best_mlp_path, map_location=device, weights_only=False)
    else:
        mlp_model = torch.load(mlp_path, map_location=device, weights_only=False)

    # move model to device
    mlp_model.to(device)

    # === Causal discovery ===

    # 2.2) run causal discovery on mlp --> saves scratch/cgc/outputs/mlp-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz

    ## 2.2a) run inference - create 1000 samples 20 timesteps
    print(f"Running inference on mlp: {n_samples} samples, {rollouts} timesteps...")

    mlp_rollouts_path = run_rollouts(model=mlp_model, experiment_name=experiment_name, rollouts=rollouts, **rollout_args)

    ## 2.2b) run causal discovery
    mlp_graph, mlp_val_matrix, mlp_p_matrix, mlp_corr_matrix, mlp_var_names = causal_discovery(
        timeseries=mlp_rollouts_path,
        model_name=mlp_model.name,
        subsample="mean",
        experiment_name=experiment_name,
        **causal_discovery_args,
    )

    # 2.3) Run rollouts with 1 step for mlp (for rmse, statistical metrics)
    mlp_rollouts_path = run_rollouts(model=mlp_model, experiment_name=experiment_name, rollouts=1, **rollout_args)

    # 2.4) Flatten temporal adjacency graphs: causal discovery
    mlp_flat_cd_graph = flatten_temporal_adjacency_graph(
        shape="parent_child_time", causal_method="cd", experiment_name=experiment_name
    )

    # 2.5) Binarize causal discovery learning graph
    mlp_flat_cd_graph_binary = binarize_array(mlp_flat_cd_graph)
    np.savez(f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-cd.npz", graph=mlp_flat_cd_graph_binary)
    print("Shape of mlp_flat_cd_graph-binary: ", mlp_flat_cd_graph_binary.shape)


    # === Causal representation learning ===

    # 2.6) Train picabu on mlp --> saves scratch/results/SAVAR_DATA_TEST/picabu-mlp-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
    run = wandb.init(project="climatem", config={"model": "picabu-mlp", **wandb_dict})
    train_picabu(trained_model=mlp_model, trained_model_params=trained_model_params, wandb=run, **picabu_train_args)
    run.finish()

    # 2.7) Permute learned adjacency graph outputs using ground truth so modes are in same order as in ground truth
    mlp_permuted_crl_graph = permute_graph(datamodule, f"{experiment_name}-linear")
    
    # 2.8) Flatten temporal adjacency graphs: causal representation learning
    mlp_flat_crl_graph = flatten_temporal_adjacency_graph(
        shape="time_child_parent", causal_method="crl", graph=mlp_permuted_crl_graph, experiment_name=f"{experiment_name}-linear"
    )

    # 2.9) Binarize causal representation learning graphs
    mlp_flat_crl_graph_binary = binarize_array(mlp_flat_crl_graph)
    np.savez(f"{OUTPUTS_DIR}/{experiment_name}-linear-flat_graph-binary-crl.npz", graph=mlp_flat_crl_graph_binary)
    print("Shape of mlp_flat_crl_graph-binary: ", mlp_flat_crl_graph_binary.shape)

    #2.10) Run interventions

    print(f"Running intervention on {args.model}: generating next step + targets")

    mlp_intervention_path = intervention(model=mlp_model, experiment_name=experiment_name, test_loader=inference_loader, datamodule=datamodule, device=device)

    # ====== LSTM ======
elif args.model == "lstm":

    experiment_name = f"lstm-{experiment_name}"

    # 3.1) Train lstm --> saves scratch/cgc/models/lstm-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
    input_size = experiment_params.d_x
    num_layers = trained_model_params["lstm"]["model_params"]["num_layers"]

    lstm = LSTM(input_size=input_size, hidden_size=int(input_size / 2), num_layers=num_layers).to(device)
    print("=== training lstm on savar")
    run = wandb.init(project="climatem", config={"model": "lstm", **wandb_dict})
    lstm_path, best_lstm_path = run_trainer(lstm, experiment_name, **model_train_args)
    wandb.finish()

    # load trained lstm model
    if best_lstm_path.exists():
        lstm_model = torch.load(best_lstm_path, map_location=device, weights_only=False)
    else:
        lstm_model = torch.load(lstm_path, map_location=device, weights_only=False)

    # move model to device
    lstm_model.to(device)

    # === Causal discovery ===

    # 3.2) Run causal discovery on lstm
    ## 3.2a) run inference --> saves scratch/cgc/outputs/lstm-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
    print(f"Running inference on lstm: {n_samples} samples, {rollouts} timesteps...")
    lstm_rollouts_path = run_rollouts(model=lstm_model, experiment_name=experiment_name, rollouts=rollouts, **rollout_args)

    ## 3.2b) run causal discovery
    lstm_graph, lstm_val_matrix, lstm_p_matrix, lstm_corr_matrix, lstm_var_names = causal_discovery(
        timeseries=lstm_rollouts_path,
        model_name=lstm_model.name,
        subsample="mean",
        experiment_name=experiment_name,
        **causal_discovery_args,
    )

    # 3.3) Run rollouts with 1 step for lstm (for rmse, statistical metrics)
    lstm_rollouts_path = run_rollouts(model=lstm_model, experiment_name=experiment_name, rollouts=1, **rollout_args)

    # 3.4) Flatten temporal adjacency graphs: causal discovery
    lstm_flat_cd_graph = flatten_temporal_adjacency_graph(
        shape="parent_child_time", causal_method="cd", experiment_name=experiment_name
    )

    # 3.5) Binarize causal discovery learning graph
    lstm_flat_cd_graph = binarize_array(lstm_flat_cd_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-linear-flat_graph-binary-cd.npz",
        graph=lstm_flat_cd_graph,
    )

    # === Causal representation learning ===

    # 3.6) Train picabu on lstm --> saves scratch/results/SAVAR_DATA_TEST/picabu-lstm-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
    run = wandb.init(project="climatem", config={"model": "picabu-lstm", **wandb_dict})
    train_picabu(trained_model=lstm_model, trained_model_params=trained_model_params, wandb=run, **picabu_train_args)
    run.finish()

    # 3.7) Permute learned adjacency graph outputs using ground truth so modes are in same order as in ground truth
    lstm_permuted_crl_graph = permute_graph(datamodule, f"{experiment_name}-linear")

    # 3.8) Flatten temporal adjacency graphs: causal representation learning
    lstm_flat_crl_graph = flatten_temporal_adjacency_graph(
        shape="time_child_parent", causal_method="crl", graph=lstm_permuted_crl_graph, experiment_name=f"{experiment_name}-linear"
    )

    # 3.9) Binarize causal discovery & causal representation learning graphs
    lstm_flat_crl_graph = binarize_array(lstm_flat_crl_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-linear-flat_graph-binary-crl.npz",
        graph=lstm_flat_crl_graph,
    )

    #2.10) Run interventions

    print(f"Running intervention on {args.model}: generating next step + targets")

    lstm_intervention_path = intervention(model=lstm_model, experiment_name=experiment_name, test_loader=inference_loader, datamodule=datamodule, device=device)


# ====== CNN ======
elif args.model == "cnn":

    experiment_name = f"cnn-{experiment_name}"

    # 4.1) Train cnn --> saves scratch/cgc/models/cnn-modes_{modes}-diff_{difficulty}-seed_{seed}.pth
    cnn_input_channels = experiment_params.tau
    cnn_output_channels = experiment_params.future_timesteps
    cnn_image_size = experiment_params.lon

    cnn = CNN(
        input_channels=cnn_input_channels,
        output_channels=cnn_output_channels,
        image_size=cnn_image_size,
        channels=trained_model_params["cnn"]["model_params"]["channels"],
        kernels=trained_model_params["cnn"]["model_params"]["kernels"],
        fc_layers=trained_model_params["cnn"]["model_params"]["fc_layers"],
    ).to(device)

    print("=== training cnn on savar")
    run = wandb.init(project="climatem", config={"model": "cnn", **wandb_dict})
    cnn_path, best_cnn_path = run_trainer(cnn, experiment_name, **model_train_args)
    wandb.finish()

    # load trained cnn model
    if best_cnn_path.exists():
        cnn_model = torch.load(best_cnn_path, map_location=device, weights_only=False)
    else:
        cnn_model = torch.load(cnn_path, map_location=device, weights_only=False)
    
    # move model to device
    cnn_model.to(device)

    # === Causal discovery ===

    # 4.2) Run causal discovery on cnn
    ## 4.2a) run inference --> saves scratch/cgc/outputs/cnn-modes_4-diff_easy-seed_1-samples_999-rollouts_20steps.npz
    print(f"Running inference on cnn: {n_samples} samples, {rollouts} timesteps...")

    cnn_rollouts_path = run_rollouts(model=cnn_model, experiment_name=experiment_name, rollouts=rollouts, **rollout_args)

    ## 4.2b) run causal discovery
    cnn_graph, cnn_val_matrix, cnn_p_matrix, cnn_corr_matrix, cnn_var_names = causal_discovery(
        timeseries=cnn_rollouts_path,
        model_name=cnn_model.name,
        subsample="mean",
        experiment_name=experiment_name,
        **causal_discovery_args,
    )

    # 4.3) Run rollouts with 1 step for cnn (for rmse, statistical metrics)
    cnn_rollouts_path = run_rollouts(model=cnn_model, experiment_name=experiment_name, rollouts=1, **rollout_args)

    # 4.4) Flatten temporal adjacency graphs: causal discovery
    cnn_flat_cd_graph = flatten_temporal_adjacency_graph(
        shape="parent_child_time", causal_method="cd", experiment_name=experiment_name
    )

    # 4.5) Binarize causal discovery learned graph
    cnn_flat_cd_graph = binarize_array(cnn_flat_cd_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-cd.npz",
        graph=cnn_flat_cd_graph,
    )

    # === Causal representation learning ===

    # 4.6) Train picabu on cnn --> saves scratch/results/SAVAR_DATA_TEST/picabu-cnn-modes_{modes}-diff_{difficulty}-seed_{seed}/plots/graphs.npy (graph)
    run = wandb.init(project="climatem", config={"model": "picabu-cnn", **wandb_dict})
    train_picabu(trained_model=cnn_model, trained_model_params=trained_model_params, wandb=run, **picabu_train_args)
    run.finish()

    # 4.7) Permute learned adjacency graph outputs using ground truth so modes are in same order as in ground truth
    cnn_permuted_crl_graph = permute_graph(datamodule, f"{experiment_name}-linear")

    # 4.8) Flatten temporal adjacency graphs: causal representation learning
    cnn_flat_crl_graph = flatten_temporal_adjacency_graph(
        shape="time_child_parent", causal_method="crl", graph=cnn_permuted_crl_graph, experiment_name=f"{experiment_name}-linear"
    )

    # 4.9) Binarize causal discovery & causal representation learning graphs
    cnn_flat_crl_graph = binarize_array(cnn_flat_crl_graph)
    np.savez(
        f"{OUTPUTS_DIR}/{experiment_name}-linear-flat_graph-binary-crl.npz",
        graph=cnn_flat_crl_graph,
    )

    #2.10) Run interventions

    print(f"Running intervention on {args.model}: generating next step + targets")

    cnn_intervention_path = intervention(model=cnn_model, experiment_name=experiment_name, test_loader=inference_loader, datamodule=datamodule, device=device)

