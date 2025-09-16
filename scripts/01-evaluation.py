import pickle
from climatem.data_loader import savar_dataset
from climatem.synthetic_data.graph_evaluation_ilija import extract_adjacency_matrix
import torch
import numpy as np

from gadjid import ancestor_aid, oset_aid, parent_aid, shd, sid

from causal_graph_comparison import *
from causal_graph_comparison.psd import power_spectral_density
from climatem.model.metrics import *
from pathlib import Path


def eval(model, dataset, num_modes, difficulty, seed):
    print(
        f"======= Evaluating model: {model} on SAVAR data with difficulty: {difficulty}, num_modes: {num_modes}, seed: {seed} ======="
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1) Load data
    print("Loading data...")

    data_name = f"modes_{num_modes}-diff_{difficulty}-seed_{seed}"
    experiment_name = f"{model}-{data_name}"
    savar_name = f"{dataset}-{data_name}"

    model_files = {
    "flat_graph-binary-cd": "cd_graph_path", # lin/nonlin doesn't matter, but it can
    "flat_graph-binary-crl": "crl_graph_path", # has to say lin/nonlin
    "samples_1000-rollouts_20steps": "rollouts_path",
    "samples_1000-rollouts_1steps": "next_step_path",
    "interventions": "intervention_path",
    }

    path = Path("/Volumes/GOOS/eval")

    model_paths = {}

    for file_name, var in list(model_files.items()):
        file = list(path.glob(f"{experiment_name}*{file_name}*.npz"))[0]
        model_paths[var] = file

    savar_paths = {}

    savar_files = {
    "flat_graph-binary-cd": "savar_cd_graph_path", # lin/nonlin doesn't matter, but it can
    "flat_graph-binary-crl": "savar_crl_graph_path", # has to say lin/nonlin
    }

    for file_name, var in list(savar_files.items()):
        file = list(path.glob(f"{savar_name}*{file_name}*.npz"))[0]
        savar_paths[var] = file

    next_step_data = np.load(model_paths["next_step_path"])
    next_step_outputs = next_step_data["outputs"]
    next_step_inputs = next_step_data["inputs"]
    next_step_targets = next_step_data["targets"]

    print("next_step_outputs", next_step_outputs.shape)
    print("next_step_targets", next_step_targets.shape)
    print("next_step_inputs", next_step_inputs.shape)

    rollouts_outputs = np.load(model_paths["rollouts_path"])["outputs"]
    rollouts_targets = np.load(model_paths["rollouts_path"])["targets"]
    rollouts_inputs = np.load(model_paths["rollouts_path"])["inputs"]

    print("rollouts_outputs", rollouts_outputs.shape)
    print("rollouts_targets", rollouts_targets.shape)
    print("rollouts_inputs", rollouts_inputs.shape)

    crl_graph = np.load(model_paths["crl_graph_path"])["graph"]
    cd_graph = np.load(model_paths["cd_graph_path"])["graph"]

    # causal graph learned from savar data
    savar_crl_graph = np.load(savar_paths["savar_crl_graph_path"])["graph"]
    savar_cd_graph = np.load(savar_paths["savar_cd_graph_path"])["graph"]

    intervention_data = np.load(model_paths["intervention_path"])
    intervention_outputs = intervention_data["intervened_outputs"]
    intervention_inputs = intervention_data["intervened_inputs"]
    intervention_targets = intervention_data["intervened_targets"]

    print("intervention_targets", intervention_targets.shape)
    print("intervention_outputs", intervention_outputs.shape)
    print("intervention_inputs", intervention_inputs.shape)

    print("Data done loading.")
    print(f"==============================\n")

    print("Graph shapes")

    print("Causal representation learning:")
    print("crl graph.shape", crl_graph.shape)
    print("crl graph \n", crl_graph)
    print("")
    print("savar crl graph.shape", savar_crl_graph.shape)
    print("savar crl graph \n", savar_crl_graph)
    print("")
    print("Causal discovery:")
    print("cd graph.shape", cd_graph.shape)
    print("cd graph \n", cd_graph)
    print("")
    print("savar cd graph.shape", savar_cd_graph.shape)
    print("savar cd graph \n", savar_cd_graph)
    print("")

    print("==============================\n")

    # -- 2. Calculate metrics --

    print("\nStructural & causal metrics on CRL...")

    # Gadjid package: G_true, G_guess
    crl_parent_aid = parent_aid(savar_crl_graph, crl_graph, edge_direction="from row to column")
    crl_oset_aid = oset_aid(savar_crl_graph, crl_graph, edge_direction="from row to column")
    crl_ancestor_aid = ancestor_aid(savar_crl_graph,crl_graph, edge_direction="from row to column")
    crl_sid = sid(savar_crl_graph, crl_graph, edge_direction="from row to column")
    crl_shd = shd(savar_crl_graph, crl_graph)
    # F1 & precision/recall: G_guess, G_true
    crl_f1 = f1_score(crl_graph, savar_crl_graph)
    crl_precision, crl_recall = precision_recall(crl_graph, savar_crl_graph)

    print("crl_parent_aid: ", crl_parent_aid)
    print("crl_oset_aid: ", crl_oset_aid)
    print("crl_ancestor_aid: ", crl_ancestor_aid)
    print("crl_sid: ", crl_sid)
    print("crl_shd: ", crl_shd)
    print("crl_f1: ", crl_f1)
    print("crl_precision: ", crl_precision)
    print("crl_recall: ", crl_recall)
    print("-----------------------------")

    print("\nStructural & causal metrics on CD...")

    # cd
    # Gadjid package: G_true, G_guess
    cd_parent_aid = parent_aid(savar_cd_graph, cd_graph, edge_direction="from row to column")
    cd_oset_aid = oset_aid(savar_cd_graph, cd_graph, edge_direction="from row to column")
    cd_ancestor_aid = ancestor_aid(savar_cd_graph, cd_graph, edge_direction="from row to column")
    cd_sid = sid(savar_cd_graph, cd_graph, edge_direction="from row to column")
    cd_shd = shd(savar_cd_graph, cd_graph)
    # F1 & precision/recall: G_guess, G_true
    cd_f1 = f1_score(cd_graph, savar_cd_graph)
    cd_precision, cd_recall = precision_recall(cd_graph, savar_cd_graph)

    print("cd_parent_aid", cd_parent_aid)
    print("cd_oset_aid", cd_oset_aid)
    print("cd_ancestor_aid", cd_ancestor_aid)
    print("cd_sid", cd_sid)
    print("cd_shd", cd_shd)
    print("cd_f1", cd_f1)
    print("cd_precision", cd_precision)
    print("cd_recall", cd_recall)

    print("==============================")

    # 3) Run RMSE, statistical metrics on all trained models

    print("\nRunning statistical metrics on next step...")

    # RMSE
    next_step_rmse = np.sqrt(np.mean((next_step_outputs - next_step_targets) ** 2))
    print("next_step_rmse", next_step_rmse)

    # MSE
    next_step_mse = np.mean((next_step_outputs - next_step_targets) ** 2)
    print("next_step_mse", next_step_mse)

    # MAE
    next_step_mae = np.mean(np.abs(next_step_outputs - next_step_targets))
    print("next_step_mae", next_step_mae)

    # R2
    next_step_r2 = 1 - (
        np.sum((next_step_outputs - next_step_targets) ** 2) / np.sum((next_step_targets - np.mean(next_step_targets)) ** 2)
    )
    print("next_step_r2", next_step_r2)

    # Variance
    next_step_variance = np.var(next_step_outputs)
    print("next_step_variance", next_step_variance)

    next_step_targets_variance = np.var(next_step_targets)
    print("next_step_targets_variance", next_step_targets_variance)

    # Bias
    next_step_bias = np.mean(np.mean(next_step_outputs - next_step_targets) ** 2) - next_step_targets_variance
    print("next_step_bias", next_step_bias)

    # stdev
    next_step_stdev = np.std(next_step_outputs)
    print("next_step_stdev", next_step_stdev)

    next_step_targets_stdev = np.std(next_step_targets)
    print("next_step_targets_stdev", next_step_targets_stdev)

    # range
    next_step_range = np.max(next_step_outputs) - np.min(next_step_outputs)
    print("next_step_range", next_step_range)

    next_step_targets_range = np.max(next_step_targets) - np.min(next_step_targets)
    print("next_step_targets_range", next_step_targets_range)

    print("==============================")

    # 4) Apply power spectral density script from climatem module to: inference from models, targets

    print("\nRunning power spectral density on rollouts...")

    # LSD = least square difference not linear spectral density
    LSD, fft_coeffs_rollouts, fft_coeffs_savar = power_spectral_density(model_paths["rollouts_path"], num_modes)

    # 5) Interventions
    print("\nAnalyzing interventions...")

    intervention_rmse = np.sqrt(np.mean((intervention_outputs - intervention_targets) ** 2))
    print("intervention_rmse", intervention_rmse)

    print("==============================\n")

    # 9) Sav
    output_dict = {
        "crl_parent_aid": crl_parent_aid,
        "crl_oset_aid": crl_oset_aid,
        "crl_ancestor_aid": crl_ancestor_aid,
        "crl_sid": crl_sid,
        "crl_shd": crl_shd,
        "crl_f1": crl_f1,
        "crl_precision": crl_precision,
        "crl_recall": crl_recall,
        "cd_parent_aid": cd_parent_aid,
        "cd_oset_aid": cd_oset_aid,
        "cd_ancestor_aid": cd_ancestor_aid,
        "cd_sid": cd_sid,
        "cd_shd": cd_shd,
        "cd_f1": cd_f1,
        "cd_precision": cd_precision,
        "cd_recall": cd_recall,
        "next_step_rmse": next_step_rmse,
        "next_step_mse": next_step_mse,
        "next_step_mae": next_step_mae,
        "next_step_r2": next_step_r2,
        "next_step_variance": next_step_variance,
        "next_step_targets_variance": next_step_targets_variance,
        "next_step_bias": next_step_bias,
        "next_step_stdev": next_step_stdev,
        "next_step_targets_stdev": next_step_targets_stdev,
        "next_step_range": next_step_range,
        "next_step_targets_range": next_step_targets_range,
        "lsd": LSD,
        "fft_coeffs_rollouts": fft_coeffs_rollouts,
        "fft_coeffs_savar": fft_coeffs_savar,
        "intervention_rmse": intervention_rmse,
        "intervention_inputs_sample": intervention_inputs[0],
        "intervention_outputs_sample": intervention_outputs[0],
        "intervention_targets_sample": intervention_targets[0],
        "next_step_sample": next_step_outputs[0],
        "next_step_targets_sample": next_step_targets[0],
        "next_step_inputs_sample": next_step_inputs[0],
        "rollouts_sample": rollouts_outputs[0],
        "rollouts_targets_sample": rollouts_targets[0],
        "rollouts_inputs_sample": rollouts_inputs[0],
    }

    return output_dict

# ---- CMD LINE ARGS
results_pkl_path = OUTPUTS_DIR / f"evaluation_final.pkl"
output_dict = {}
seed = 1
dataset = "savar"

for model in ["mlp", "cnn", "lstm"]:
    for num_modes in [4, 16]:
        for difficulty in ["easy","med_easy", "med_hard", "hard"]:
            try:
                print(f"Evaluating model: {model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}")
                outputs = eval(model, dataset, num_modes, difficulty, seed)
                output_dict[f"{model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}"] = outputs
            except Exception as e:
                print(f"====== ERROR evaluating model: {model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}")
                print(e)
                continue
            print("======== End =========\n")

# Save evaluation results to pickle file
with open(results_pkl_path, "wb") as f:
    pickle.dump(output_dict, f)
print("keys:")
for key in output_dict.keys():
    print(key)
print (f"Saved evaluation results to {results_pkl_path}")
