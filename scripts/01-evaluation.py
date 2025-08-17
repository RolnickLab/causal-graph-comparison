import pickle
from climatem.synthetic_data.graph_evaluation_ilija import extract_adjacency_matrix
import torch
import numpy as np

from gadjid import ancestor_aid, oset_aid, parent_aid, shd, sid

from causal_graph_comparison import *
from causal_graph_comparison.psd import power_spectral_density
from climatem.model.metrics import *

#TODO flatten gt_adj
#TODO compare cd and crl on savar
#TODO why is cd graph 64x64 for vae instead of 80x80
#TODO run vae main 4 hard <-- make sure crl is saved with proper shape (6, 4, 4)
#TODO run vae 64 hard picabu training


def eval(model, num_modes, difficulty, seed):
    print(
        f"======= Evaluating model: {model} on SAVAR data with difficulty: {difficulty}, num_modes: {num_modes}, seed: {seed} ======="
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_name = f"modes_{num_modes}-diff_{difficulty}-seed_{seed}"
    experiment_name = f"{model}-{data_name}"
    savar_name = f"{dataset}-{data_name}"

    crl_graph_path = OUTPUTS_DIR / f"{experiment_name}-flat_graph-binary-crl.npz"
    cd_graph_path = OUTPUTS_DIR / f"{experiment_name}-flat_graph-binary-cd.npz"

    savar_crl_graph_path = OUTPUTS_DIR / f"{savar_name}-flat_graph-binary-crl.npz"
    savar_cd_graph_path = OUTPUTS_DIR / f"{savar_name}-flat_graph-binary-cd.npz"

    next_step_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_1steps.npz"
    next_step_data = np.load(next_step_path)
    print("next_step_path", next_step_data["targets"].shape)

    # 1) Load data

    print("Loading data...")

    crl_graph = np.load(crl_graph_path)["graph"]
    cd_graph = np.load(cd_graph_path)["graph"]
    savar_crl_graph = np.load(savar_crl_graph_path)["graph"]
    savar_cd_graph = np.load(savar_cd_graph_path)["graph"]

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

    # 3) Apply causal & structural comparison metrics to mlp (picabu, causal_discovery)
    # TODO: add scripts/11-graph-eval_mlp.py to pipeline
    # TODO: implement Distance Average Causal Effect: https://www.nature.com/articles/s41467-024-50813-z
    print("==============================")

    # 0) Report recovery accuracy of causal discovery vs cdsd (picabu) for dataset
    # get gt_adj from data file
    # compare learned graphs on savar to gt_adj

    print("Loading gt_adj from data file...")

    links_coeffs = np.load(DATA_DIR / f"{data_name}_parameters.npy", allow_pickle=True).item()["links_coeffs"]
    print("links_coeffs", links_coeffs)

    for child_node, relationships in links_coeffs.items():
        for relationship in relationships:
            parent_node = relationship[0][0]
            time_lag = abs(relationship[0][1])  # Convert negative lag to positive
            weight = relationship[1]
            print(f"{parent_node} --> {child_node}, lag {time_lag}, weight {weight}")

    gt_adj = np.array(extract_adjacency_matrix(links_coeffs, num_modes, tau=5))
    print("gt_adj.shape", gt_adj.shape)
    print("gt_adj", gt_adj)

    # -- 4. Calculate metrics --

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

    # 7) Run RMSE, statistical metrics on all trained models

    print("\nRunning statistical metrics on next step...")

    next_step_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_1steps.npz"
    next_step = np.load(next_step_path)["outputs"]
    next_step_inputs = np.load(next_step_path)["inputs"]
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
    next_step_r2 = 1 - (
        np.sum((next_step - next_step_targets) ** 2) / np.sum((next_step_targets - np.mean(next_step_targets)) ** 2)
    )
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

    print("==============================")

    # 8) Apply power spectral density script from climatem module to: inference from models, targets

    print("\nRunning power spectral density on rollouts...")

    rollouts_path = f"{OUTPUTS_DIR}/{experiment_name}-samples_1000-rollouts_20steps.npz"
    rollouts = np.load(rollouts_path)["outputs"]
    rollouts_targets = np.load(rollouts_path)["targets"]
    rollouts_inputs = np.load(rollouts_path)["inputs"]
    # LSD = least square difference not linear spectral density
    LSD, fft_coeffs_rollouts, fft_coeffs_savar = power_spectral_density(rollouts_path, num_modes)

    # 9) Interventions
    print("\nAnalyzing interventions...")

    intervention_path = f"{OUTPUTS_DIR}/{experiment_name}-interventions.npz"
    intervention_data = np.load(intervention_path)
    intervention_outputs = intervention_data["intervened_outputs"]
    intervention_inputs = intervention_data["intervened_inputs"]
    print("intervention_outputs", intervention_outputs.shape)

    intervention_targets = intervention_data["intervened_targets"]
    print("intervention_targets", intervention_targets.shape)

    intervention_rmse = np.sqrt(np.mean((intervention_outputs - intervention_targets) ** 2))
    print("intervention_rmse", intervention_rmse)

    print("==============================\n")

    need = [
        # "gt_shd",
        # "gt_f1",
        # "gt_precision",
        # "gt_recall",
        # "gt_parent_aid",
        # "gt_oset_aid",
        # "gt_ancestor_aid",
        # "gt_sid",
        # "time series rollouts",
        # "time series targets",
        # "5 prev time steps + 1 next time step",
    ]

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
        "next_step_sample": next_step[0],
        "next_step_targets_sample": next_step_targets[0],
        "next_step_inputs_sample": next_step_inputs[0],
        "rollouts_sample": rollouts[0],
        "rollouts_targets_sample": rollouts_targets[0],
        "rollouts_inputs_sample": rollouts_inputs[0],
    }

    return output_dict


# ---- CMD LINE ARGS
results_pkl_path = OUTPUTS_DIR / f"evaluation.pkl"
output_dict = {}
seed = 1
dataset = "savar"

for model in ["mlp", "cnn", "lstm", "vae"]:
    for num_modes in [4, 16, 64]:
        for difficulty in ["easy", "med_easy", "med_hard", "hard"]:
            if model == "vae" and num_modes == 64 and difficulty == "hard":
                print("Skipping vae 64 hard...")
                continue
            if model == "vae" and num_modes == 64 and difficulty == "med_hard":
                print("Skipping vae 64 med_hard...")
                continue
            try:
                print(f"Evaluating model: {model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}")
                outputs = eval(model, num_modes, difficulty, seed)
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
