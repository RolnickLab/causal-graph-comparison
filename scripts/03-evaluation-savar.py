import pickle
from climatem.synthetic_data.graph_evaluation_ilija import extract_adjacency_matrix
import torch
import numpy as np

from gadjid import ancestor_aid, oset_aid, parent_aid, shd, sid

from causal_graph_comparison import *
from causal_graph_comparison.graph_utils import binarize_array, flatten_temporal_adjacency_graph
from causal_graph_comparison.psd import power_spectral_density
from climatem.model.metrics import *

def eval_savar(dataset,num_modes, difficulty, seed, linear=False):
    print(
        f"======= Evaluating CD/CRL on SAVAR data with difficulty: {difficulty}, num_modes: {num_modes}, seed: {seed} ======="
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    data_name = f"modes_{num_modes}-diff_{difficulty}-seed_{seed}"
    crl_name = data_name

    if linear:
        crl_name = f"{crl_name}-linear"

    savar_name = f"{dataset}-{data_name}"
    gt_name = f"gt-{savar_name}"

    savar_crl_graph_path = OUTPUTS_DIR / f"{dataset}-{crl_name}-flat_graph-binary-crl.npz"
    savar_cd_graph_path = OUTPUTS_DIR / f"{dataset}-{data_name}-flat_graph-binary-cd.npz"

    # temporal_crl_graph_path = OUTPUTS_DIR / f"{savar_name}-picabu_cdsd.npz"
    # temporal_crl_graph = np.load(temporal_crl_graph_path)["val_matrix"]
    # print("Learned temporal CRL graph:")
    # print("temporal_crl_graph.shape", temporal_crl_graph.shape)
    # print("temporal_crl_graph \n", temporal_crl_graph)

    # temporal_cd_graph = OUTPUTS_DIR / f"{savar_name}-pcmci_causal_discovery.npz"
    # temporal_cd_graph = np.load(temporal_cd_graph)["val_matrix"]
    # print("Learned temporalCD graph:")
    # print("temporal_cd_graph.shape", temporal_cd_graph.shape)
    # print("temporal_cd_graph \n", temporal_cd_graph)

    # flatten temporal cd graph
    # flat_cd_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", graph=temporal_cd_graph, experiment_name=savar_name)
    # binary_cd_graph = binarize_array(flat_cd_graph)
    # print("Binary CD graph:")
    # print("binary_cd_graph.shape", binary_cd_graph.shape)
    # print("binary_cd_graph \n", binary_cd_graph)

    # 1) Load data

    print("Loading data...")

    savar_crl_flat = np.load(savar_crl_graph_path)["graph"]
    savar_cd_flat = np.load(savar_cd_graph_path)["graph"]

    print("Causal representation learning:")
    print("crl graph.shape", savar_crl_flat.shape)
    print("")
    print("Causal discovery:")
    print("cd graph.shape", savar_cd_flat.shape)

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
    # add extra tau to first dimension of gt_adj dimensions (5, 4, 4) -> (6, 4, 4)
    gt_adj = np.concatenate([np.zeros((1, num_modes, num_modes)), gt_adj], axis=0)
    print("GT binary adjacency matrix:")
    print("gt_adj.shape", gt_adj.shape)
    print("gt_adj \n", gt_adj)

    # flatten gt_adj
    print("Flattening gt_adj...")
    gt_adj = flatten_temporal_adjacency_graph(shape="time_child_parent", causal_method="cd", graph=gt_adj, experiment_name=gt_name)
    gt_adj_flat = gt_adj.astype(np.int8)
    print("gt_adj_flat.shape", gt_adj_flat.shape)
    print("gt_adj_flat \n", gt_adj_flat)

    print("CRL flattened graph:")
    print("savar_crl_flat.shape", savar_crl_flat.shape)
    print("savar_crl_flat \n", savar_crl_flat)

    print("CD flattened graph:")
    print("savar_cd_flat.shape", savar_cd_flat.shape)
    print("savar_cd_flat \n", savar_cd_flat)

    gt_adj_flat = gt_adj_flat.astype(np.int8)
    savar_crl_flat = savar_crl_flat.astype(np.int8)
    savar_cd_flat = savar_cd_flat.astype(np.int8)

    gt_adj_flat = np.ascontiguousarray(gt_adj_flat)
    savar_crl_flat = np.ascontiguousarray(savar_crl_flat)
    savar_cd_flat = np.ascontiguousarray(savar_cd_flat)

    print("Type cd graph: ", type(savar_cd_flat))
    print("Type crl graph: ", type(savar_crl_flat))
    print("Type gt_adj_flat: ", type(gt_adj_flat))


    # -- 4. Calculate metrics --

    # calculate accuracy of causal discovery vs cdsd (picabu) for dataset
    cd_true_positives = int(np.array_equal(savar_cd_flat, gt_adj_flat))
    crl_true_positives = int(np.array_equal(savar_crl_flat, gt_adj_flat))
    print("Did CD recover the full ground truth graph? ", cd_true_positives)
    print("Did CRL recover the full ground truth graph? ", crl_true_positives)

    # print("\nStructural & causal metrics on CRL...")

    crl_parent_aid = parent_aid(gt_adj_flat, savar_crl_flat, edge_direction="from row to column")
    crl_oset_aid = oset_aid(gt_adj_flat, savar_crl_flat, edge_direction="from row to column")
    crl_ancestor_aid = ancestor_aid(gt_adj_flat, savar_crl_flat, edge_direction="from row to column")
    crl_sid = sid(gt_adj_flat, savar_crl_flat, edge_direction="from row to column")
    crl_shd = shd(gt_adj_flat, savar_crl_flat)
    crl_f1 = f1_score(savar_crl_flat, gt_adj_flat)
    crl_precision, crl_recall = precision_recall(savar_crl_flat, gt_adj_flat)

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
    cd_parent_aid = parent_aid(gt_adj_flat, savar_cd_flat, edge_direction="from row to column")
    cd_oset_aid = oset_aid(gt_adj_flat, savar_cd_flat, edge_direction="from row to column")
    cd_ancestor_aid = ancestor_aid(gt_adj_flat, savar_cd_flat, edge_direction="from row to column")
    cd_sid = sid(gt_adj_flat, savar_cd_flat, edge_direction="from row to column")
    cd_shd = shd(gt_adj_flat, savar_cd_flat)
    cd_f1 = f1_score(savar_cd_flat, gt_adj_flat)
    cd_precision, cd_recall = precision_recall(savar_cd_flat, gt_adj_flat)
    # proportion of False positive / False negative

    print("cd_parent_aid", cd_parent_aid)
    print("cd_oset_aid", cd_oset_aid)
    print("cd_ancestor_aid", cd_ancestor_aid)
    print("cd_sid", cd_sid)
    print("cd_shd", cd_shd)
    print("cd_f1", cd_f1)
    print("cd_precision", cd_precision)
    print("cd_recall", cd_recall)

    print("==============================")

    return {
        "cd_true_positives": cd_true_positives,
        "cd_parent_aid": cd_parent_aid,
        "cd_oset_aid": cd_oset_aid,
        "cd_ancestor_aid": cd_ancestor_aid,
        "cd_sid": cd_sid,
        "cd_shd": cd_shd,
        "cd_f1": cd_f1,
        "cd_precision": cd_precision,
        "cd_recall": cd_recall,        
        "crl_true_positives": crl_true_positives,
        "crl_parent_aid": crl_parent_aid,
        "crl_oset_aid": crl_oset_aid,
        "crl_ancestor_aid": crl_ancestor_aid,
        "crl_sid": crl_sid,
        "crl_shd": crl_shd,
        "crl_f1": crl_f1,
        "crl_precision": crl_precision,
        "crl_recall": crl_recall,
    }


# ---- CMD LINE ARGS
results_pkl_path = OUTPUTS_DIR / f"evaluation_savar_gt_final.pkl"


seed = 1
dataset = "savar"
model = "savar"
output_dict = {}

for num_modes in [4, 16]:
    for difficulty in ["easy", "med_easy", "med_hard", "hard"]:
        try:
            print(f"Evaluating dataset: {dataset}-modes_{num_modes}-diff_{difficulty}-seed_{seed}")
            if num_modes == 4 and difficulty == "med_easy":
                print("Using non-linear model")
                output_dict[f"{model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}"] = eval_savar(dataset,num_modes, difficulty, seed, linear=False)
            elif num_modes == 4 and difficulty == "med_hard":
                print("Using non-linear model")
                output_dict[f"{model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}"] = eval_savar(dataset,num_modes, difficulty, seed, linear=False)
            else:
                print("Using linear model")
                output_dict[f"{model}-modes_{num_modes}-diff_{difficulty}-seed_{seed}"] = eval_savar(dataset,num_modes, difficulty, seed, linear=True)
        except Exception as e:
            print(f"====== ERROR evaluating dataset: {dataset}-modes_{num_modes}-diff_{difficulty}-seed_{seed}")
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
