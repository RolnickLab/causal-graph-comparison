from causal_graph_comparison import *
import numpy as np
from climatem.synthetic_data.graph_evaluation_ilija import *
from pathlib import Path

model = "mlp" # "mlp", "lstm", "cnn", "vae"
gt = "savar"
difficulty = "easy" # "easy", "med_easy", "med_hard", "hard"
modes = 4 # 4, 16, 64
seed = 1

data_dir = Path("/Users/cisaicu/dev/mila/scratch/data/SAVAR_DATA_TEST")

experiment_name = f"modes_{modes}-diff_{difficulty}-seed_{seed}"
model_name = f"{model}-{experiment_name}"
gt_name = f"{gt}-{experiment_name}"

data_path = data_dir / f"{experiment_name}_parameters.npy"

cd_path = OUTPUTS_DIR / f"{model_name}-pcmci_causal_discovery.npz"
crl_path = OUTPUTS_DIR / f"{model_name}-picabu_cdsd.npz"

cd_savar_path = OUTPUTS_DIR / f"{gt_name}-pcmci_causal_discovery.npz"
crl_savar_path = OUTPUTS_DIR / f"{gt_name}-picabu_cdsd.npz"

cd_graph = np.load(cd_path)["val_matrix"][:,:,1:]
crl_graph = np.load(crl_path)["val_matrix"]
cd_graph_savar = np.load(cd_savar_path)["val_matrix"][:,:,1:]
crl_graph_savar = np.load(crl_savar_path)["val_matrix"]

links_coeffs = np.load(data_path, allow_pickle=True).item()["links_coeffs"]

gt_graph = extract_adjacency_matrix(links_coeffs, modes, tau=5)

print("cd graph shape: ", cd_graph.shape)
print("crl graph shape: ", crl_graph.shape)
print("cd graph savar shape: ", cd_graph_savar.shape)
print("crl graph savar shape: ", crl_graph_savar.shape)
print("gt graph shape: ", gt_graph.shape)

# Parse and print causal relationships
causal_dict = {"0": [[[0, -2], 0.5]], "1": [[[1, -3], 0.38]], "2": [[[2, -2], 0.36]], "3": [[[3, -4], 0.38]]}

for child_node, relationships in causal_dict.items():
    for relationship in relationships:
        parent_node = relationship[0][0]
        time_lag = abs(relationship[0][1])  # Convert negative lag to positive
        weight = relationship[1]
        print(f"{parent_node} --> {child_node}, lag {time_lag}, weight {weight}")

print("gt graph: ", gt_graph)
print("--------------------------------")
print("causal discovery graph: ", cd_graph)
print("--------------------------------")
print("causal discovery graph savar: ", cd_graph_savar)
print("--------------------------------")
print("picabu graph: ", crl_graph)
print("--------------------------------")
print("picabu graph savar: ", crl_graph_savar)
print("--------------------------------")






