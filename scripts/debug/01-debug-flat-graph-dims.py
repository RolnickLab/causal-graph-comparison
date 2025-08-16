import numpy as np
from causal_graph_comparison import OUTPUTS_DIR
from causal_graph_comparison.graph_utils import binarize_array, flatten_temporal_adjacency_graph


models = ["mlp", "savar", "vae", "lstm", "cnn"]
modes = [4, 16, 64]
seed = 1
difficulties = ["easy", "med_easy", "med_hard", "hard"]


for model in models:
    for mode in modes:
        for difficulty in difficulties:
            experiment_name = f"{model}-modes_{mode}-diff_{difficulty}-seed_{seed}"
            print(f"Experiment name: {experiment_name}")

            try:
                cd_graph = np.load(f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-cd.npz")["graph"]
                crl_graph = np.load(f"{OUTPUTS_DIR}/{experiment_name}-flat_graph-binary-crl.npz")["graph"]
                print(f"cd_graph.shape: {cd_graph.shape}")
                print(f"crl_graph.shape: {crl_graph.shape}")
            except FileNotFoundError:
                print(f"File not found for experiment: {experiment_name}")
                continue