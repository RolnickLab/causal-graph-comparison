import numpy as np
from causal_graph_comparison import *


models = ["mlp", "lstm", "cnn", "vae", "savar"]
modes = [4, 16, 64]
difficulties = ["med_hard", "med_easy", "easy", "hard"]
seed = 1

for model in models:
    for mode in modes:
        for difficulty in difficulties:
                try:
                    experiment_name = f"{model}-modes_{mode}-diff_{difficulty}-seed_1"
                    print(f"Reversing CRL graph for experiment: {experiment_name}")
                    temporal_crl_graph_path = OUTPUTS_DIR / f"{experiment_name}-picabu_cdsd.npz"
                    temporal_crl_graph = np.load(temporal_crl_graph_path)["val_matrix"]
                    temporal_crl_graph = np.flip(temporal_crl_graph, axis=0)
                    np.savez(temporal_crl_graph_path, val_matrix=temporal_crl_graph)
                except Exception as e:
                    print(f"Error reversing CRL graph for experiment: {experiment_name}")
                    print(e)
                    continue