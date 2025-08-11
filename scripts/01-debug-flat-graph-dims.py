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

            try:
                # flatten graph
                print(f"Reshaping flat graph for experiment: {experiment_name}")
                flattened_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", experiment_name=experiment_name)
                # binarize graph
                binary_graph = binarize_array(flattened_graph)
                np.savez(
                    f"{OUTPUTS_DIR}/{experiment_name}-flat_graph_binary-cd.npz",
                    graph=binary_graph,
                )
                print("flattened_graph.shape: ", flattened_graph.shape)
                print("flattened_graph: ", flattened_graph)
            except FileNotFoundError:
                print(f"File not found for experiment: {experiment_name}")
                continue