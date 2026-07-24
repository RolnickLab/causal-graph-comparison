import numpy as np
from causal_graph_comparison.part_0 import SyntheticGraphFactory

n_nodes = [4, 16 ,64]

graph_seed = 20

probabilities = {}
expected_edges = {}
for n in n_nodes:
    denominator = 2
    if n <= 4:
        denominator = 1
    expected_edges[n] = {
        "easy": n + n * (n-1) * 0,
        "med_easy": n + n * (n - 1) * (1 / (n - 1)),
        "med_hard": n + n * (n - 1) * (2 / (n - 1)),
        "hard": n + n * (n - 1) * (1 / denominator)
    }
