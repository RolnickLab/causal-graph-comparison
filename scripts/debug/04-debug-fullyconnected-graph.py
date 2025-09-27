from causal_graph_comparison import *
import numpy as np

graph_path = OUTPUTS_DIR / "vae-modes_4-diff_med_hard-seed_1-nonlinear-nonlinear-flat_graph-binary-crl.npz"
graph = np.load(graph_path)["graph"]
print("graph.shape", graph.shape)
print("graph \n", graph)





