from causal_graph_comparison import *
import numpy as np

mlp_graph_path = OUTPUTS_DIR / "vae-modes_4-diff_med_easy-seed_1-nonlinear-picabu_cdsd.npz"
mlp_graph = np.load(mlp_graph_path)["val_matrix"]
print("mlp_graph.shape", mlp_graph.shape)
print("mlp_graph \n", mlp_graph)





