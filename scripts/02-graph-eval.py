import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import gadjid
from gadjid import example, ancestor_aid, oset_aid, parent_aid, shd, sid

import causal_graph_comparison
from causal_graph_comparison.graph_builder import GraphBuilder
from causal_graph_comparison.graph_modifier import GraphModifier
from causal_graph_comparison.utils import binarize_array

# --- 1. Build ground truth graph with graph builder ---

data_path = causal_graph_comparison.DATA_DIR
chosen_file = 0  # TODO: make this dynamic, e.g. via command line argument, eventually loop through all files
gb = GraphBuilder(data_path=data_path, file_num=chosen_file)

# -- 2. TIME LAG: shift all nodes by time lag --
gm_lag_shifted = GraphModifier(gb)
gm_lag_shifted.shift_all_nodes_lag(shift_value=1)

# -- 3. Convert wighted dense adjacency matrix using networkx function
gt_adjacency_matrix_sparse = nx.adjacency_matrix(gb.graph)  # sparse
Gtrue = gt_adjacency_matrix_sparse.todense()  # dense
# print("Gtrue weighted:\n", Gtrue)

Gtrue = binarize_array(Gtrue)
print("Gtrue:\n", Gtrue)

mod_adjacency_matrix_sparse = nx.adjacency_matrix(gm_lag_shifted.gb.graph)  # sparse
Gguess = mod_adjacency_matrix_sparse.todense()  # dense
# print("Gguess weighted:\n", Gguess)

Gguess = binarize_array(Gguess)
print("Gguess:\n", Gguess)

# -- 4. Calculate metrics --
# TODO: use gadjid to calculate metrics
print(ancestor_aid(Gtrue, Gguess, edge_direction="from row to column"))
print(sid(Gtrue, Gguess, edge_direction="from row to column"))
print(shd(Gtrue, Gguess))
