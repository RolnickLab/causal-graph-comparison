import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import gadjid
from gadjid import example, ancestor_aid, oset_aid, parent_aid, shd, sid
from causal_graph_comparison import OUTPUTS_DIR

import causal_graph_comparison
from causal_graph_comparison.graph_builder import GraphBuilder
from causal_graph_comparison.graph_modifier import GraphModifier
from causal_graph_comparison.utils import binarize_array
import numpy as np

# --- 1. Build ground truth graph with graph builder ---
true_file = OUTPUTS_DIR / "g_true.csv"
guess_file = OUTPUTS_DIR / "g_guess.csv"

adj_true = np.loadtxt(true_file, delimiter=',')
adj_guess = np.loadtxt(guess_file, delimiter=',')

Gguess = binarize_array(adj_guess)
Gtrue = binarize_array(adj_true)

print("Gtrue: ", Gtrue)

# -- 4. Calculate metrics --
# TODO: use gadjid to calculate metrics
print(parent_aid(Gtrue, Gguess, edge_direction="from row to column"))
print(sid(Gtrue, Gguess, edge_direction="from row to column"))
print(shd(Gtrue, Gguess))
