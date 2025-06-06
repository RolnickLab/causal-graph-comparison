import networkx as nx
import matplotlib.pyplot as plt

from cgc.graph_builder import GraphBuilder
from cgc.graph_modifier import GraphModifier

# --- INSTRUCTIONS: uncomment the plt.show() below to display graphs ---
# --- I didn't want to generate an unreadable plot with a dozen subplots
# #TODO: wrap in class/function to generate dataset


# --- 1. Build ground truth graph with graph builder ---
data_path = "../data"
chosen_file = 0  # TODO: make this dynamic, e.g. via command line argument, eventually loop through all files
gb = GraphBuilder(data_path=data_path, file_num=chosen_file)

# -- 2.  Visualize the flattened ground truth temporal graph  --
plt.subplot(1, 2, 1)
gb.plot_graph()

# ------- MODIFICATIONS TO THE GROUND TRUTH GRAPH -------

# -- 3. WEIGHTS: Randomly increase/decrease weights ---
gm_weights_rnd = GraphModifier(gb)
gm_weights_rnd.randomize_weights(num_changes=3)
plt.subplot(1, 2, 2)
gm_weights_rnd.gb.plot_graph()
plt.show()

# -- 4. WEIGHTS: Uniformly shift weights of all edges  ---
# gm_weights_shifted = GraphModifier(gb)
# gm_weights_shifted.shift_weights(shift=0.5)
# plt.subplot(1, 2, 2)
# gm_weights_shifted.gb.plot_graph()
# plt.show()

# -- 5. WEIGHTS: Scale weights of all edges by factor of x  ---
# gm_weights_scaled = GraphModifier(gb)
# gm_weights_scaled.shift_weights(scale=2)
# plt.subplot(1, 2, 2)
# gm_weights_scaled.gb.plot_graph()
# plt.show()

# -- 6. TIME LAG: randomly modify time lag of x nodes  --
# gm_random_lag = GraphModifier(gb)
# gm_random_lag.randomly_modify_lag(num_changes=2)
# plt.subplot(1, 2, 2)
# gm_random_lag.gb.plot_graph()
# plt.show()

# -- 7. TIME LAG: shift all nodes by time lag --
# gm_lag_shifted = GraphModifier(gb)
# gm_lag_shifted.shift_all_nodes_lag(shift_value=1)
# plt.subplot(1, 2, 2)
# gm_lag_shifted.gb.plot_graph()
# plt.show()

# -- 8. REMOVE NODES + edge: Remove either specific nodes or random number of nodes --
# gm_remove_nodes = GraphModifier(gb)
# gm_remove_nodes.delete_nodes(to_delete=2, avoid_t0=True)  # remove 2 random nodes & don't delete any T0 nodes
# # gm_remove_nodes.delete_nodes(to_delete=["N2, T-4", "N3, T-3"])  # OPTION 2: remove specific nodes
# plt.subplot(1, 2, 2)
# gm_remove_nodes.gb.plot_graph()
# plt.show()

# -- 9. REMOVE EDGES: Remove either specific edges or random number of edges --
# gm_remove_edges = GraphModifier(gb)
# gm_remove_edges.delete_edges(to_delete=2)  # remove 2 random edges
# # gm_remove_edges.delete_edges(to_delete=[("N0, T-2", "N0, T0"), ("N3, T-5", "N0, T0"), ("N2, T-4", "N0, T0")]) # OPTION 2: remove specific edges
# plt.subplot(1, 2, 2)
# gm_remove_edges.gb.plot_graph()
# plt.show()

# -- 10. ADD EDGE: Randomly add x edges from nodes that already exist to T0 nodes
# gm_insert_edges = GraphModifier(gb)
# gm_insert_edges.insert_edges(to_add=2)  # add 2 random edges
# plt.subplot(1, 2, 2)
# gm_insert_edges.gb.plot_graph()
# plt.show()

# -- 11. ADD NODES: Randomly add x nodes to the graph from nodes that
# ---- don't already exist in the graph (but are within the time range of the graph) ---
# gm_insert_nodes = GraphModifier(gb)
# gm_insert_nodes.insert_nodes(to_add=3)  # add 2 random nodes
# plt.subplot(1, 2, 2)
# gm_insert_nodes.gb.plot_graph()
# plt.show()

# ------ OTHER OPERATIONS ---------

# -- 12. Extract equations from gt & adjacency matrices
# equations = gb.extract_latent_equations()
# equations_from_adj = gb.extract_equations_from_adjacency()
# print("Latent Equations:\n", equations)
# print("Equations from Adjacency Matrices:\n", equations_from_adj)

# -- 13. Check if graph is acyclic ---
# print("Is the graph acyclic?", nx.is_directed_acyclic_graph(gb.G))  # check if G is acyclic

# -- 14. Convert G to wighted dense adjacency matrix using networkx function
# nx_adjacency_matrix_sparse = nx.adjacency_matrix(gb.G)  # sparse
# nx_adjacency_matrix_dense = nx_adjacency_matrix_sparse.todense()  # dense
# print("NetworkX Adjacency Matrix (Sparse):\n", nx_adjacency_matrix_sparse)
# print("NetworkX Adjacency Matrix (Dense):\n", nx_adjacency_matrix_dense)
# print("Shape of the dense adjacency matrix:", nx_adjacency_matrix_dense.shape)
