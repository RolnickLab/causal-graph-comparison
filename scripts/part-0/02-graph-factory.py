import numpy as np
from gadjid import parent_aid, shd, sid

from causal_graph_comparison.part_0 import SyntheticGraphFactory, GraphModifier
from climatem.model.metrics import f1_score
from causal_graph_comparison.graph_utils import binarize_array, flatten_temporal_adjacency_graph2


rng = np.random.default_rng(42)

factory = SyntheticGraphFactory(n_nodes=4, difficulty="med_easy", max_time_steps=5)

graph = factory.generate(seed_graph=0)
print("Original graph:")
print(graph)

modifier = GraphModifier()
# Inserting 1 node to the graph:
# new_graph, new_gt = modifier.apply(graph, operation="insert_nodes", k=1, seed=0)
# print("Modified graph:")
# print(new_graph)
# print("Modified ground truth graph:")
# print(new_gt)

# print("Difference between original and modified graph:")
# print(new_gt - new_graph)

# =======
# print("Deleting 1 edge from the graph:")
# new_graph, _ = modifier.apply(graph, operation="delete_edges", k=1, seed=0)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(graph - new_graph)

# ==========

# print("Inserting 1 edge to the graph:")
# new_graph, _ = modifier.apply(graph, operation="insert_edges", k=1, seed=0)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(graph - new_graph)

# =====

# print("Deleting 1 node from the graph:")
# new_graph, _ = modifier.apply(graph, operation="delete_nodes", k=1, seed=0)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(graph - new_graph)

# =====

# print("Modifying the lag of 1 node in the graph:")
# new_graph, _ = modifier.apply(graph, operation="modify_lag", k=1, seed=0, mod_val=-1)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(graph - new_graph)

# =====

# binarize graph
binarized_gt = binarize_array(graph)

print("Modifying the lag of 1 random node in the graph:")
new_graph, _ = modifier.apply(binarized_gt, operation="randomly_modify_lag", k=1, seed=0)
print("Modified graph:")
print(new_graph)

print("Difference between original and modified graph:")
print(binarized_gt - new_graph)

# flatten graph
flat_graph = flatten_temporal_adjacency_graph2(shape="time_child_parent", graph=new_graph)
flat_gt = flatten_temporal_adjacency_graph2(shape="time_child_parent", graph=binarized_gt)

# # gadjid requires contiguous int8 adjacency matrices
# flat_graph = np.ascontiguousarray(flat_graph.astype(np.int8))
# flat_gt = np.ascontiguousarray(flat_gt.astype(np.int8))

# apply causal metrics
f1 = f1_score(flat_gt, flat_graph)
print(f"F1 score: {f1}")

shd_score = shd(flat_gt, flat_graph)
print(f"SHD: {shd_score}")

aid_score = parent_aid(flat_gt, flat_graph, edge_direction="from row to column")
print(f"Parent AID: {aid_score}")

sid_score = sid(flat_gt, flat_graph, edge_direction="from row to column")
print(f"SID: {sid_score}")


# =======
# Next steps:
# 1. generate many types of graphs (difficulty, modes) + modifications for each one