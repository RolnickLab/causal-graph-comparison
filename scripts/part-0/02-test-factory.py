from causal_graph_comparison.part_0 import SyntheticGraphFactory, GraphModifier, score_pair
from causal_graph_comparison.graph_utils import binarize_array

# # one graph example
factory = SyntheticGraphFactory(n_nodes=4, difficulty="hard", max_time_steps=5)

gt = factory.generate(seed_graph=0)
print("Ground truth graph:")
print(gt)

# binarize graph
binarized_gt = binarize_array(gt)

modifier = GraphModifier()

# Inserting 1 node to the graph:

# new_graph, new_gt = modifier.apply(binarized_gt, operation="insert_nodes", k=1, seed=42)
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

print("Deleting 1 node from the graph:")
new_graph, _ = modifier.apply(binarized_gt, operation="delete_nodes", k=1, seed=0)
print("Modified graph:")
print(new_graph)

print("Difference between original and modified graph:")
print(binarized_gt - new_graph)

# =====

# print("Modifying the lag of 1 node in the graph:")
# new_graph, _ = modifier.apply(graph, operation="modify_lag", k=1, seed=0, mod_val=-1)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(graph - new_graph)

# =====

# print("Modifying the lag of 1 random node in the graph:")
# new_graph, _ = modifier.apply(binarized_gt, operation="randomly_modify_lag", k=1, seed=0)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(binarized_gt - new_graph)


# ======= scoring example

f1, shd_score, sid_score, parent_aid_score, oset_aid_score = score_pair(binarized_gt, new_graph)
print(f"F1 score: {f1}")
print(f"SHD score: {shd_score}")
print(f"SID score: {sid_score}")
print(f"Parent AID score: {parent_aid_score}")
print(f"OSet AID score: {oset_aid_score}")


# TODO: confirm flattening works as expected?