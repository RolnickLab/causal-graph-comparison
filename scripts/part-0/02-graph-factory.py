from causal_graph_comparison.part_0 import SyntheticGraphFactory, GraphModifier
import numpy as np
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

# print("Modifying the lag of 1 random node in the graph:")
# new_graph, _ = modifier.apply(graph, operation="randomly_modify_lag", k=1, seed=0)
# print("Modified graph:")
# print(new_graph)

# print("Difference between original and modified graph:")
# print(graph - new_graph)


# =======
# Next steps:
# 1. binarize graph
# 2. generate many types of graphs (difficulty, modes) + modifications for each one
# 3. flatten graphs
# 4. apply comparison metrics  