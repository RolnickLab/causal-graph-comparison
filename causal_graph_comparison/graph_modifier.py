import numpy as np
import networkx as nx
import copy
from random import randint, random, sample
from causal_graph_comparison.graph_builder import GraphBuilder
from causal_graph_comparison.utils import get_timelag
from typing import Union, List, Tuple

# TODO: convert graph to binary


class GraphModifier:
    """
    A class to modify the graph G built by GraphBuilder.
    Modifications include:
    - Randomly modifying edge weights
    - Shifting all edge weights by a multiplicative and/or additive factor
    - Randomly modifying time lag of x nodes
    - Shifting all nodes' time lags by given value
    - Deleting nodes (specified or random)
    - Deleting edges (specified or random)
    - Inserting random edges
    - Inserting random nodes
    """

    def __init__(self, gb: GraphBuilder) -> None:
        """Initialize the GraphModifier with a copy of the GraphBuilder instance.

        Args:
            gb: The GraphBuilder instance to modify.
        """
        self.gb: GraphBuilder = copy.deepcopy(gb)

    def randomize_weights(self, num_changes: int = 1, normal_std: float = 0.1) -> None:
        """Modify the weights of x edges in graph G by random value.

        Args:
            num_changes: The number of weights to modify.
            normal_std: The standard deviation for the normal distribution used to generate new weights.
        """
        # print("List of edges in the graph:", self.gb.list_edges)

        num_edges = len(self.gb.list_edges)

        if num_changes > num_edges:
            raise ValueError("Number of changes exceeds the number of edges in the graph.")

        if num_changes < 1:
            raise ValueError("Number of changes must be at least 1.")

        # Create a list of available edge indices
        available_edges = list(range(num_edges))

        for _ in range(num_changes):
            # Randomly select an edge from available edges
            change_idx = np.random.choice(available_edges)
            # Remove the selected edge from available edges
            available_edges.remove(change_idx)

            edge = self.gb.list_edges[change_idx]

            # get weight of selected edge
            current_weight = self.gb.graph[edge[0]][edge[1]]["weight"]
            print(f"Current weight of edge {edge} is {current_weight}")

            # Try to generate a significant weight using normal distribution
            max_tries = 5
            new_weight = None
            for _ in range(max_tries):
                potential_weight = np.random.normal(0, normal_std)
                if np.abs(potential_weight) > 0.01:  # Ensure the new weight is significant
                    new_weight = potential_weight

            # If normal distribution didn't yield a significant weight, use uniform distribution
            if not new_weight:
                # naively generate a weight between 0.01 and 0.3
                sign = np.random.choice([-1, 1])
                new_weight = sign * np.random.uniform(0.01, 0.3)

            # Apply the new weight to the edge
            final_weight = np.round(np.clip(new_weight + current_weight, 0.01, 1), 2)
            self.gb.graph[edge[0]][edge[1]]["weight"] = final_weight
            print("New weight of edge", edge, "is", final_weight, "\n")

        self.gb.title_suffix = f"Randomly modified {num_changes} edge weights"

    def shift_weights(self, shift: float = 0, scale: float = 1) -> None:
        """Shift the weights of the edges in graph G by an additive and/or multiplicative factor.

        Args:
            shift: Value to add to each edge weight.
            scale: Factor to multiply each edge weight by.
        """
        if shift == 0 and scale == 1:
            raise ValueError("Adding 0 and scaling by 1. No modification applied.")

        for _, _, params in self.gb.graph.edges(data=True):
            old_weight = params["weight"]
            new_weight = old_weight * scale  # multiply
            new_weight += shift  # add
            params["weight"] = np.clip(
                np.round(new_weight, 2), 0.01, 1
            )  # Round to two decimal places, clip to [0.01, 1]

        self.gb.title_suffix = f"Shifted all edge weights by {shift} and scaled by {scale}"

    def modify_lag(self, node: str, new_node: str) -> None:
        """Modify the lag of a specific node in the graph by replacing it with the same node value at a different time lag.

        Args:
            node: The node whose lag will be modified.
            new_node: The new node with the modified lag.
        """
        # Update the node's lag
        if node in self.gb.graph.nodes():
            # replace node with new_node:
            nx.relabel_nodes(self.gb.graph, {node: new_node}, copy=False)
            # Update the used nodes set
            self.gb.used_nodes.discard(node)  # Remove the old node
            self.gb.used_nodes.add(new_node)  # Add the new node
        else:
            raise ValueError(f"Node {node} does not exist in the graph.")

    def randomly_modify_lag(self, num_changes: int = 1) -> None:
        """Randomly modify the lag of x parent nodes in the graph.

        Args:
            num_changes: The number of nodes to modify.
        """
        # Get all eligible nodes (non-T0 nodes that haven't been changed)
        eligible_nodes = [node for node in self.gb.used_nodes if "T0" not in node]

        if num_changes > len(eligible_nodes):
            raise ValueError("Number of changes exceeds the number of eligible nodes (non-T0 nodes) in the graph.")

        # Randomly select nodes to modify
        nodes_to_modify = np.random.choice(eligible_nodes, size=num_changes, replace=False)

        for node in nodes_to_modify:
            print("Modifying lag for node:", node)

            # Get current lag
            node_name, time_lag = node.split(", ")
            current_lag = int(time_lag[1:])

            # Create list of available lags (excluding current lag)
            available_lags = [lag for lag in range(-self.gb.max_lag, 0) if lag != current_lag]

            # Randomly select a new lag from available lags
            new_lag = np.random.choice(available_lags)
            new_node = f"{node_name}, T{new_lag}"
            self.modify_lag(node, new_node)

        self.gb.title_suffix = f"Randomly modified {num_changes} nodes' lags"

    def _get_sorted_nodes(self, shift_value: int) -> List[str]:
        """Get nodes sorted based on shift direction.

        Args:
            shift_value: The value by which to shift the lags.

        Returns:
            List of nodes sorted in appropriate order.
        """
        reverse = shift_value < 0
        return sorted(self.gb.used_nodes, key=lambda node: get_timelag(node), reverse=reverse)

    def _should_shift_node(self, time_lag: int) -> bool:
        """Check if a node should be shifted based on its time lag.

        Args:
            time_lag: The current time lag of the node.

        Returns:
            True if the node should be shifted, False otherwise.
        """
        return time_lag < 0

    def _calculate_new_lag(self, current_lag: int, shift_value: int) -> int:
        """Calculate the new lag value after shifting.

        Args:
            current_lag: The current time lag.
            shift_value: The value by which to shift.

        Returns:
            The new lag value.
        """
        new_lag = current_lag + shift_value
        return np.clip(new_lag, -self.gb.max_lag, -1)

    def shift_all_nodes_lag(self, shift_value: int = 1) -> None:
        """Shift all nodes' lags by a specified value.

        Args:
            shift_value: The value by which to shift the lags.
        """
        sorted_nodes = self._get_sorted_nodes(shift_value)
        new_used_nodes = set()

        for node in sorted_nodes:
            node_name, _ = node.split(", ")
            time_lag = get_timelag(node)

            # don't shift nodes with T0 time lag
            if not self._should_shift_node(time_lag):
                new_used_nodes.add(node)
                continue

            new_lag = self._calculate_new_lag(time_lag, shift_value)

            if new_lag == time_lag:
                new_used_nodes.add(node)
                continue

            new_node = f"{node_name}, T{new_lag}"
            self.modify_lag(node, new_node)
            new_used_nodes.add(new_node)

        self.gb.rebuild_dense_graph()
        self.gb.title_suffix = f"Shifted all nodes' lags by {shift_value}"

    def delete_empty_nodes(self) -> None:
        """Remove nodes from graph G that have no incoming or outgoing edges."""
        empty_nodes = [
            node
            for node in self.gb.used_nodes
            if self.gb.graph.out_degree(node) == 0 and self.gb.graph.in_degree(node) == 0
        ]

        for node in empty_nodes:
            self.gb.graph.remove_node(node)
            self.gb.used_nodes.discard(node)
            print(f"Removed empty node {node}")

    def delete_nodes(self, to_delete: Union[int, List[str]], avoid_t0: bool = False) -> None:
        """Remove a node from the graph G.

        Args:
            to_delete: The number of nodes to randomly delete or a list of nodes to specifically delete.
            avoid_t0: If True, avoid deleting nodes with T0 time lag.
        """

        eligible_nodes = self.gb.used_nodes.copy()

        if avoid_t0:
            # Filter out nodes with T0 time lag
            eligible_nodes = {node for node in self.gb.used_nodes if "T0" not in node}

        # Option 1: Delete "to_delete" number of nodes randomly
        if isinstance(to_delete, int):
            if to_delete < 1:
                raise ValueError("Number of nodes to delete must be at least 1.")
            if to_delete > len(eligible_nodes):
                raise ValueError("Number of nodes to delete exceeds the number of used nodes in the graph.")
            # Randomly select nodes to delete
            nodes_to_delete = np.random.choice(list(eligible_nodes), size=to_delete, replace=False)
            self.gb.title_suffix = f"Randomly deleted {to_delete} nodes"

        # Option 2: Delete specific nodes from a list
        elif isinstance(to_delete, list):
            nodes_to_delete = to_delete
            if not all(node in eligible_nodes for node in nodes_to_delete):
                raise ValueError("Some nodes to delete do not exist in the graph.")
            self.gb.title_suffix = f"Deleted nodes: {'& '.join(nodes_to_delete)}"

        else:
            raise ValueError("to_delete must be an integer or a list of nodes.")

        for node in nodes_to_delete:
            self.gb.graph.remove_node(node)
            self.gb.used_nodes.discard(node)
            print(f"Removed node {node}")

        # Make sure to delete empty nodes
        self.delete_empty_nodes()

    def delete_edges(self, to_delete: Union[int, List[Tuple[str, str]]]) -> None:
        """Remove specified number of edges from the graph G.

        Args:
            to_delete: The number of edges to randomly delete or a list of edges to specifically delete.
        """
        # Option 1: Delete "to_delete" number of edges randomly
        if isinstance(to_delete, int):

            if to_delete < 1:
                raise ValueError("Number of edges to delete must be at least 1.")
            elif to_delete > len(self.gb.graph.edges()):
                raise ValueError("Number of edges to delete exceeds the number of edges in the graph.")

            # Randomly select to_delete number of edges to delete
            indices_to_remove = np.random.choice(range(len(self.gb.list_edges)), size=to_delete, replace=False)
            edges_to_remove = [self.gb.list_edges[i] for i in indices_to_remove]

        # Option 2: Delete specific edges from a list
        elif isinstance(to_delete, list):
            edges_to_remove = to_delete

            # Check if all edges to delete exist in the graph
            if not all(self.gb.graph.has_edge(*edge) for edge in edges_to_remove):
                raise ValueError("Some edges to delete do not exist in the graph.")

            self.gb.title_suffix = f"Deleted edges: {'& '.join(map(str, edges_to_remove))}"

        else:
            raise ValueError("to_delete must be an integer or a list of edges.")

        print("Edges to remove: ", edges_to_remove)

        self.gb.graph.remove_edges_from(edges_to_remove)

        # make sure to delete empty nodes
        self.delete_empty_nodes()

    def create_new_weight(self) -> float:
        """Create a new weight for an edge in the graph.

        Returns:
            New random weight for an edge in the graph based on the mean and std of the existing edge weights.
        """
        edge_weights = list(self.gb.graph.edges(data="weight"))  # list (source, target, weight)
        mean_edge_weight = np.mean([weight for _, _, weight in edge_weights])
        stdv_edge_weight = np.std([weight for _, _, weight in edge_weights])

        weight = np.random.normal(mean_edge_weight, stdv_edge_weight)
        # Clip the weight to be within [0.01, 1.0] and round to 2 decimal places
        weight = np.round(np.clip(weight, 0.01, 1.0), 2)

        return weight

    def insert_edges(self, to_add: int) -> None:
        """Randomly add specified number of edges to the graph.

        Args:
            to_add: The number of edges to add.
        """
        print("Adding edges to the graph...")
        # Get all existing nodes in the graph

        # Get T0 nodes (target nodes) and non-T0 nodes (source nodes)
        target_nodes = [node for node in self.gb.used_nodes if "T0" in node]
        source_nodes = [node for node in self.gb.used_nodes if "T0" not in node]

        # Create list of all possible edges that don't exist yet
        available_edges = [
            (source, target)
            for source in source_nodes
            for target in target_nodes
            if not self.gb.graph.has_edge(source, target)
        ]

        if to_add < 1:
            raise ValueError("Must add at least one edge.")
        if to_add > len(available_edges):
            raise ValueError(
                "Number of edges to add cannot exceed the number of possible edges remaining in the graph."
            )

        # Randomly select edges to add
        edges_to_add = sample(available_edges, k=to_add)

        for source_node, target_node in edges_to_add:
            weight = self.create_new_weight()
            self.gb.graph.add_edge(source_node, target_node, weight=weight)
            print(f"Added edge from {source_node} to {target_node} with weight {weight}")

        self.gb.title_suffix = f"Inserted {to_add} random edges"

    def insert_nodes(self, to_add: int) -> None:
        """Randomly add a new node in graph G within the time range of the graph (edge to node in T0).

        Args:
            to_add: The number of nodes to add.
        """
        target_nodes = [node for node in self.gb.graph.nodes() if "T0" in node]  # all T0 nodes

        # all nodes not in use or T0
        possible_source_nodes = list(self.gb.graph.nodes() - target_nodes - self.gb.used_nodes)

        # print("Possible source nodes:", possible_source_nodes)
        # error checks
        if to_add < 1:
            raise ValueError("Must add at least one node.")
        if to_add > len(possible_source_nodes):
            raise ValueError("Number of nodes to add exceeds the number of nodes in the graph.")

        for _ in range(to_add):
            # randomly select a source & target node from the possible nodes
            weight = self.create_new_weight()
            source_node = np.random.choice(possible_source_nodes)
            target_node = np.random.choice(target_nodes)

            self.gb.graph.add_edge(source_node, target_node, weight=float(weight))  # add the new node to the graph
            self.gb.used_nodes.add(source_node)  # add source node to used nodes

            # remove node from possible source nodes so it's not reused
            possible_source_nodes.remove(source_node)
            print(f"Added node {source_node} connected to {target_node} with weight {weight}")

        self.gb.title_suffix = f"Randomly added {to_add} new nodes"
