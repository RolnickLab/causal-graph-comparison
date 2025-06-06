import numpy as np
import networkx as nx
import copy
from random import randint
from cgc.graph_builder import GraphBuilder
from cgc.utils import get_timelag


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

    def __init__(self, gb: GraphBuilder):
        self.gb: GraphBuilder = copy.deepcopy(gb)

    def randomize_weights(self, num_changes=1, normal_std=0.1):
        """
        Modify the weights of x edges in graph G by random value.

        Args:
            num_changes (int): The number of weights to modify.
            normal_std (float): The standard deviation for the normal distribution used to generate new weights.
        """
        # print("List of edges in the graph:", self.gb.list_edges)

        num_edges = len(self.gb.list_edges)
        change_idx = randint(0, num_edges - 1)

        changed_edges = []

        if num_changes > num_edges:
            raise ValueError("Number of changes exceeds the number of edges in the graph.")

        if num_changes < 1:
            raise ValueError("Number of changes must be at least 1.")

        for _ in range(num_changes):

            # Ensure we don't change the same edge multiple times
            while change_idx in changed_edges:
                change_idx = randint(0, num_edges - 1)  # randomly choose edge

            changed_edges.append(change_idx)  # Store the index of the changed edge

            edge = self.gb.list_edges[change_idx]

            # get weight of selected edge
            current_weight = self.gb.G[edge[0]][edge[1]]["weight"]
            print(f"Current weight of edge {edge} is {current_weight}")

            while True:
                # Randomly modify weight of the edge
                new_weight = np.random.normal(0, normal_std)  #

                if np.abs(new_weight) > 0.01:  # Ensure the new weight is significant
                    # print("New weight generated:", np.round(new_weight, 2))
                    break

            self.gb.G[edge[0]][edge[1]]["weight"] = np.round(np.clip(new_weight + current_weight, 0.01, 1), 2)
            print("New weight of edge", edge, "is", self.gb.G[edge[0]][edge[1]]["weight"], "\n")

        self.gb.title_suffix = f"Randomly modified {num_changes} edge weights"

    def shift_weights(self, shift=0, scale=1):
        """
        Shift the weights of the edges in graph G by an
         additive and/or multiplicative factor.

        Args:
            shift (float): Value to add to each edge weight.
            scale (float): Factor to multiply each edge weight by.
        """

        if shift == 0 and scale == 1:
            raise ValueError("Adding 0 and scaling by 1. No modification applied.")

        for _, _, params in self.gb.G.edges(data=True):
            old_weight = params["weight"]
            new_weight = old_weight * scale  # multiply
            new_weight += shift  # add
            params["weight"] = np.clip(
                np.round(new_weight, 2), 0.01, 1
            )  # Round to two decimal places, clip to [0.01, 1]

        self.gb.title_suffix = f"Shifted all edge weights by {shift} and scaled by {scale}"

    def modify_lag(self, node, new_node):
        """
        Modify the lag of a specific node in graph G by replacing it with
        the same node value at a different time lag

        Args:
            G (nx.DiGraph): The graph to modify.
            node (str): The node whose lag will be modified.
            new_node (str): The new node with the modified lag.

        """
        # Update the node's lag
        if node in self.gb.G.nodes():
            # replace node with new_node:
            nx.relabel_nodes(self.gb.G, {node: new_node}, copy=False)
            # Update the used nodes set
            self.gb.used_nodes.discard(node)  # Remove the old node
            self.gb.used_nodes.add(new_node)  # Add the new node
        else:
            raise ValueError(f"Node {node} does not exist in the graph.")

    def randomly_modify_lag(self, num_changes=1):
        """
        Randomly modify the lag of x parent nodes in graph G.

        Args:
            num_changes (int): The number of nodes to modify
        """
        changed_nodes = []

        if num_changes > len(self.gb.used_nodes):
            raise ValueError("Number of changes exceeds the number of nodes in the graph.")

        for _ in range(num_changes):
            while True:
                # Randomly choose a node to modify that hasn't already been modified
                node = np.random.choice(list(self.gb.used_nodes - set(changed_nodes)))

                # split node into node name and time lag
                node_name, time_lag = node.split(", ")

                # don't modify node with T0
                if node not in changed_nodes and time_lag != "T0":
                    changed_nodes.append(node)
                    break
            print("Modifying lag for node:", node)

            while True:
                new_lag = np.random.randint(-self.gb.max_lag, -1)  # Randomly choose a new lag
                if new_lag != int(time_lag[1:]):  # Ensure the new lag is different from the current one
                    break
            new_node = f"{node_name}, T{new_lag}"  # Create a new node with the modified lag
            self.modify_lag(node, new_node)

            # add new node to used_nodes
            # used_nodes.add(new_node) # TODO: here, we probably also have to rebuild the graph after and make a new used_nodes list
            # TODO integrate the following:
            # self.gb.used_nodes = new_used_nodes  # Update the used nodes in the new graph builder
            # self.gb.rebuild_dense_graph()  # Rebuild the dense graph with updated lags
        self.gb.title_suffix = f"Randomly modified {num_changes} nodes' lags"

    def shift_all_nodes_lag(self, shift_value=1):
        """
        Shift all nodes' lags by a specified value.

        Args:
            shift_value (int): The value by which to shift the lags.

        Returns:
            nx.DiGraph: The modified graph with updated node lags.
        """
        # sort ascending / descending based on sign of shift value
        if shift_value > 0:
            sorted_used_nodes = sorted(self.gb.used_nodes, key=lambda node_val: get_timelag(node_val))  # ascending
        else:
            sorted_used_nodes = sorted(self.gb.used_nodes, key=lambda node_val: get_timelag(node_val), reverse=True)

        new_used_nodes = set()
        for node in sorted_used_nodes:

            # get name and time lag #TODO: make into function?
            node_name, _ = node.split(", ")
            time_lag = get_timelag(node)

            # don't shift nodes with T0 time lag
            if time_lag < 0:

                new_lag = time_lag + shift_value  # Shift the lag
                new_lag = np.clip(new_lag, -5, -1)  # max of t-1, min of -tau

                # Ensure the new lag is different from the current one i.e. keep within T-1 and T-tau boundary
                if new_lag != time_lag:
                    new_node = f"{node_name}, T{new_lag}"  # Create a new node with the modified lag
                    self.modify_lag(node, new_node)
                    node = new_node

            new_used_nodes.add(node)

        self.gb.rebuild_dense_graph()  # Rebuild the dense graph with updated lags
        self.gb.title_suffix = f"Shifted all nodes' lags by {shift_value}"

    def delete_empty_nodes(self):
        """
        Remove nodes from graph G that have no incoming or outgoing edges
        """
        empty_nodes = [
            node for node in self.gb.used_nodes if self.gb.G.out_degree(node) == 0 and self.gb.G.in_degree(node) == 0
        ]

        for node in empty_nodes:
            self.gb.G.remove_node(node)
            self.gb.used_nodes.discard(node)
            print(f"Removed empty node {node}")

    def delete_nodes(self, to_delete, avoid_t0=False):
        """
        Remove a node from the graph G.

        Args:
            to_delete (int, list): The number of nodes to randomly delete or a list of nodes to specifically delete.
            avoid_T0 (bool): If True, avoid deleting nodes with T0 time lag.
        """
        if avoid_t0:
            # Filter out nodes with T0 time lag
            eligible_nodes = {node for node in self.gb.used_nodes if "T0" not in node}

        else:
            # If avoid_t0 is False, consider all used nodes
            eligible_nodes = self.gb.used_nodes.copy()

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
            self.gb.G.remove_node(node)
            self.gb.used_nodes.discard(node)
            print(f"Removed node {node}")

        # Make sure to delete empty nodes
        self.delete_empty_nodes()

    def delete_edges(self, to_delete):
        """
        Remove specified number of edges from the graph G.

        Args:
            to_delete (list, int): The number of edges to randomly delete or a list of edges to specifically delete.
        """

        # Option 1: Delete "to_delete" number of edges randomly
        if isinstance(to_delete, int):

            if to_delete < 1:
                raise ValueError("Number of edges to delete must be at least 1.")
            elif to_delete > len(self.gb.G.edges()):
                raise ValueError("Number of edges to delete exceeds the number of edges in the graph.")

            # Randomly select to_delete number of edges to delete
            indices_to_remove = np.random.choice(range(len(self.gb.list_edges)), size=to_delete, replace=False)
            edges_to_remove = [self.gb.list_edges[i] for i in indices_to_remove]

        # Option 2: Delete specific edges from a list
        elif isinstance(to_delete, list):
            edges_to_remove = to_delete

            # Check if all edges to delete exist in the graph
            if not all(self.gb.G.has_edge(*edge) for edge in edges_to_remove):
                raise ValueError("Some edges to delete do not exist in the graph.")

            self.gb.title_suffix = f"Deleted edges: {'& '.join(map(str, edges_to_remove))}"

        else:
            raise ValueError("to_delete must be an integer or a list of edges.")

        print("Edges to remove: ", edges_to_remove)

        self.gb.G.remove_edges_from(edges_to_remove)

        # make sure to delete empty nodes
        self.delete_empty_nodes()

    def create_random_edge(self, source_nodes, target_nodes):
        """
        Create a new edge in graph G by randomly selecting a source and target node from provided potential nodes

        Args:
            source_nodes (list): List of source nodes to choose from.
            target_nodes (list): List of target nodes to choose from.
        """
        # sample new edge weight from mean edge weight and add noise
        edge_weights = list(self.gb.G.edges(data="weight"))  # list (source, target, weight)
        mean_edge_weight = np.mean([weight for _, _, weight in edge_weights])
        stdv_edge_weight = np.std([weight for _, _, weight in edge_weights])

        weight = np.random.normal(mean_edge_weight, stdv_edge_weight)
        # Clip the weight to be within [0.01, 1.0] and round to 2 decimal places
        weight = np.round(np.clip(weight, 0.01, 1.0), 2)

        source_node = np.random.choice(source_nodes)
        target_node = np.random.choice(target_nodes)

        return source_node, target_node, weight

    def insert_edges(self, to_add):
        """
        Randomly add specified number of edges to graph G.

        Args:
            to_add (int): The number of edges to add.
        """
        print("Adding edges to the graph...")
        source_nodes = list(self.gb.used_nodes - {node for node in self.gb.used_nodes if "T0" in node})
        target_nodes = [node for node in self.gb.used_nodes if "T0" in node]
        possible_source_nodes = list(self.gb.G.nodes() - target_nodes)
        possible_edges = (
            len(possible_source_nodes) * len(target_nodes) - len(self.gb.G.edges()) - 1
        )  # possible edges not including existing edges

        # print(f"Possible source nodes: {len(possible_source_nodes)}, Target nodes: {len(target_nodes)}")
        # print(f"Possible edges to add: {possible_edges}")

        if isinstance(to_add, int):  # this int check might be a bit overkill
            if to_add < 1:
                raise ValueError("Must add at least one edge.")
            elif to_add > possible_edges:  # check if to_add is greater than possible edges
                raise ValueError(
                    "Number of edges to add cannot exceed the number of possible edges remaining in the graph."
                )

            for _ in range(to_add):
                while True:
                    source_node, target_node, weight = self.create_random_edge(source_nodes, target_nodes)

                    # Add edge only if it doesn't already exist
                    if not self.gb.G.has_edge(source_node, target_node):
                        self.gb.G.add_edge(source_node, target_node, weight=float(weight))
                        print(f"Added edge from {source_node} to {target_node} with weight {weight}")
                        break

            self.gb.title_suffix = f"Inserted {to_add} random edges"

        else:
            raise ValueError("to_add must be an integer.")

    def insert_nodes(self, to_add):
        """
        Randomly add a new node in graph G within the time range of the graph (edge to node in T0)
        :param to_add (int): The number of nodes to add.
        :return:
        """

        target_nodes = [node for node in self.gb.G.nodes() if "T0" in node]  # all T0 nodes

        # all nodes not in use or T0
        possible_source_nodes = list(self.gb.G.nodes() - target_nodes - self.gb.used_nodes)
        # print("Possible source nodes:", possible_source_nodes)

        if isinstance(to_add, int):
            # error checks
            if to_add < 1:
                raise ValueError("Must add at least one node.")
            elif to_add > len(possible_source_nodes):
                raise ValueError("Number of nodes to add exceeds the number of nodes in the graph.")

            for _ in range(to_add):

                # randomly select a source & target node from the possible nodes
                source_node, target_node, weight = self.create_random_edge(possible_source_nodes, target_nodes)

                self.gb.G.add_edge(source_node, target_node, weight=float(weight))  # add the new node to the graph
                self.gb.used_nodes.add(source_node)  # add source node to used nodes

                # remove node from possible source nodes so it's not reused
                possible_source_nodes.remove(source_node)
                print(f"Added node {source_node} connected to {target_node} with weight {weight}")

            self.gb.title_suffix = f"Randomly added {to_add} new nodes"
