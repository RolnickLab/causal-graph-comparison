import os
from os import listdir
from os.path import isfile, join
import csv
from csv import DictReader
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import networkx_temporal as tx
import matplotlib.patches as mpatches
import math
import copy
from random import randint


def get_timelag(node):
    return int(node.split("T")[1])


class GraphBuilder:
    def __init__(self, data_path, file_num=0):
        """
        Initialize the GraphBuilder with the path to the data directory and the file number.

        Args:
            data_path (str): The path to the directory containing the data files.
            file_num (int): The index of the file to extract data from.
        """
        self.data_path = data_path
        self.file_num = file_num
        data_dict = self._extract_data_from_file(data_path, file_num)
        N = int(data_dict["N"])
        tau = int(data_dict["tau"])
        gt = eval(data_dict["links_coeffs"])
        # print("Ground Truth Links Coefficients (gt):\n", gt)

        # -- 2. Extract adjacency matrices from ground truth links_coeffs
        adj_matrices = self._extract_gt_adjacency_matrix(gt, N, tau)

        self.G = nx.DiGraph()  # Create a directed graph
        self._build_graph(gt)
        self.title_suffix = ""

    def _extract_data_from_file(self, data_path, file_num=0):

        files = [f for f in listdir(data_path) if isfile(os.path.join(data_path, f)) and f.endswith(".csv")]

        data_dict = {}

        with open(os.path.join(data_path, files[file_num]), "r") as f:
            for line in csv.DictReader(f):
                data_dict[line["Parameter"]] = line["Value"]
        return data_dict

    def _extract_gt_adjacency_matrix(self, gt, N, tau):
        """
        Extract the ground truth adjacency matrices for each time lag from the links_coeffs.

        Args:
            links_coeffs (dict): The dictionary of causal links between latent variables.
            N (int): The number of latent variables.
            tau (int): The maximum time lag.

        Returns:
            adj_matrices (np.ndarray): The ground truth adjacency matrices (tau x N x N),
                                    where each matrix corresponds to a different time lag.
        """
        # Initialize a 3D array to store adjacency matrices for each time lag (tau x N x N)
        adj_matrices = np.zeros((tau, N, N))

        # Loop through each component and its links
        for source_node, values in gt.items():
            for link, weight in values:
                target_node, lag = link  # Unpack the link tuple into target variable and lag
                time_lag = -lag  # Convert the negative lag to a positive index
                print(f"Processing link: {int(source_node), target_node} with coefficient: {weight} at time lag {lag}")
                # Only consider lags that are within the specified time window (tau)
                if time_lag <= tau:
                    if abs(weight) > 0.01:
                        adj_matrices[time_lag - 1, int(source_node), int(target_node)] = float(
                            weight
                        )  # Fill the adjacency matrix at the appropriate time lag
                    else:
                        adj_matrices[time_lag - 1, int(source_node), int(target_node)] = 0

        return adj_matrices

    def rebuild_dense_graph(self):
        for node_idx in range(self.num_nodes):
            for time_lag_idx in range(self.max_lag + 1):
                # Create unique nodes for each time lag
                node_name = f"N{node_idx}, T{-time_lag_idx}"
                node_exists = False
                for node in self.G.nodes():
                    if str(node) == node_name:
                        node_exists = True
                        break
                if not node_exists:  # if node doesn't already exist, add to graph
                    self.G.add_node(node_name)

    def _build_graph(self, gt):
        """
        Build a flattened temporal graph from the ground truth links_coeffs.

        Args:
            gt (dict): The dictionary of causal links between latent variables.

        Returns:
            nx.MultiDiGraph: A static graph with all edges from the temporal graph.
        """
        # G = nx.DiGraph()

        print(f"Flattening temporal graph into static DiGraph...")

        all_nodes = gt.keys()
        self.num_nodes = len(all_nodes)
        self.max_lag = 0

        # find longest timestep
        for child, values in gt.items():
            for (parent, lag), weight in values:
                if abs(lag) > self.max_lag:
                    self.max_lag = abs(lag)

        self.used_nodes = set()

        self.rebuild_dense_graph()

        for child, values in gt.items():
            for (parent, lag), weight in values:
                print(f"Adding edge from N{parent} -> N{child} with coefficient {weight} at lag {lag}")
                target_node = f"N{child}, T0"
                source_node = f"N{parent}, T{lag}"  # Create unique nodes for each time lag
                self.G.add_edge(source_node, target_node, weight=float(weight))

                self.used_nodes.add(source_node)
                self.used_nodes.add(target_node)

        # get node indices for used nodes
        # used_nodes = [i for i, node in enumerate(G.nodes()) if node in used_nodes]
        # remove all unused nodes
        # all_nodes = list(G.nodes())
        # for node in all_nodes:
        #     if node not in used_nodes:
        #         G.remove_node(node)

    def plot_graph(self):
        colour_t0 = "red"
        colour_tx = "blue"

        plt.gca().invert_yaxis()
        # pos = nx.spring_layout(G, seed=42)
        # {('N0', 'T0'): array([-0.2355536 ,  0.09840265]), etc
        pos = {}
        node_colours = []

        visibility = []
        for node in self.G.nodes():
            if node in self.used_nodes:
                visibility.append(1)
            else:
                visibility.append(0)

            node_name, time_lag = node.split(", ")  # Split node name and time lag
            node_num = int(node_name[1:])  # extract node number
            lag_num = int(time_lag[1:])  # extract lag number
            if lag_num == 0:
                node_colours.append(colour_t0)
            else:
                node_colours.append(colour_tx)

            # print(f"Node: {node}, Node Number: {node_num}, Lag Number: {lag_num}")
            pos[node] = np.array([lag_num, node_num])  # Use node number and lag number as coordinates

        print("Node positions:", pos)

        # Draw nodes with specified colors
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colours, node_size=1000, alpha=visibility)

        # Draw edges, labels, edge labels (as before)
        nx.draw_networkx_edges(self.G, pos, arrowstyle="simple", arrowsize=20, edge_color="gray")

        node_labels = {node: str(node) for node in self.G.nodes()}
        print("Node labels:", node_labels)
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_color="blue", font_size=8)

        # only render labels for visible nodes
        pos_to_render = {}
        node_labels_to_render = {}
        for k, v in pos.items():
            if k in self.used_nodes:
                pos_to_render[k] = v
                node_labels_to_render[k] = node_labels[k]

        nx.draw_networkx_labels(
            self.G, pos_to_render, labels=node_labels_to_render, font_size=8, font_weight="bold", font_color="white"
        )

        # create a list of nodes with T0 in them, and a list of the rest
        # colour nodes with T0 a different colour

        # TODO figure out how many original nodes there are
        title = f"Temporal Causal Graph for \n{self.num_nodes} latent variables with {self.max_lag} time lag"
        if self.title_suffix:
            title += f"\n({self.title_suffix})"
        plt.title(title, fontsize=12)  # Overall title
        # plt.show()
        plt.tight_layout()


class GraphModifier:
    def __init__(self, gb: GraphBuilder):
        self.gb: GraphBuilder = copy.deepcopy(gb)

    def randomize_weights(self, num_changes=1, normal_std=0.1):
        """
        Modify the weights of the edges in the graph G.

        Args:
            num_changes (int): The number of weights to modify.
            normal_std (float): The standard deviation for the normal distribution used to generate new weights.
        """
        list_edges = list(self.gb.G.edges())
        print("List of edges in the graph:", list_edges)
        num_edges = len(list_edges)
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

            edge = list_edges[change_idx]
            current_weight = self.gb.G[edge[0]][edge[1]]["weight"]
            print(f"Current weight of edge {edge} is {current_weight}")

            while True:
                # Randomly modify its weight
                new_weight = np.random.normal(0, normal_std)  #
                print("New weight generated:", new_weight)
                if np.abs(new_weight) > 0.01:  # Ensure the new weight is significant
                    break
            self.gb.G[edge[0]][edge[1]]["weight"] = np.round(np.clip(new_weight + current_weight, 0.01, 1), 2)
            print("New weight of edge", edge, "is", self.gb.G[edge[0]][edge[1]]["weight"])

        self.gb.title_suffix = f"Randomly modified {num_changes} edge weights with std {normal_std}"

    def shift_weights(self, shift=0, scale=1):
        """
        Shift the weights of the edges in the graph G by a
        multiplicative and/or additive factor.

        Args:
            shift (float): The factor to add to each edge weight.
            scale (float): The factor to multiply each edge weight by.
        """

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
        Modify the lag of a specific node in the graph G.

        Args:
            G (nx.DiGraph): The graph to modify.
            node (str): The node whose lag will be modified.
            new_lag (int): The new lag value to set for the node.

        Returns:
            nx.DiGraph: The modified graph with updated node lag.
        """

        # Update the node's lag
        if node in self.gb.G.nodes():
            # replace node with new_node:
            nx.relabel_nodes(self.gb.G, {node: new_node}, copy=False)
        else:
            raise ValueError(f"Node {node} does not exist in the graph.")

    def randomly_modify_lag(self, num_changes=1):
        """
        Randomly modify the lag of nodes in the graph G.

        Args:
            num_changes (int): The number of nodes to modify.
            max_lag (int): The maximum lag value to assign.
        """
        changed_nodes = []

        if num_changes > len(self.gb.used_nodes):
            raise ValueError("Number of changes exceeds the number of nodes in the graph.")

        for _ in range(num_changes):
            while True:
                # Randomly choose a node to modify
                node = np.random.choice(list(self.gb.used_nodes - set(changed_nodes)))
                # split node into node name and time lag
                node_name, time_lag = node.split(", ")
                # don't modify node with T0
                if node not in changed_nodes and time_lag != "T0":
                    changed_nodes.append(node)
                    break
            print("Modifying lag for node:", node)
            # don't modify node with T0
            # don't change node to itself
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
            sorted_used_nodes = sorted(self.gb.used_nodes, key=lambda node_val: get_timelag(node_val))  # TODO maybe

        else:
            sorted_used_nodes = sorted(self.gb.used_nodes, key=lambda node_val: get_timelag(node_val), reverse=True)

        new_used_nodes = set()
        for node in sorted_used_nodes:
            node_name, _ = node.split(", ")
            time_lag = get_timelag(node)
            if time_lag < 0:
                new_lag = time_lag + shift_value  # Shift the lag
                # max of t-1, min of -tau
                new_lag = np.clip(new_lag, -5, -1)
                if new_lag != time_lag:  # Ensure the new lag is different from the current one
                    new_node = f"{node_name}, T{new_lag}"  # Create a new node with the modified lag
                    self.modify_lag(node, new_node)
                    node = new_node
            new_used_nodes.add(node)

        self.gb.used_nodes = new_used_nodes  # Update the used nodes in the new graph builder
        self.gb.rebuild_dense_graph()  # Rebuild the dense graph with updated lags
        self.gb.title_suffix = f"Shifted all nodes' lags by {shift_value}"


def extract_latent_equations(gt):
    equations = {}

    for source_node, links in gt.items():
        equation_terms = []
        for (target_node, lag), weight in links:
            term = f"{weight} * L{target_node}(t{f' - {abs(lag)}' if lag != 0 else ''})"
            equation_terms.append(term)

        equation = " + ".join(equation_terms)
        equations[source_node] = f"L{source_node}(t) = {equation}"

    return equations


def extract_equations_from_adjacency(adj_matrices):
    num_lags, num_latents, _ = adj_matrices.shape  # 5 lags, 16 latents

    equations = {}
    for source_node in range(num_latents):
        equation_terms = []
        for lag in range(num_lags):
            adj_matrix_at_lag = adj_matrices[lag]  # Get the adjacency matrix for the current lag
            for target_node in range(num_latents):
                weight = adj_matrix_at_lag[source_node, target_node]
                if weight != 0:  # Only include non-zero coefficients
                    term = f"{weight} * L{target_node}(t - {lag + 1})"
                    equation_terms.append(term)

        # Join the terms to create the equation
        if equation_terms:
            equation = " + ".join(equation_terms)
            equations[source_node] = f"L{source_node}(t) = {equation}"
        else:
            equations[source_node] = f"L{source_node}(t) = 0"  # No dependencies found

    return equations


# ----- Main script -----

# TODO: rename G to graph


# --- 1. Extract data from params file ---
data_path = "../data"
chosen_file = 0  # TODO: make this dynamic, e.g. via command line argument, eventually loop through all files
gb = GraphBuilder(data_path=data_path, file_num=chosen_file)

# -- 3. Extract equations from gt & adjacency matrices
# equations = extract_latent_equations(gt)
# equations_from_adj = extract_equations_from_adjacency(adj_matrices)
# print("Latent Equations:\n", equations)
# print("Equations from Adjacency Matrices:\n", equations_from_adj)

# -- 4. TEMPORAL Build temporal graph from ground truth links_coeffs ---
# print("Building temporal graph from ground truth links")
# G_temp = build_temporal_graph(gt)

# -- 4. TEMPORAL Visualize the temporal graph snapshots --
# print("Visualizing temporal graph snapshots")
# plot_temporal_graph(G_temp)

# -- 5. STATIC Build static graph from ground_truth links_coeffs ---
print("Building static graph from ground truth links")
print("Is the graph acyclic?", nx.is_directed_acyclic_graph(gb.G))  # check if G is acyclic

# -- 5. STATIC Convert G to wighted dense adjacency matrix
nx_adjacency_matrix_sparse = nx.adjacency_matrix(gb.G)  # sparse
nx_adjacency_matrix_dense = nx_adjacency_matrix_sparse.todense()  # dense
print("NetworkX Adjacency Matrix (Sparse):\n", nx_adjacency_matrix_sparse)
print("NetworkX Adjacency Matrix (Dense):\n", nx_adjacency_matrix_dense)
print("Shape of the dense adjacency matrix:", nx_adjacency_matrix_dense.shape)

# -- 5. STATIC Visualize the flattened temporal graph  --
print("Visualizing flattened temporal graph")
plt.subplot(1, 2, 1)
gb.plot_graph()

# -- 1. WEIGHTS: Randomly increase/decrease weights ---
# gm_weights_rnd = GraphModifier(gb)
# gm_weights_rnd.randomize_weights(num_changes=1)
# plt.subplot(1, 2, 2)
# gm_weights_rnd.gb.plot_graph()
# plt.show()

# -- 1. WEIGHTS: Uniformly shift weights of all edges  ---
# gm_weights_shifted = GraphModifier(gb)
# gm_weights_shifted.shift_weights(shift=0.5)
# plt.subplot(1, 2, 2)
# gm_weights_shifted.gb.plot_graph()
# plt.show()

# -- 1. WEIGHTS: Scale weights of all edges  ---
# gm_weights_scaled = GraphModifier(gb)
# gm_weights_scaled.shift_weights(scale=2)
# plt.subplot(1, 2, 2)
# gm_weights_scaled.gb.plot_graph()
# plt.show()


# -- 2. TIME LAG: modify each time lag, all 0, all tau, ch target node by 1, etc. --
# shift everything by x time with max and min of time lag / -1
# or randomize
# gm_random_lag = GraphModifier(gb)
# gm_random_lag.randomly_modify_lag(num_changes=2)
# plt.subplot(1, 2, 2)
# gm_random_lag.gb.plot_graph()
# plt.show()

# -- 2. TIME LAG: shift all nodes by time lag --
gm_shifted = GraphModifier(gb)
gm_shifted.shift_all_nodes_lag(shift_value=-1)
plt.subplot(1, 2, 2)
gm_shifted.gb.plot_graph()
plt.show()


# -- 3. REMOVE NODES + edge: remove a node or 2 for each node in graph --
# if there is a node with no connections, remove it

# -- 4. ADD NODES + edge: add a node or 2 and connect to each node but ensure always moving forward in time --
# existing nodes
# new nodes

# -- 5. ADD EDGES: Create list of graphs from gt with 1 edge added -- existing nodes

# -- 6. REMOVE EDGES: Create list of graphs from gt with 1 edge removed --
# if multiple children then also remove edge <-- mult child
# TODO: check
