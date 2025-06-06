from os import listdir
from os.path import isfile, join
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class GraphBuilder:
    def __init__(self, data_path, file_num=0):
        """
        Initialize the GraphBuilder with the path to the data directory and the param file index.

        Args:
            data_path (str): The path to the directory containing the data files.
            file_num (int): The index of the param file to extract data from.
        """
        self.data_path = data_path
        self.file_num = file_num
        data_dict = self._extract_data_from_file(data_path, file_num)
        self.N = int(data_dict["N"])
        self.tau = int(data_dict["tau"])
        self.gt = eval(data_dict["links_coeffs"])
        # print("Ground Truth Links Coefficients (gt):\n", gt)

        # Extract adjacency matrices from ground truth links_coeffs
        self.adj_matrices = self._extract_gt_adjacency_matrix()

        self.G = nx.DiGraph()  # Create a directed graph
        self._build_graph(self.gt)
        self.title_suffix = ""
        self.list_edges = list(self.G.edges())

    def _extract_data_from_file(self, data_path, file_num=0):
        """
        Extract data from a CSV file in the specified directory.
        :param data_path(str): Data directory path containing the CSV files.
        :param file_num(int): Index of the file to extract data from (default is 0).
        :return:
        """

        files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith(".csv")]

        data_dict = {}

        with open(join(data_path, files[file_num]), "r") as f:
            for line in csv.DictReader(f):
                data_dict[line["Parameter"]] = line["Value"]
        return data_dict

    def _extract_gt_adjacency_matrix(self):
        """
        Extract the ground truth adjacency matrices for each time lag from the links_coeffs.

        Returns:
            adj_matrices (np.ndarray): The ground truth adjacency matrices (tau x N x N),
                                    where each matrix corresponds to a different time lag.
        """
        # Initialize a 3D array to store adjacency matrices for each time lag (tau x N x N)
        adj_matrices = np.zeros((self.tau, self.N, self.N))

        # Loop through each component and its links
        for source_node, values in self.gt.items():
            for link, weight in values:
                target_node, lag = link  # Unpack the link tuple into target variable and lag
                time_lag = -lag  # Convert the negative lag to a positive index

                # print(f"Processing link: {int(source_node), target_node} with coefficient: {weight} at time lag {lag}")

                # Only consider lags that are within the specified time window (tau)
                if time_lag <= self.tau:
                    if abs(weight) > 0.01:
                        adj_matrices[time_lag - 1, int(source_node), int(target_node)] = float(
                            weight
                        )  # Fill the adjacency matrix at the appropriate time lag
                    else:
                        adj_matrices[time_lag - 1, int(source_node), int(target_node)] = 0

        return adj_matrices

    def extract_latent_equations(self):
        """
        Extract equations for each latent variable based on the ground truth links_coeffs.
        :param gt: Dictionary of ground truth links_coeffs, where keys are target nodes and values are lists of tuples (parent, lag).
        :return:
        """
        equations = {}

        for source_node, links in self.gt.items():
            equation_terms = []
            for (target_node, lag), weight in links:
                term = f"{weight} * L{target_node}(t{f' - {abs(lag)}' if lag != 0 else ''})"
                equation_terms.append(term)

            equation = " + ".join(equation_terms)
            equations[source_node] = f"L{source_node}(t) = {equation}"

        return equations

    def extract_equations_from_adjacency(self):
        """
        Extract equations for each latent variable based on the adjacency matrices.
        :return: equations: Dictionary where keys are source nodes and values are equations in string format.
        """
        num_lags, num_latents, _ = self.adj_matrices.shape  # n lags, x latents

        equations = {}
        for source_node in range(num_latents):
            equation_terms = []
            for lag in range(num_lags):
                adj_matrix_at_lag = self.adj_matrices[lag]  # Get the adjacency matrix for the current lag
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

    def rebuild_dense_graph(self):
        """
        Make G a dense graph adding nodes for each latent variable and time lag.
        Purpose: ensure adjacency matrix remains consistent after modifications
        :return:
        """
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

        print("Building flattened temporal graph from ground truth links_coeffs...\n")

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
                # print(f"Adding edge from N{parent} -> N{child} with coefficient {weight} at lag {lag}")

                target_node = f"N{child}, T0"
                source_node = f"N{parent}, T{lag}"  # Create unique nodes for each time lag
                self.G.add_edge(source_node, target_node, weight=float(weight))

                self.used_nodes.add(source_node)
                self.used_nodes.add(target_node)

    def plot_graph(self):
        colour_t0 = "red"
        colour_tx = "blue"

        plt.gca().invert_yaxis()

        pos = {}
        node_colours = []

        visibility = []
        for node in self.G.nodes():
            if node in self.used_nodes:
                visibility.append(1)
            else:
                # if node is not used, set visibility to 0
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

        # print("Node positions:", pos)

        # Draw nodes with specified colors
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colours, node_size=1000, alpha=visibility)

        # Draw edges, labels, edge labels
        nx.draw_networkx_edges(self.G, pos, arrowstyle="simple", arrowsize=20, edge_color="gray")

        node_labels = {node: str(node) for node in self.G.nodes()}
        # print("Node labels:", node_labels)
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

        title = f"Temporal Causal Graph for \n{self.num_nodes} latent variables with {self.max_lag} time lag"
        if self.title_suffix:
            title += f"\n({self.title_suffix})"
        plt.title(title, fontsize=12)  # Overall title
        plt.tight_layout()
        # plt.show()
