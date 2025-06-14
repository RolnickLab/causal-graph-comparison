from os import listdir
from os.path import isfile, join
from pathlib import Path
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple


class GraphBuilder:
    def __init__(self, data_path: str, file_num: int = 0) -> None:
        """
        Initialize the GraphBuilder with the path to the data directory and the param file index.

        Args:
            data_path: The path to the directory containing the data files.
            file_num: The index of the param file to extract data from.
        """

        # List all object variables
        self.data_path = data_path
        self.file_num = file_num
        self.N = None
        self.tau = None
        self.ground_truth = None
        self.adj_matrices = None
        self.graph = None
        self.equations = None
        self.adj_equations = None
        self.title_suffix = ""

        # Process config file
        self._extract_data_from_file()
        # print("Ground Truth Links Coefficients (gt):\n", gt)

        # Build graph
        self.graph = nx.DiGraph()  # Create a directed graph

        self._build_graph()

        # Extract adjacency matrices from ground truth links_coeffs
        self._extract_gt_adjacency_matrix()

        self._extract_latent_equations()
        self._extract_equations_from_adjacency()

    @property
    def list_edges(self) -> List[Tuple[str, str]]:
        """
        Get the current list of edges in the graph.
        
        Returns:
            List of tuples representing the edges in the graph.
        """
        return list(self.graph.edges())

    def _extract_data_from_file(self):
        """
        Extract data from a CSV file in the specified directory.

        Args:
            data_path: Data directory path containing the CSV files.
            file_num: Index of the file to extract data from (default is 0).

        Returns:
            Dictionary containing the extracted data.
        """
        files = [str(f) for f in Path(self.data_path).glob("*.csv")]

        data_dict = {}

        with open(join(self.data_path, files[self.file_num]), "r") as f:
            for line in csv.DictReader(f):
                data_dict[line["Parameter"]] = line["Value"]

        self.ground_truth = eval(data_dict["links_coeffs"])
        self.N = int(data_dict["N"])
        self.tau = int(data_dict["tau"])
        self.gt = eval(data_dict["links_coeffs"])

    def _extract_gt_adjacency_matrix(self):
        """Extract the ground truth adjacency matrices for each time lag from the links_coeffs.

        Returns:
            The ground truth adjacency matrices (tau x N x N), where each matrix corresponds to a different time lag.
        """
        # Initialize a 3D array to store adjacency matrices for each time lag (tau x N x N)
        adj_matrices = np.zeros((self.tau, self.N, self.N))

        # First, collect all valid links and their weights
        valid_links = []
        for source_node, values in self.ground_truth.items():
            for link, weight in values:
                target_node, lag = link  # Unpack the link tuple into target variable and lag
                time_lag = -lag  # Convert the negative lag to a positive index

                # print(f"Processing link: {int(source_node), target_node} with coefficient: {weight} at time lag {lag}")

                # Only consider lags that are within the specified time window (tau)
                if time_lag <= self.tau:
                    valid_links.append((time_lag, int(source_node), int(target_node), float(weight)))

        # Fill the adjacency matrices with the valid links
        for time_lag, source_node, target_node, weight in valid_links:
            # Only use weights that are significant
            if abs(weight) > 0.01:
                adj_matrices[time_lag - 1, source_node, target_node] = weight

        self.adj_matrices = adj_matrices

    def _extract_latent_equations(self) -> dict:
        """
        Extract equations for each latent variable based on the ground truth links_coeffs.

        Returns:
            Dictionary where keys are source nodes and values are equations in string format.
        """
        equations = {}

        for source_node, links in self.ground_truth.items():
            equation_terms = []
            for (target_node, lag), weight in links:
                term = f"{weight} * L{target_node}(t{f' - {abs(lag)}' if lag != 0 else ''})"
                equation_terms.append(term)

            equation = " + ".join(equation_terms)
            equations[source_node] = f"L{source_node}(t) = {equation}"

        self.equations = equations

    def _extract_equations_from_adjacency(self) -> dict:
        """Extract equations for each latent variable based on the adjacency matrices.

        Returns:
            Dictionary where keys are source nodes and values are equations in string format.
        """
        num_lags, num_latents, _ = self.adj_matrices.shape  # n lags, x latents

        equations = {}

        # First, collect all non-zero terms for each source node
        for source_node in range(num_latents):
            equation_terms = []

            # Get all non-zero terms for this source node across all lags
            for lag in range(num_lags):
                adj_matrix_at_lag = self.adj_matrices[lag]
                # Find all target nodes with non-zero weights for this source node
                target_nodes = np.where(adj_matrix_at_lag[source_node] != 0)[0]

                # Create terms for each target node with non-zero weight
                for target_node in target_nodes:
                    weight = adj_matrix_at_lag[source_node, target_node]
                    term = f"{weight} * L{target_node}(t - {lag + 1})"
                    equation_terms.append(term)

            # Create the equation for this source node
            equation = " + ".join(equation_terms) if equation_terms else "0"
            equations[source_node] = f"L{source_node}(t) = {equation}"

        self.adj_equations = equations

    def rebuild_dense_graph(self) -> None:
        """
        Make the graph dense by adding nodes for each latent variable and time lag.
        Purpose is to ensure adjacency matrix remains consistent after modifications
        [N0 T-2, N1 T-1, N1 T0], [N1 T-2, N1 T-1, N1 T0] ... etc.
        """
        existing_graph_nodes = [str(node) for node in list(self.graph.nodes())]

        for node_idx in range(self.num_nodes):
            for time_lag_idx in range(self.max_lag + 1):
                # Create unique nodes for each time lag
                node_name = f"N{node_idx}, T{-time_lag_idx}"

                # add nodes that don't already exist
                if node_name not in existing_graph_nodes:
                    self.graph.add_node(node_name)

    def _build_graph(self) -> None:
        """
        Build a flattened temporal graph from the ground truth links_coeffs.
        """

        print("Building flattened temporal graph from ground truth links_coeffs...\n")

        all_nodes = self.ground_truth.keys()
        self.num_nodes = len(all_nodes)
        self.max_lag = 0

        # find longest timestep
        for child, values in self.ground_truth.items():
            for (parent, lag), weight in values:
                if abs(lag) > self.max_lag:
                    self.max_lag = abs(lag)

        self.used_nodes = set()

        self.rebuild_dense_graph()

        for child, values in self.ground_truth.items():
            for (parent, lag), weight in values:
                # print(f"Adding edge from N{parent} -> N{child} with coefficient {weight} at lag {lag}")

                target_node = f"N{child}, T0"
                source_node = f"N{parent}, T{lag}"  # Create unique nodes for each time lag
                self.graph.add_edge(source_node, target_node, weight=float(weight))

                self.used_nodes.add(source_node)
                self.used_nodes.add(target_node)

    def plot_graph(self) -> None:
        """Plot the graph with appropriate styling and labels."""
        colour_t0 = "red"
        colour_tx = "blue"

        plt.gca().invert_yaxis()

        pos = {}
        node_colours = []

        visibility = []
        for node in self.graph.nodes():
            if node in self.used_nodes:
                visibility.append(1)
            else:
                # if node is not used, set visibility to 0
                visibility.append(0)

            node_name, time_lag = node.split(", ")  # Split node name and time lag
            node_num = int(node_name[1:])  # extract node number
            lag_num = int(time_lag[1:])  # extract lag number

            colour_to_use = colour_tx

            if lag_num == 0:
                colour_to_use = colour_t0

            node_colours.append(colour_to_use)

            # print(f"Node: {node}, Node Number: {node_num}, Lag Number: {lag_num}")
            pos[node] = np.array([lag_num, node_num])  # Use node number and lag number as coordinates

        # print("Node positions:", pos)

        # Draw nodes with specified colors
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colours, node_size=1000, alpha=visibility)

        # Draw edges, labels, edge labels
        nx.draw_networkx_edges(self.graph, pos, arrowstyle="simple", arrowsize=20, edge_color="gray")

        node_labels = {node: str(node) for node in self.graph.nodes()}
        # print("Node labels:", node_labels)
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color="blue", font_size=8)

        # only render labels for visible nodes
        pos_to_render = {}
        node_labels_to_render = {}
        for k, v in pos.items():
            if k in self.used_nodes:
                pos_to_render[k] = v
                node_labels_to_render[k] = node_labels[k]

        nx.draw_networkx_labels(
            self.graph, pos_to_render, labels=node_labels_to_render, font_size=8, font_weight="bold", font_color="white"
        )

        title = f"Temporal Causal Graph for \n{self.num_nodes} latent variables with {self.max_lag} time lag"
        if self.title_suffix:
            title += f"\n({self.title_suffix})"
        plt.title(title, fontsize=12)  # Overall title
        plt.tight_layout()
        # plt.show()
