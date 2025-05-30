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

def extract_data_from_file(data_path, file_num=0):
    """
    Extract data from csv file.

    Args:
        data_path (str): The path to the directory containing the data files.
        file_num (int): The index of the file to extract data from.

    Returns:
        data_dict (dict): A dictionary containing the extracted data.

    """
    files = [f for f in listdir(data_path) if isfile(os.path.join(data_path, f)) and f.endswith(".csv")]

    data_dict = {}

    with open(os.path.join(data_path, files[file_num]), "r") as f:
        for line in csv.DictReader(f):
            data_dict[line['Parameter']] = line['Value']

    return data_dict

def extract_adjacency_matrix(gt, N, tau):
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
                    adj_matrices[time_lag - 1, int(source_node), int(target_node)] = (
                        float(weight)  # Fill the adjacency matrix at the appropriate time lag
                    )
                else:
                    adj_matrices[time_lag - 1, int(source_node), int(target_node)] = 0

    return adj_matrices


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
                    term = f"{weight} * L{target_node}(t - {lag+1})"
                    equation_terms.append(term)

        # Join the terms to create the equation
        if equation_terms:
            equation = " + ".join(equation_terms)
            equations[source_node] = f"L{source_node}(t) = {equation}"
        else:
            equations[source_node] = f"L{source_node}(t) = 0"  # No dependencies found

    return equations

def build_temporal_graph(gt):
    # build graph from gt
    G = tx.TemporalMultiDiGraph()
    for source_node, values in gt.items():
        for (target_node, lag), weight in values:
            print(f"Adding edge from N{source_node} -> N{target_node} with coefficient {weight} at lag {lag}")
            G.add_edge(int(source_node), int(target_node), lag=int(lag), weight=float(weight))

    # # Debugging output
    # print(f"Graph G: {G}")
    # print(f"V = {G.order()} nodes ({G.temporal_order()} unique, {G.total_order()} total)\n")
    # print(f"E = {G.size()} edges ({G.temporal_size()} unique, {G.total_size()} total)")

    G = G.slice(attr="lag")  # Slice the graph into snapshots based on the 'lag' attribute

    return G

def plot_temporal_graph(G):
    """
    Plot the temporal graph G using NetworkX and Matplotlib.

    Args:
        G (nx.classes.reportviews.OutMultiEdgeDataView): The sliced temporal graph to plot.

    Returns:
        None
    """

    # Set axes based on the number of time snapshots
    num_timesteps = len(G)
    n_cols = math.ceil(math.sqrt(num_timesteps))
    n_rows = math.ceil(num_timesteps / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Indexing for axes
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten() # Flatten axes array for indexing
    else:
        axes = [axes] # Exception if there's only one subplot

    # Iterate through each time snapshot and plot
    for i, G_snapshot in enumerate(reversed(G.edges(data=True))):

        # Convert temporal class to MultiDiGraph class to use plotting functions
        G_snapshot, time_lag = convert_to_static_graph(G_snapshot)

        ax = axes[i]
        ax.set_title(f"Time lag = {time_lag}", fontsize=14)

        # Set position of nodes using spring layout
        pos = nx.spring_layout(G_snapshot, weight='weight', seed=42)

        # Draw nodes, edges, & labes
        nx.draw_networkx_nodes(G_snapshot, pos, ax=ax, node_size=500)
        nx.draw_networkx_edges(G_snapshot, pos, ax=ax, arrowstyle='->', edge_color='grey')
        nx.draw_networkx_labels(G_snapshot, pos, ax=ax, font_size=10, font_weight='bold', font_color='white')

        # Get edge labels (weights)
        edge_attributes = nx.get_edge_attributes(G_snapshot, 'weight')
        edge_labels = {(u, v): f"{w:.2f}" for (u, v, x), w in edge_attributes.items()}

        nx.draw_networkx_edge_labels(G_snapshot, pos, ax=ax, edge_labels=edge_labels, font_color='blue',
                                     font_size=8)

        ax.axis('off')

    # Hide empty subplots
    for i in range(num_timesteps, len(axes)):
        fig.delaxes(axes[i])

    # --- 5. Final adjustments and display ---
    plt.suptitle("Causal Graphs at Different Lags (Snapshots)", fontsize=16, y=0.98)  # Overall title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def convert_to_static_graph(G_snapshot):
    """
    Convert a temporal graph G into a static graph by merging all time snapshots.

    Args:
        G (nx.MultiDiGraph): The temporal graph to convert.

    Returns:
        nx.MultiDiGraph: A static graph with all edges from the temporal graph.

    """
    static_graph = nx.MultiDiGraph()

    for source_node, target_node, data in G_snapshot:
        static_graph.add_edge(source_node, target_node, **data)
        lag = data.get('lag', None)  # Get the lag attribute if it exists
        print(f"Added edge: {source_node} -> {target_node} with weight={data.get('weight', None)} at lag={lag}")

    return static_graph, lag
def plot_temporal_graph(G):
    """
    Plot the temporal graph G using NetworkX and Matplotlib.

    Args:
        G (nx.classes.reportviews.OutMultiEdgeDataView): The sliced temporal graph to plot.

    Returns:
        None
    """

    # Set axes based on the number of time snapshots
    num_timesteps = len(G)
    n_cols = math.ceil(math.sqrt(num_timesteps))
    n_rows = math.ceil(num_timesteps / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

    # Indexing for axes
    if n_rows > 1 or n_cols > 1:
        axes = axes.flatten() # Flatten axes array for indexing
    else:
        axes = [axes] # Exception if there's only one subplot

    # Iterate through each time snapshot and plot
    for i, G_snapshot in enumerate(reversed(G.edges(data=True))):

        # Convert temporal class to MultiDiGraph class to use plotting functions
        G_snapshot, time_lag = convert_to_static_graph(G_snapshot)

        ax = axes[i]
        ax.set_title(f"Time lag = {time_lag}", fontsize=14)

        # Set position of nodes using spring layout
        pos = nx.spring_layout(G_snapshot, weight='weight', seed=42)

        # Draw nodes, edges, & labes
        nx.draw_networkx_nodes(G_snapshot, pos, ax=ax, node_size=500)
        nx.draw_networkx_edges(G_snapshot, pos, ax=ax, arrowstyle='->', edge_color='grey')
        nx.draw_networkx_labels(G_snapshot, pos, ax=ax, font_size=10, font_weight='bold', font_color='white')

        # Get edge labels (weights)
        edge_attributes = nx.get_edge_attributes(G_snapshot, 'weight')
        edge_labels = {(u, v): f"{w:.2f}" for (u, v, x), w in edge_attributes.items()}

        nx.draw_networkx_edge_labels(G_snapshot, pos, ax=ax, edge_labels=edge_labels, font_color='blue',
                                     font_size=8)

        ax.axis('off')

    # Hide empty subplots
    for i in range(num_timesteps, len(axes)):
        fig.delaxes(axes[i])

    # --- 5. Final adjustments and display ---
    plt.suptitle("Causal Graphs at Different Lags (Snapshots)", fontsize=16, y=0.98)  # Overall title
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def build_flattened_temporal_graph(gt):
    """
    Build a flattened temporal graph from the ground truth links_coeffs.

    Args:
        gt (dict): The dictionary of causal links between latent variables.

    Returns:
        nx.MultiDiGraph: A static graph with all edges from the temporal graph.
    """
    G = nx.DiGraph()

    print(f"Flattening temporal graph into static DiGraph...")

    all_nodes = gt.keys()
    num_nodes = len(all_nodes)
    max_lag = 0

    # find longest timestep
    for child, values in gt.items():
        for (parent, lag), weight in values:
            if abs(lag) > max_lag:
                max_lag = abs(lag)

    used_nodes = set()
    for node_idx in range(num_nodes):
        for time_lag_idx in range(max_lag + 1):
            # Create unique nodes for each time lag
            node_name = f"N{node_idx}, T{-time_lag_idx}"
            G.add_node(node_name)


    for child, values in gt.items():
        for (parent, lag), weight in values:
            print(f"Adding edge from N{parent} -> N{child} with coefficient {weight} at lag {lag}")
            target_node = f"N{child}, T0"
            source_node = f"N{parent}, T{lag}"  # Create unique nodes for each time lag
            G.add_edge(source_node, target_node, weight=float(weight))

            used_nodes.add(source_node)
            used_nodes.add(target_node)

    # get node indices for used nodes
    # used_nodes = [i for i, node in enumerate(G.nodes()) if node in used_nodes]
    # remove all unused nodes
    # all_nodes = list(G.nodes())
    # for node in all_nodes:
    #     if node not in used_nodes:
    #         G.remove_node(node)

    return G, used_nodes

def plot_flattened_graph(G, used_nodes):

    colour_t0 = 'red'
    colour_tx = 'blue'

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.invert_yaxis()
    # pos = nx.spring_layout(G, seed=42)
    # {('N0', 'T0'): array([-0.2355536 ,  0.09840265]), etc
    pos = {}
    node_colours = []

    visibility = []
    max_node_num = 0  # Track the maximum node number for layout
    max_time_lag = 0  # Track the maximum time lag for layout

    for node in G.nodes():
        if node in used_nodes:
            visibility.append(1)
        else:
            visibility.append(0)

        node_name, time_lag = node.split(', ')  # Split node name and time lag
        node_num = int(node_name[1:]) # extract node number
        lag_num = int(time_lag[1:]) # extract lag number
        if lag_num == 0:
            node_colours.append(colour_t0)
        else:
            node_colours.append(colour_tx)

        # print(f"Node: {node}, Node Number: {node_num}, Lag Number: {lag_num}")
        pos[node] = np.array([lag_num, node_num])  # Use node number and lag number as coordinates

        # keep track of max node num
        max_node_num = max(max_node_num, node_num)
        max_time_lag = max(-max_time_lag, -lag_num)

    print("Node positions:", pos)

    # Draw nodes with specified colors
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours, node_size=1000, alpha=visibility)

    # Draw edges, labels, edge labels (as before)
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='simple', arrowsize=20, edge_color='gray')

    node_labels = {node: str(node) for node in G.nodes()}
    print("Node labels:", node_labels)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8)

    # only render labels for visible nodes
    pos_to_render = {}
    node_labels_to_render = {}
    for k, v in pos.items():
        if k in used_nodes:
            pos_to_render[k] = v
            node_labels_to_render[k] = node_labels[k]

    nx.draw_networkx_labels(G, pos_to_render, ax=ax, labels=node_labels_to_render,
                            font_size=8,
                            font_weight='bold',
                            font_color='white')

    # create a list of nodes with T0 in them, and a list of the rest
    # colour nodes with T0 a different colour

    # TODO figure out how many original nodes there are
    plt.suptitle(f"Flattened Temporal Causal Graph for {max_node_num + 1} latent variables with {max_time_lag} time lag", fontsize=16, y=0.98)  # Overall title
    plt.show()

if __name__ == "__main__":

    # --- 1. Extract data from params file ---
    data_path = "../cgc/data"
    chosen_file = 0 # TODO: make this dynamic, e.g. via command line argument, eventually loop through all files
    data_dict = extract_data_from_file(data_path, chosen_file)  # Extract data from the params file

    N = int(data_dict['N'])
    tau = int(data_dict['tau'])
    gt = eval(data_dict['links_coeffs'])
    # print("Ground Truth Links Coefficients (gt):\n", gt)

    # -- 2. Extract adjacency matrices from ground truth links_coeffs
    adj_matrices = extract_adjacency_matrix(gt, N, tau)
    # print("Adjacency Matrices:\n", adj_matrices)

    # -- 3. Extract equations from gt & adjacency matrices
    equations = extract_latent_equations(gt)
    equations_from_adj = extract_equations_from_adjacency(adj_matrices)
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
    G, used_nodes = build_flattened_temporal_graph(gt)
    print("Is the graph acyclic?", nx.is_directed_acyclic_graph(G)) # check if G is acyclic

    # -- 5. STATIC Convert G to wighted dense adjacency matrix
    nx_adjacency_matrix_sparse = nx.adjacency_matrix(G)  # sparse
    nx_adjacency_matrix_dense = nx_adjacency_matrix_sparse.todense() # dense
    print("NetworkX Adjacency Matrix (Sparse):\n", nx_adjacency_matrix_sparse)
    print("NetworkX Adjacency Matrix (Dense):\n", nx_adjacency_matrix_dense)

    # -- 5. STATIC Visualize the flattened temporal graph  --
    print("Visualizing flattened temporal graph")
    plot_flattened_graph(G, used_nodes)

    # -- X. EDGES: Create list of graphs from gt with 1 edge flipped -- NOT NECESSARY WILL NEVER OCCUR

    # -- 6. EDGES: Create list of graphs from gt with 1 edge added --
    # create func to create list of graphs with 1 edge flipped. maybe multiple edges

    # -- 6. EDGES: Create list of graphs from gt with 1 edge removed --

    # -- 7. ADD NODES: remove a node or 2 for each node in graph --

    # -- 8. REMOVE NODES: add a node or 2 and connect to each node but ensure acyclicity --

    # -- 9. TIME LAG: modify each time lag, all 0, all tau,  ch target node by 1, etc. --

    # -- 10. WEIGHTS: weights same as time lag

    # -- 11. Metrics: https://github.com/CausalDisco/gadjid




