from networkx.generators import time_series
import numpy as np

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
import pandas as pd
from matplotlib import pyplot as plt

from tigramite.independence_tests.parcorr import ParCorr

from causal_graph_comparison.dim_reduction import get_mean_modes, spatial_subsample
from causal_graph_comparison import *

def correlation_matrix(subsampled, var_names):

    # Calculate correlations between modes
    reshaped = subsampled[0, :, :].reshape(-1, len(var_names)).T
    correlations = np.corrcoef(reshaped)

    print(f"\nCorrelation matrix between modes (first sample):")
    corr_matrix = pd.DataFrame(correlations, index=var_names, columns=var_names)
    print(corr_matrix)
    return corr_matrix

def count_causal_links(graph, var_names, tau_max):
    # Count causal links per mode
    print(f"\nCausal link counts per mode:")
    for i, mode in enumerate(var_names):
        incoming = np.sum(graph[i, :, :] != "")
        outgoing = np.sum(graph[:, i, :] != "")
        print(f"{mode}: {incoming} incoming, {outgoing} outgoing links")
    return

def print_links(graph, var_names, tau_max):
    # Print the discovered links
    print(f"\nDiscovered causal links:")
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            for tau in range(tau_max):
                if graph[i, j, tau] != "":
                    print(f"{var_names[i]} {graph[i, j, tau]} {var_names[j]} (lag {tau+1})")
    return

def plot_causal_graph(graph, val_matrix, var_names, num_modes, difficulty, model_name, seed):
    print(f"Plotting causal graph...")

    # Plot graph for each lag
    tp.plot_time_series_graph(
        graph = graph,
        val_matrix=val_matrix,
        figsize=(8, 8),
        node_size=0.05,
        var_names=var_names,
        link_colorbar_label='PC')
    plt.title(f"Causal graph for {model_name} (PCMCI+)\n Trained on SAVAR with {num_modes} modes, difficulty {difficulty}, seed {seed}", pad=20)
    plt.savefig(f"{OUTPUTS_DIR}/{model_name}-modes_{num_modes}-diff_{difficulty}-seed_{seed}-pcmci_causal_discovery.png", dpi=300)
    plt.close()
    return

def get_num_gt_connections(links_coeffs):
    """
    Get the number of ground truth connections from the links_coeffs.
    """
    # count number of links
    n_links = 0
    for link in links_coeffs.values():
        n_links += len(link)
    return n_links

def causal_discovery(timeseries, num_modes, links_coeffs, tau_max, model_name, difficulty, seed, subsample, tau_min=1, significance_level=0.0001):
    """
    Run causal discovery on the outputs, targets, and inputs.
    Accept either array or path to array.
    """
    output_filename = f"{OUTPUTS_DIR}/{model_name}-modes_{num_modes}-diff_{difficulty}-seed_{seed}-pcmci_causal_discovery.npz"
    if Path(output_filename).exists():
        print(f"Causal discovery results already exist for {model_name}, skipping...")
        data = np.load(output_filename)
        return data['graph'], data['val_matrix'], data['p_matrix'], data['corr_matrix'], data['var_names']

    # if timeseries is path to array, load it
    if isinstance(timeseries, Path) or isinstance(timeseries, str):
        timeseries = np.load(timeseries)['outputs']

    # subsample outputs
    if subsample == "spatial":
        # Spatial subsampling
        timeseries = spatial_subsample(timeseries, num_modes)
    elif subsample == "mean":
        # Mean dim reduction
        timeseries = get_mean_modes(timeseries, num_modes)
    else:
        raise ValueError(f"Invalid subsampling method: {subsample}, try 'spatial' or 'mean'")

    var_names = [f"mode_{i}" for i in range(num_modes)]
    dataframe = pp.DataFrame(timeseries, analysis_mode = 'multiple', var_names=var_names)

    # Learn causal graph using PCMCI+

    # Initialize PCMCI+
    pcmci = PCMCI(
        dataframe=dataframe,
        cond_ind_test=ParCorr(),
        verbosity=2
    )

    # Run PCMCI+
    results = pcmci.run_pcmciplus(
        tau_max=tau_max,  # Maximum time lag
        tau_min=tau_min,
        pc_alpha=significance_level  # Significance level 
    )

    # Extract results
    graph = results['graph']
    val_matrix = results['val_matrix']
    p_matrix = results['p_matrix']

    # Filter the graph to only include the top n_gt_connections links
    n_gt_connections = get_num_gt_connections(links_coeffs)
    graph[val_matrix< np.abs(val_matrix.flatten()[val_matrix.flatten().argsort()[::-1][n_gt_connections - 1]])] = ""

    # filter val_matrix to only include the top n_gt_connections links
    val_matrix[val_matrix< np.abs(val_matrix.flatten()[val_matrix.flatten().argsort()[::-1][n_gt_connections - 1]])] = 0

    # filter p_matrix to only include the top n_gt_connections links
    p_matrix[val_matrix< np.abs(val_matrix.flatten()[val_matrix.flatten().argsort()[::-1][n_gt_connections - 1]])] = 1

    print_links(graph, var_names, tau_max)
    count_causal_links(graph, var_names, tau_max)
    corr_matrix = correlation_matrix(timeseries, var_names)

    plot_causal_graph(graph, val_matrix, var_names, num_modes, difficulty, model_name, seed)

    # save results
    print(f"Saving causal discovery results to {output_filename}")

    # drop autocorrelation lag 0
    np.savez(output_filename, graph=graph[:,:,1:], val_matrix=val_matrix[:,:,1:], p_matrix=p_matrix[:,:,1:], corr_matrix=corr_matrix, var_names=var_names)

    return graph, val_matrix, p_matrix, corr_matrix, var_names

if __name__ == "__main__":

    links_coeffs = {"0": [[[0, -2], 0.5], [[2, -4], 0.48]], "1": [[[1, -2], 0.17], [[0, -4], 0.28]], "2": [[[2, -3], 0.29], [[0, -5], 0.36]], "3": [[[3, -3], 0.17], [[2, -5], 0.68]]}
    n_gt_connections = get_num_gt_connections(links_coeffs)
    print(n_gt_connections)

    timeseries = np.load(OUTPUTS_DIR / "test_results-mlp-savar-1000samples-50steps.npz")['outputs']
    timeseries = np.transpose(timeseries, (1, 0, 2))
    print(timeseries.shape)

    graph, val_matrix, p_matrix, corr_matrix, var_names = causal_discovery(timeseries=timeseries, num_modes=4, links_coeffs=links_coeffs, tau_max=5, model_name="mlp", difficulty="med-easy", seed=1, subsample=True)