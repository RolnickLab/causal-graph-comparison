import numpy as np
from pathlib import Path

from causal_graph_comparison import *
from climatem.synthetic_data.graph_evaluation_ilija import *
from climatem.plotting.plot_model_output import Plotter

from causal_graph_comparison.utils import PicabuResultsPaths

def binarize_array(array: np.ndarray) -> np.ndarray:
    """Binarize an array.

    Args:
        array: The array to binarize.

    Returns:
        The binarized array.
    """

    return (array > 0).astype(np.int8)


def permute_graph(datamodule, experiment_name):

    results = PicabuResultsPaths(experiment_name)

    # load results
    learned_graph = results.graph
    inferred_modes = results.w_decoder

    # load params
    lat = datamodule.hparams.lat
    lon = datamodule.hparams.lon
    tau = datamodule.tau

    # load savar gt from datamodule
    print(f"Loading savar gt from SAVAR datamodule")
    savar_gt = datamodule.savar_gt_adj
    modes_gt = datamodule.savar_gt_modes_weights

    print("savar_gt: ", savar_gt.shape)
    print("modes_gt from datamodule: ", modes_gt.shape)
    num_modes = savar_gt.shape[1]

    # if loading from dataloader still doesnt work after reloading the datamodule, use this instead
    # modes_gt = results.modes_gt
    # print("modes_gt from numpy: ", modes_gt.shape)

    plotter = Plotter()

    # Plot learned graph vs savar gt before permutation
    plotter.plot_adjacency_matrix(
        mat1=learned_graph[::-1],
        mat2=savar_gt,
        path=results.results_path/Path("plots"),
        name_suffix="final",
        no_gt=False,
        iteration=0,
        plot_through_time=False,
    )

    # TODO: remember hardcoded permutation function for 100 mondes
    # Permute learned graph using CDSD modes
    permuted_temporal_matrix = np.array(
        load_and_permute_all_matrices(inferred_modes, modes_gt, learned_graph, savar_gt, lat, lon, tau)
    )

    # Plot permuted graph vs savar gt after permutation
    plotter.plot_adjacency_matrix(
        mat1=permuted_temporal_matrix[::-1],
        mat2=savar_gt,
        path=results.results_path/Path("plots"),
        name_suffix="permuted",
        no_gt=False,
        iteration=0,
        plot_through_time=False,
    )

    # reverse order of time steps (permutation outputs matrix in reverse order with last time step first)
    permuted_temporal_matrix = np.flip(permuted_temporal_matrix, axis=0)

    # add extra autocorr timesteps to first dimension of permuted_temporal_matrix dimensions (5, 4, 4) -> (6, 4, 4)
    permuted_temporal_matrix = np.concatenate([np.zeros((1, num_modes, num_modes)), permuted_temporal_matrix], axis=0)

    print("permuted_temporal_matrix shape: ", permuted_temporal_matrix.shape)
    print("permuted_temporal_matrix: ", permuted_temporal_matrix)

    save_name = f"{experiment_name}-picabu_cdsd.npz"

    save_path = OUTPUTS_DIR / Path(save_name)

    np.savez(save_path, val_matrix=permuted_temporal_matrix)

    return permuted_temporal_matrix


def flatten_temporal_adjacency_graph(shape: str, causal_method: str = None, exp_params: dict = None, savar_params: dict = None, model_name: str = None, graph: np.ndarray = None, experiment_name: str = None, density_output: str = 'sparse', density_input: str = 'sparse') -> np.ndarray:
    """Flatten a temporal adjacency graph.

    Args:
        shape: The shape of the temporal adjacency graph (Num_vars (parent) x Num_vars (child) x Num_time_steps) "parent_child_time" OR (Num_time_steps, Num_var (child), Num_vars (parent)) "time_child_parent"
        graph: The temporal adjacency graph (Num_vars (parent) x Num_vars (child) x Num_time_steps) OR (Num_time_steps, Num_var (child), Num_vars (parent))

    Returns:
        The flattened temporal adjacency graph (Num_vars * Num_time_steps, Num_vars * Num_time_steps)
    """

    if causal_method != "cd" and causal_method != "crl" and causal_method is not None:
        raise ValueError(f"Invalid causal method: {causal_method}, please choose from 'cd' (causal discovery) or 'crl' (causal representation learning)")

    if not all(density_arg in ["dense", "sparse"] for density_arg in [density_output, density_input]):
        raise ValueError(f"Invalid density: {density_output} and {density_input}, please choose from 'dense' or 'sparse'")

    # if graph is not provided, load it from the results path
    if graph is None:
        try:
            if causal_method == "crl":
                print("Loading picabu graph at: ", f"{OUTPUTS_DIR}/{experiment_name}-picabu_cdsd.npz")
                temporal_graph_path = f"{OUTPUTS_DIR}/{experiment_name}-picabu_cdsd.npz"
                temporal_graph = np.load(temporal_graph_path)['val_matrix']
            else: # causal_method == "cd"
                print("Loading causal discovery graph at: ", f"{OUTPUTS_DIR}/{experiment_name}-pcmci_causal_discovery.npz")
                temporal_graph_path = f"{OUTPUTS_DIR}/{experiment_name}-pcmci_causal_discovery.npz"
                temporal_graph = np.load(temporal_graph_path)['val_matrix']
        except FileNotFoundError:
            raise FileNotFoundError(f"Temporal graph not found, check your exp_params, savar_params, and model_name")

    else:
        temporal_graph = graph

    # print("Initial graph shape:", temporal_graph.shape)
    # print("Initial graph:\n", temporal_graph)
    # print("")

    if shape == "parent_child_time":
        num_nodes, _, time_steps = temporal_graph.shape
    elif shape == "time_child_parent":
        time_steps, _, num_nodes = temporal_graph.shape
    else:
        raise ValueError(f"Invalid shape: {shape}, please choose from 'parent_child_time' or 'time_child_parent'")

    print(f"DBG: Num nodes: {num_nodes}")
    print(f"DBG: time_steps: {time_steps}")

    # print(f"num_vars: {num_vars}")
    # print(f"time_steps: {time_steps}")

    new_dims = num_nodes * time_steps 

    # initialize new adjacency matrix with zeros
    flat_graph = np.zeros((new_dims, new_dims))

    # for debugging, uncomment:
    # flattened_graph = np.empty((new_dims, new_dims), dtype='<U10')

    if shape == "parent_child_time":
        print("parent_child_time: swapping axis to time_child_parent...")
        temporal_graph = np.swapaxes(temporal_graph, 0, -1)

    for t_lag in range(time_steps): #i
        for child in range(num_nodes): #j
            for parent in range(num_nodes): #k
                # DBG:
                # print(f"i: {i}, j: {j}, k: {k}")
                # print(f"graph[i, j, k]: {graph[i, j, k]}")
                # print(f"new row: {k * time_steps +i}")
                # print(f"new col: {j * time_steps}")
                # print("=============")
                flat_graph[parent * time_steps +t_lag, child * time_steps] = temporal_graph[t_lag, child, parent]

    # keep copy of initial flat graph
    initial_flat_graph = np.copy(flat_graph)
    
    # Iterate over all possible source nodes and times
    for node_src in range(num_nodes):
        # Original connections are from t > 0
        for time_src in range(1, time_steps):
            # Iterate over all possible destination nodes
            for node_dst in range(num_nodes):
                # Check for an initial connection to time=0
                time_dst_initial = 0
                row_initial = node_src * time_steps + time_src
                col_initial = node_dst * time_steps + time_dst_initial
                
                value = initial_flat_graph[row_initial, col_initial]
                
                if value > 0:
                    # Propagate this connection forward in time
                    for k in range(1, time_steps):
                        new_time_src = time_src + k
                        new_time_dst = time_dst_initial + k
                        
                        # Stop if either the new source or dest time is out of bounds
                        if new_time_src >= time_steps or new_time_dst >= time_steps:
                            break
                        
                        # Calculate new indices and set the value
                        row_new = node_src * time_steps + new_time_src
                        col_new = node_dst * time_steps + new_time_dst
                        flat_graph[row_new, col_new] = value

    # causality constraints (connections can only go forward in time, no instantaneous connections)

    # Create a vector representing the time step for each row/column
    time_indices = np.arange(new_dims) % time_steps
    
    # Create a boolean mask where connections are invalid (t_src <= t_dst)
    # We use broadcasting to compare every source time with every dest time.
    invalid_mask = time_indices[:, np.newaxis] <= time_indices[np.newaxis, :]
    
    # Apply the mask to zero out all invalid connections
    flat_graph[invalid_mask] = 0

    # Convert to numpy array if not already
    if not isinstance(flat_graph, np.ndarray):
        # print("Converting matrix to ndarray")
        flat_graph = np.array(flat_graph, dtype=np.int8)

    # populate new adjacency matrix with values from original graph

    if experiment_name is not None:
        save_name = f"{experiment_name}-flat_graph-{causal_method}.npz"
    else:
        save_name = f"flat_graph{causal_method}.npz"

    print(f"New dims: {new_dims}")
    print(f"Size of flat_graph: {flat_graph.shape}")

    save_path = OUTPUTS_DIR / Path(save_name)
    print("Saving flattened graph to: ", save_path)

    np.savez(save_path, val_matrix=flat_graph)

    return flat_graph

# Num_vars (parent) x Num_vars (child) x Num_time_steps
# temporal_test_graph = np.array([
#     [["x1t0->x1t0", "x1t1->x1t0", "x1t2->x1t0", "x1t3->x1t0", "x1t4->x1t0", "x1t5->x1t0"],
#      ["x2t0->x2t0", "x2t1->x2t0", "x2t2->x2t0", "x2t3->x2t0", "x2t4->x2t0", "x2t5->x2t0"],
#      ["x3t0->x3t0", "x3t1->x3t0", "x3t2->x3t0", "x3t3->x3t0", "x3t4->x3t0", "x3t5->x3t0"],
#      ["x4t0->x4t0", "x4t1->x4t0", "x4t2->x4t0", "x4t3->x4t0", "x4t4->x4t0", "x4t5->x4t0"]],
#     [["x1t0->x1t0", "x1t1->x1t0", "x1t2->x1t0", "x1t3->x1t0", "x1t4->x1t0", "x1t5->x1t0"],
#      ["x2t0->x2t0", "x2t1->x2t0", "x2t2->x2t0", "x2t3->x2t0", "x2t4->x2t0", "x2t5->x2t0"],
#      ["x3t0->x3t0", "x3t1->x3t0", "x3t2->x3t0", "x3t3->x3t0", "x3t4->x3t0", "x3t5->x3t0"],
#      ["x4t0->x4t0", "x4t1->x4t0", "x4t2->x4t0", "x4t3->x4t0", "x4t4->x4t0", "x4t5->x4t0"]],
#     [["x3t0->x1t0", "x3t1->x1t0", "x3t2->x1t0", "x3t3->x1t0", "x3t4->x1t0", "x3t5->x1t0"],
#      ["x3t0->x2t0", "x3t1->x2t0", "x3t2->x2t0", "x3t3->x2t0", "x3t4->x2t0", "x3t5->x2t0"],
#      ["x3t0->x3t0", "x3t1->x3t0", "x3t2->x3t0", "x3t3->x3t0", "x3t4->x3t0", "x3t5->x3t0"],
#      ["x3t0->x4t0", "x3t1->x4t0", "x3t2->x4t0", "x3t3->x4t0", "x3t4->x4t0", "x3t5->x4t0"]],
#     [["x4t0->x1t0", "x4t1->x1t0", "x4t2->x1t0", "x4t3->x1t0", "x4t4->x1t0", "x4t5->x1t0"],
#      ["x4t0->x2t0", "x4t1->x2t0", "x4t2->x2t0", "x4t3->x2t0", "x4t4->x2t0", "x4t5->x2t0"],
#      ["x4t0->x3t0", "x4t1->x3t0", "x4t2->x3t0", "x4t3->x3t0", "x4t4->x3t0", "x4t5->x3t0"],
#      ["x4t0->x4t0", "x4t1->x4t0", "x4t2->x4t0", "x4t3->x4t0", "x4t4->x4t0", "x4t5->x4t0"]],
# ]) 

# Num_time_steps x Num_var (child)x Num_vars (parent)
# temporal_test_graph = np.array([
#     [["x1t0->x1t0", "x2t0->x1t0", "x3t0->x1t0", "x4t0->x1t0"],
#      ["x1t0->x2t0", "x2t0->x2t0", "x3t0->x2t0", "x4t0->x2t0"],
#      ["x1t0->x3t0", "x2t0->x3t0", "x3t0->x3t0", "x4t0->x3t0"],
#      ["x1t0->x4t0", "x2t0->x4t0", "x3t0->x4t0", "x4t0->x4t0"]],
#     [["x1t1->x1t0", "x2t1->x1t0", "x3t1->x1t0", "x4t1->x1t0"],
#      ["x1t1->x2t0", "x2t1->x2t0", "x3t1->x2t0", "x4t1->x2t0"],
#      ["x1t1->x3t0", "x2t1->x3t0", "x3t1->x3t0", "x4t1->x3t0"],
#      ["x1t1->x4t0", "x2t1->x4t0", "x3t1->x4t0", "x4t1->x4t0"]],
#     [["x1t2->x1t0", "x2t2->x1t0", "x3t2->x1t0", "x4t2->x1t0"],
#      ["x1t2->x2t0", "x2t2->x2t0", "x3t2->x2t0", "x4t2->x2t0"],
#      ["x1t2->x3t0", "x2t2->x3t0", "x3t2->x3t0", "x4t2->x3t0"],
#      ["x1t2->x4t0", "x2t2->x4t0", "x3t2->x4t0", "x4t2->x4t0"]],
#     [["x1t3->x1t0", "x2t3->x1t0", "x3t3->x1t0", "x4t3->x1t0"],
#      ["x1t3->x2t0", "x2t3->x2t0", "x3t3->x2t0", "x4t3->x2t0"],
#      ["x1t3->x3t0", "x2t3->x3t0", "x3t3->x3t0", "x4t3->x3t0"],
#      ["x1t3->x4t0", "x2t3->x4t0", "x3t3->x4t0", "x4t3->x4t0"]],
#     [["x1t4->x1t0", "x2t4->x1t0", "x3t4->x1t0", "x4t4->x1t0"],
#      ["x1t4->x2t0", "x2t4->x2t0", "x3t4->x2t0", "x4t4->x2t0"],
#      ["x1t4->x3t0", "x2t4->x3t0", "x3t4->x3t0", "x4t4->x3t0"],
#      ["x1t4->x4t0", "x2t4->x4t0", "x3t4->x4t0", "x4t4->x4t0"]],
#     [["x1t5->x1t0", "x2t5->x1t0", "x3t5->x1t0", "x4t5->x1t0"],
#      ["x1t5->x2t0", "x2t5->x2t0", "x3t5->x2t0", "x4t5->x2t0"],
#      ["x1t5->x3t0", "x2t5->x3t0", "x3t5->x3t0", "x4t5->x3t0"],
#      ["x1t5->x4t0", "x2t5->x4t0", "x3t5->x4t0", "x4t5->x4t0"]],
# ]) 

# flattened_temporal_test_graph = flatten_temporal_adjacency_graph(temporal_test_graph)

# desired_flattened_graph = np.array(
#   [["x0t0->x1t0", "x0t0->x1t1", "x0t0->x1t2", "x0t0->x1t3", "x0t0->x1t4", "x0t0->x1t5", "x0t0->x2t0", "x0t0->x2t1", "x0t0->x2t2", "x0t0->x2t3", "x0t0->x2t4", "x0t0->x2t5", "x0t0->x3t0", "x0t0->x3t1", "x0t0->x3t2", "x0t0->x3t3", "x0t0->x3t4", "x0t0->x3t5", "x0t0->x4t0", "x0t0->x4t1", "x0t0->x4t2", "x0t0->x4t3", "x0t0->x4t4", "x0t0->x4t5"],
#    ["x0t1->x1t0", "x0t1->x1t1", "x0t1->x1t2", "x0t1->x1t3", "x0t1->x1t4", "x0t1->x1t5", "x0t1->x2t0", "x0t1->x2t1", "x0t1->x2t2", "x0t1->x2t3", "x0t1->x2t4", "x0t1->x2t5", "x0t1->x3t0", "x0t1->x3t1", "x0t1->x3t2", "x0t1->x3t3", "x0t1->x3t4", "x0t1->x3t5", "x0t1->x4t0", "x0t1->x4t1", "x0t1->x4t2", "x0t1->x4t3", "x0t1->x4t4", "x0t1->x4t5"],
#    ["x0t2->x1t0", "x0t2->x1t1", "x0t2->x1t2", "x0t2->x1t3", "x0t2->x1t4", "x0t2->x1t5", "x0t2->x2t0", "x0t2->x2t1", "x0t2->x2t2", "x0t2->x2t3", "x0t2->x2t4", "x0t2->x2t5", "x0t2->x3t0", "x0t2->x3t1", "x0t2->x3t2", "x0t2->x3t3", "x0t2->x3t4", "x0t2->x3t5", "x0t2->x4t0", "x0t2->x4t1", "x0t2->x4t2", "x0t2->x4t3", "x0t2->x4t4", "x0t2->x4t5"],
#    ["x0t3->x1t0", "x0t3->x1t1", "x0t3->x1t2", "x0t3->x1t3", "x0t3->x1t4", "x0t3->x1t5", "x0t3->x2t0", "x0t3->x2t1", "x0t3->x2t2", "x0t3->x2t3", "x0t3->x2t4", "x0t3->x2t5", "x0t3->x3t0", "x0t3->x3t1", "x0t3->x3t2", "x0t3->x3t3", "x0t3->x3t4", "x0t3->x3t5", "x0t3->x4t0", "x0t3->x4t1", "x0t3->x4t2", "x0t3->x4t3", "x0t3->x4t4", "x0t3->x4t5"],
#    ["x0t4->x1t0", "x0t4->x1t1", "x0t4->x1t2", "x0t4->x1t3", "x0t4->x1t4", "x0t4->x1t5", "x0t4->x2t0", "x0t4->x2t1", "x0t4->x2t2", "x0t4->x2t3", "x0t4->x2t4", "x0t4->x2t5", "x0t4->x3t0", "x0t4->x3t1", "x0t4->x3t2", "x0t4->x3t3", "x0t4->x3t4", "x0t4->x3t5", "x0t4->x4t0", "x0t4->x4t1", "x0t4->x4t2", "x0t4->x4t3", "x0t4->x4t4", "x0t4->x4t5"],
#    ["x0t5->x1t0", "x0t5->x1t1", "x0t5->x1t2", "x0t5->x1t3", "x0t5->x1t4", "x0t5->x1t5", "x0t5->x2t0", "x0t5->x2t1", "x0t5->x2t2", "x0t5->x2t3", "x0t5->x2t4", "x0t5->x2t5", "x0t5->x3t0", "x0t5->x3t1", "x0t5->x3t2", "x0t5->x3t3", "x0t5->x3t4", "x0t5->x3t5", "x0t5->x4t0", "x0t5->x4t1", "x0t5->x4t2", "x0t5->x4t3", "x0t5->x4t4", "x0t5->x4t5"],
#    ["x1t0->x1t0", "x1t0->x1t1", "x1t0->x1t2", "x1t0->x1t3", "x1t0->x1t4", "x1t0->x1t5", "x1t0->x2t0", "x1t0->x2t1", "x1t0->x2t2", "x1t0->x2t3", "x1t0->x2t4", "x1t0->x2t5", "x1t0->x3t0", "x1t0->x3t1", "x1t0->x3t2", "x1t0->x3t3", "x1t0->x3t4", "x1t0->x3t5", "x1t0->x4t0", "x1t0->x4t1", "x1t0->x4t2", "x1t0->x4t3", "x1t0->x4t4", "x1t0->x4t5"],
#    ["x1t1->x1t0", "x1t1->x1t1", "x1t1->x1t2", "x1t1->x1t3", "x1t1->x1t4", "x1t1->x1t5", "x1t1->x2t0", "x1t1->x2t1", "x1t1->x2t2", "x1t1->x2t3", "x1t1->x2t4", "x1t1->x2t5", "x1t1->x3t0", "x1t1->x3t1", "x1t1->x3t2", "x1t1->x3t3", "x1t1->x3t4", "x1t1->x3t5", "x1t1->x4t0", "x1t1->x4t1", "x1t1->x4t2", "x1t1->x4t3", "x1t1->x4t4", "x1t1->x4t5"],
#    ["x1t2->x1t0", "x1t2->x1t1", "x1t2->x1t2", "x1t2->x1t3", "x1t2->x1t4", "x1t2->x1t5", "x1t2->x2t0", "x1t2->x2t1", "x1t2->x2t2", "x1t2->x2t3", "x1t2->x2t4", "x1t2->x2t5", "x1t2->x3t0", "x1t2->x3t1", "x1t2->x3t2", "x1t2->x3t3", "x1t2->x3t4", "x1t2->x3t5", "x1t2->x4t0", "x1t2->x4t1", "x1t2->x4t2", "x1t2->x4t3", "x1t2->x4t4", "x1t2->x4t5"],
#    ["x1t3->x1t0", "x1t3->x1t1", "x1t3->x1t2", "x1t3->x1t3", "x1t3->x1t4", "x1t3->x1t5", "x1t3->x2t0", "x1t3->x2t1", "x1t3->x2t2", "x1t3->x2t3", "x1t3->x2t4", "x1t3->x2t5", "x1t3->x3t0", "x1t3->x3t1", "x1t3->x3t2", "x1t3->x3t3", "x1t3->x3t4", "x1t3->x3t5", "x1t3->x4t0", "x1t3->x4t1", "x1t3->x4t2", "x1t3->x4t3", "x1t3->x4t4", "x1t3->x4t5"],
#    ["x1t4->x1t0", "x1t4->x1t1", "x1t4->x1t2", "x1t4->x1t3", "x1t4->x1t4", "x1t4->x1t5", "x1t4->x2t0", "x1t4->x2t1", "x1t4->x2t2", "x1t4->x2t3", "x1t4->x2t4", "x1t4->x2t5", "x1t4->x3t0", "x1t4->x3t1", "x1t4->x3t2", "x1t4->x3t3", "x1t4->x3t4", "x1t4->x3t5", "x1t4->x4t0", "x1t4->x4t1", "x1t4->x4t2", "x1t4->x4t3", "x1t4->x4t4", "x1t4->x4t5"],
#    ["x1t5->x1t0", "x1t5->x1t1", "x1t5->x1t2", "x1t5->x1t3", "x1t5->x1t4", "x1t5->x1t5", "x1t5->x2t0", "x1t5->x2t1", "x1t5->x2t2", "x1t5->x2t3", "x1t5->x2t4", "x1t5->x2t5", "x1t5->x3t0", "x1t5->x3t1", "x1t5->x3t2", "x1t5->x3t3", "x1t5->x3t4", "x1t5->x3t5", "x1t5->x4t0", "x1t5->x4t1", "x1t5->x4t2", "x1t5->x4t3", "x1t5->x4t4", "x1t5->x4t5"],
#    ["x2t0->x1t0", "x2t0->x1t1", "x2t0->x1t2", "x2t0->x1t3", "x2t0->x1t4", "x2t0->x1t5", "x2t0->x2t0", "x2t0->x2t1", "x2t0->x2t2", "x2t0->x2t3", "x2t0->x2t4", "x2t0->x2t5", "x2t0->x3t0", "x2t0->x3t1", "x2t0->x3t2", "x2t0->x3t3", "x2t0->x3t4", "x2t0->x3t5", "x2t0->x4t0", "x2t0->x4t1", "x2t0->x4t2", "x2t0->x4t3", "x2t0->x4t4", "x2t0->x4t5"],
#    ["x2t1->x1t0", "x2t1->x1t1", "x2t1->x1t2", "x2t1->x1t3", "x2t1->x1t4", "x2t1->x1t5", "x2t1->x2t0", "x2t1->x2t1", "x2t1->x2t2", "x2t1->x2t3", "x2t1->x2t4", "x2t1->x2t5", "x2t1->x3t0", "x2t1->x3t1", "x2t1->x3t2", "x2t1->x3t3", "x2t1->x3t4", "x2t1->x3t5", "x2t1->x4t0", "x2t1->x4t1", "x2t1->x4t2", "x2t1->x4t3", "x2t1->x4t4", "x2t1->x4t5"],
#    ["x2t2->x1t0", "x2t2->x1t1", "x2t2->x1t2", "x2t2->x1t3", "x2t2->x1t4", "x2t2->x1t5", "x2t2->x2t0", "x2t2->x2t1", "x2t2->x2t2", "x2t2->x2t3", "x2t2->x2t4", "x2t2->x2t5", "x2t2->x3t0", "x2t2->x3t1", "x2t2->x3t2", "x2t2->x3t3", "x2t2->x3t4", "x2t2->x3t5", "x2t2->x4t0", "x2t2->x4t1", "x2t2->x4t2", "x2t2->x4t3", "x2t2->x4t4", "x2t2->x4t5"],
#    ["x2t3->x1t0", "x2t3->x1t1", "x2t3->x1t2", "x2t3->x1t3", "x2t3->x1t4", "x2t3->x1t5", "x2t3->x2t0", "x2t3->x2t1", "x2t3->x2t2", "x2t3->x2t3", "x2t3->x2t4", "x2t3->x2t5", "x2t3->x3t0", "x2t3->x3t1", "x2t3->x3t2", "x2t3->x3t3", "x2t3->x3t4", "x2t3->x3t5", "x2t3->x4t0", "x2t3->x4t1", "x2t3->x4t2", "x2t3->x4t3", "x2t3->x4t4", "x2t3->x4t5"],
#    ["x2t4->x1t0", "x2t4->x1t1", "x2t4->x1t2", "x2t4->x1t3", "x2t4->x1t4", "x2t4->x1t5", "x2t4->x2t0", "x2t4->x2t1", "x2t4->x2t2", "x2t4->x2t3", "x2t4->x2t4", "x2t4->x2t5", "x2t4->x3t0", "x2t4->x3t1", "x2t4->x3t2", "x2t4->x3t3", "x2t4->x3t4", "x2t4->x3t5", "x2t4->x4t0", "x2t4->x4t1", "x2t4->x4t2", "x2t4->x4t3", "x2t4->x4t4", "x2t4->x4t5"],
#    ["x2t5->x1t0", "x2t5->x1t1", "x2t5->x1t2", "x2t5->x1t3", "x2t5->x1t4", "x2t5->x1t5", "x2t5->x2t0", "x2t5->x2t1", "x2t5->x2t2", "x2t5->x2t3", "x2t5->x2t4", "x2t5->x2t5", "x2t5->x3t0", "x2t5->x3t1", "x2t5->x3t2", "x2t5->x3t3", "x2t5->x3t4", "x2t5->x3t5", "x2t5->x4t0", "x2t5->x4t1", "x2t5->x4t2", "x2t5->x4t3", "x2t5->x4t4", "x2t5->x4t5"],
#    ["x3t0->x1t0", "x3t0->x1t1", "x3t0->x1t2", "x3t0->x1t3", "x3t0->x1t4", "x3t0->x1t5", "x3t0->x2t0", "x3t0->x2t1", "x3t0->x2t2", "x3t0->x2t3", "x3t0->x2t4", "x3t0->x2t5", "x3t0->x3t0", "x3t0->x3t1", "x3t0->x3t2", "x3t0->x3t3", "x3t0->x3t4", "x3t0->x3t5", "x3t0->x4t0", "x3t0->x4t1", "x3t0->x4t2", "x3t0->x4t3", "x3t0->x4t4", "x3t0->x4t5"],
#    ["x3t1->x1t0", "x3t1->x1t1", "x3t1->x1t2", "x3t1->x1t3", "x3t1->x1t4", "x3t1->x1t5", "x3t1->x2t0", "x3t1->x2t1", "x3t1->x2t2", "x3t1->x2t3", "x3t1->x2t4", "x3t1->x2t5", "x3t1->x3t0", "x3t1->x3t1", "x3t1->x3t2", "x3t1->x3t3", "x3t1->x3t4", "x3t1->x3t5", "x3t1->x4t0", "x3t1->x4t1", "x3t1->x4t2", "x3t1->x4t3", "x3t1->x4t4", "x3t1->x4t5"],
#    ["x3t2->x1t0", "x3t2->x1t1", "x3t2->x1t2", "x3t2->x1t3", "x3t2->x1t4", "x3t2->x1t5", "x3t2->x2t0", "x3t2->x2t1", "x3t2->x2t2", "x3t2->x2t3", "x3t2->x2t4", "x3t2->x2t5", "x3t2->x3t0", "x3t2->x3t1", "x3t2->x3t2", "x3t2->x3t3", "x3t2->x3t4", "x3t2->x3t5", "x3t2->x4t0", "x3t2->x4t1", "x3t2->x4t2", "x3t2->x4t3", "x3t2->x4t4", "x3t2->x4t5"],
#    ["x3t3->x1t0", "x3t3->x1t1", "x3t3->x1t2", "x3t3->x1t3", "x3t3->x1t4", "x3t3->x1t5", "x3t3->x2t0", "x3t3->x2t1", "x3t3->x2t2", "x3t3->x2t3", "x3t3->x2t4", "x3t3->x2t5", "x3t3->x3t0", "x3t3->x3t1", "x3t3->x3t2", "x3t3->x3t3", "x3t3->x3t4", "x3t3->x3t5", "x3t3->x4t0", "x3t3->x4t1", "x3t3->x4t2", "x3t3->x4t3", "x3t3->x4t4", "x3t3->x4t5"],
#    ["x3t4->x1t0", "x3t4->x1t1", "x3t4->x1t2", "x3t4->x1t3", "x3t4->x1t4", "x3t4->x1t5", "x3t4->x2t0", "x3t4->x2t1", "x3t4->x2t2", "x3t4->x2t3", "x3t4->x2t4", "x3t4->x2t5", "x3t4->x3t0", "x3t4->x3t1", "x3t4->x3t2", "x3t4->x3t3", "x3t4->x3t4", "x3t4->x3t5", "x3t4->x4t0", "x3t4->x4t1", "x3t4->x4t2", "x3t4->x4t3", "x3t4->x4t4", "x3t4->x4t5"],
#    ["x3t5->x1t0", "x3t5->x1t1", "x3t5->x1t2", "x3t5->x1t3", "x3t5->x1t4", "x3t5->x1t5", "x3t5->x2t0", "x3t5->x2t1", "x3t5->x2t2", "x3t5->x2t3", "x3t5->x2t4", "x3t5->x2t5", "x3t5->x3t0", "x3t5->x3t1", "x3t5->x3t2", "x3t5->x3t3", "x3t5->x3t4", "x3t5->x3t5", "x3t5->x4t0", "x3t5->x4t1", "x3t5->x4t2", "x3t5->x4t3", "x3t5->x4t4", "x3t5->x4t5"],
#    ])

if __name__ == "__main__":

    # experiment_name = "mlp-modes_4-diff_easy-seed_1"
    # flattened_graph = flatten_temporal_adjacency_graph(shape="parent_child_time", causal_method="cd", experiment_name=experiment_name)
    # binarize graph
    # print("flattened_graph.shape: ", flattened_graph.shape)
    # print("flattened_graph: ", flattened_graph)

    test_graph_tcp = np.array([ 
        [["aa0", "ba0"],
        ["ab0", "bb0"]],
        [["aa1", "ba1"], # from node a to a with a time lag of 1, from b to a with a time lag of 1 (child node is at time 0)
        ["ab1", "bb1"]], 
        [["aa2", "ba2"],
        ["ab2", "bb2"]],
        [["aa3", "ba3"],
        ["ab3", "bb3"]]
    ])

    test_graph_pct = np.array([
        [["aa0", "aa1", "aa2", "aa3"],
        ["ab0", "ab1", "ab2", "ab3"]],
        [["ba0", "ba1", "ba2", "ba3"],
        ["bb0", "bb1", "bb2", "bb3"]]
    ])

    desired_output = np.array([
        ["aa0", '', '', "ab0", '', ''],
        ["aa1", '', '', "ab1", '', ''],
        ["aa2", '', '', "ab2", '', ''],
        ["aa3", '', '', "ab3", '', ''],
        ["ba0", '', '', "bb0", '', ''],
        ["ba1", '', '', "bb1", '', ''],
        ["ba2", '', '', "bb2", '', ''],
        ["ba3", '', '', "bb3", '', '']
    ])

    test_graph_tcp = np.ones((6, 4, 4))
    test_graph_pct = np.ones((4, 4, 6))

    flat_tcp = flatten_temporal_adjacency_graph(shape="time_child_parent", graph=test_graph_tcp)
    print("flat_tcp:\n", flat_tcp)

    flat_pct = flatten_temporal_adjacency_graph(shape="parent_child_time", graph=test_graph_pct)
    print(" =============== ")
    print("flat_pct:\n", flat_pct)

    test_array = np.array([
    [[0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.]],

    [[0., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 1.],
    [1., 1., 1., 1.]],

    [[1., 1., 0., 0.],
    [0., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 0., 0., 0.]],

    [[0., 0., 0., 1.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 0., 0.]],

    [[0., 0., 1., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.],
    [0., 0., 0., 0.]],

    [[0., 0., 0., 0.],
    [0., 0., 1., 1.],
    [0., 0., 1., 0.],
    [0., 0., 0., 0.]]])

    flat_test = flatten_temporal_adjacency_graph(shape="time_child_parent", graph=test_array)
    print(f"gt test: \n {flat_test}")



