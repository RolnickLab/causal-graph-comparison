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

def permute_graph(datamodule, exp_params, savar_params, model_name):

    results = PicabuResultsPaths(exp_params, savar_params, model_name)

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

    # if loading from dataloader still doesnt work after reloading the datamodule, use this instead
    # modes_gt = results.modes_gt
    # print("modes_gt from numpy: ", modes_gt.shape)

    plotter = Plotter()

    # Plot learned graph vs savar gt before permutation
    plotter.plot_adjacency_matrix(
        mat1=learned_graph,
        mat2=savar_gt[::-1],
        path=results.results_path,
        name_suffix="transition",
        no_gt=False,
        iteration=1,
        plot_through_time=True,
    )

    # TODO: remember hardcoded permutation function for 100 mondes
    # Permute learned graph using CDSD modes
    permuted_temporal_matrix = np.array(
        load_and_permute_all_matrices(inferred_modes, modes_gt, learned_graph, savar_gt, lat, lon, tau)
    )

    print("permuted_temporal_matrix shape: ", permuted_temporal_matrix.shape)
    print("permuted_temporal_matrix: ", permuted_temporal_matrix)

    # Plot permuted graph vs savar gt after permutation
    plotter.plot_adjacency_matrix(
        mat1=permuted_temporal_matrix,
        mat2=savar_gt[::-1],
        path=results.results_path,
        name_suffix="transition_permuted",
        no_gt=False,
        iteration=2,
        plot_through_time=True,
    )

    save_path = results.results_path / Path("permuted_learned_temporal_graph.npy")

    np.save(save_path, permuted_temporal_matrix)

    return permuted_temporal_matrix

def flatten_temporal_adjacency_graph(temporal_adjacency_graph):
    """
    Flatten the temporal adjacency graph.
    """
    return temporal_adjacency_graph.reshape(-1, temporal_adjacency_graph.shape[-1])


def flatten_temporal_adjacency_graph(shape: str, density_output: str = 'sparse', density_input: str = 'sparse', graph: np.ndarray = None, exp_params: dict = None, savar_params: dict = None, model_name: str = None) -> np.ndarray:
    """Flatten a temporal adjacency graph.

    Args:
        shape: The shape of the temporal adjacency graph (Num_vars (parent) x Num_vars (child) x Num_time_steps) "parent_child_time" OR (Num_time_steps, Num_var (child), Num_vars (parent)) "time_child_parent"
        graph: The temporal adjacency graph (Num_vars (parent) x Num_vars (child) x Num_time_steps) OR (Num_time_steps, Num_var (child), Num_vars (parent))

    Returns:
        The flattened temporal adjacency graph (Num_vars * Num_time_steps, Num_vars * Num_time_steps)
    """
    print("Initial graph shape:", graph.shape)
    print("Initial graph:\n", graph)
    print("")

    if not all(density_arg in ["dense", "sparse"] for density_arg in [density_output, density_input]):
        raise ValueError(f"Invalid density: {density_output} and {density_input}, please choose from 'dense' or 'sparse'")
   
    # if graph is not provided, load it from the results path
    if graph is None:
        results = PicabuResultsPaths(exp_params, savar_params, model_name)
        temporal_graph_path = results.results_path / Path("permuted_learned_temporal_graph.npy")
        temporal_graph = np.load(temporal_graph_path)

    else:
        temporal_graph = graph

    if shape == "parent_child_time":
        num_vars, _, time_steps = temporal_graph.shape
    elif shape == "time_child_parent":
        time_steps, _, num_vars = temporal_graph.shape
    else:
        raise ValueError(f"Invalid shape: {shape}, please choose from 'parent_child_time' or 'time_child_parent'")
    
    print(f"num_vars: {num_vars}")
    print(f"time_steps: {time_steps}")

    new_dims = num_vars * time_steps 

    # initialize new adjacency matrix with zeros
    flattened_graph = np.zeros((new_dims, new_dims))

    # for debugging, uncomment:
    # flattened_graph = np.empty((new_dims, new_dims), dtype='<U10')

    if shape == "parent_child_time":
        print("======= parent_child_time =======")
        for i in range(num_vars):
            for j in range(num_vars):
                for k in range(time_steps):
                    # DBG:
                    print(f"i: {i}, j: {j}, k: {k}")
                    print(f"graph[i, j, k]: {graph[i, j, k]}")
                    print(f"new row: {i * time_steps + k}")
                    print(f"new col: {j * time_steps + k}")
                    print("=============")

                    flattened_graph[i * time_steps + k, j * time_steps] = graph[i, j, k]

                    # Convert to numpy array if not already
                    if not isinstance(flattened_graph, np.ndarray):
                        print("Converting matrix to ndarray")
                        flattened_graph = np.array(flattened_graph)

    elif shape == "time_child_parent":
        print("======= time_child_parent =======")
        for i in range(time_steps):
            for j in range(num_vars):
                for k in range(num_vars):
                    # DBG:
                    print(f"i: {i}, j: {j}, k: {k}")
                    print(f"graph[i, j, k]: {graph[i, j, k]}")
                    print(f"new row: {k * time_steps +i}")
                    print(f"new col: {j * time_steps}")
                    print("=============")
                    flattened_graph[k * time_steps +i, j * time_steps] = graph[i, j, k]
        
    
    # populate new adjacency matrix with values from original graph
    save_path = results.results_path / Path("flattened_learned_temporal_graph.npy")

    np.save(save_path, flattened_graph)

    return flattened_graph

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
#   [["x1t0->x1t0", "x1t0->x1t1", "x1t0->x1t2", "x1t0->x1t3", "x1t0->x1t4", "x1t0->x1t5", "x1t0->x2t0", "x1t0->x2t1", "x1t0->x2t2", "x1t0->x2t3", "x1t0->x2t4", "x1t0->x2t5", "x1t0->x3t0", "x1t0->x3t1", "x1t0->x3t2", "x1t0->x3t3", "x1t0->x3t4", "x1t0->x3t5", "x1t0->x4t0", "x1t0->x4t1", "x1t0->x4t2", "x1t0->x4t3", "x1t0->x4t4", "x1t0->x4t5"],
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
#    ["x4t0->x1t0", "x4t0->x1t1", "x4t0->x1t2", "x4t0->x1t3", "x4t0->x1t4", "x4t0->x1t5", "x4t0->x2t0", "x4t0->x2t1", "x4t0->x2t2", "x4t0->x2t3", "x4t0->x2t4", "x4t0->x2t5", "x4t0->x3t0", "x4t0->x3t1", "x4t0->x3t2", "x4t0->x3t3", "x4t0->x3t4", "x4t0->x3t5", "x4t0->x4t0", "x4t0->x4t1", "x4t0->x4t2", "x4t0->x4t3", "x4t0->x4t4", "x4t0->x4t5"],
#    ["x4t1->x1t0", "x4t1->x1t1", "x4t1->x1t2", "x4t1->x1t3", "x4t1->x1t4", "x4t1->x1t5", "x4t1->x2t0", "x4t1->x2t1", "x4t1->x2t2", "x4t1->x2t3", "x4t1->x2t4", "x4t1->x2t5", "x4t1->x3t0", "x4t1->x3t1", "x4t1->x3t2", "x4t1->x3t3", "x4t1->x3t4", "x4t1->x3t5", "x4t1->x4t0", "x4t1->x4t1", "x4t1->x4t2", "x4t1->x4t3", "x4t1->x4t4", "x4t1->x4t5"],
#    ["x4t2->x1t0", "x4t2->x1t1", "x4t2->x1t2", "x4t2->x1t3", "x4t2->x1t4", "x4t2->x1t5", "x4t2->x2t0", "x4t2->x2t1", "x4t2->x2t2", "x4t2->x2t3", "x4t2->x2t4", "x4t2->x2t5", "x4t2->x3t0", "x4t2->x3t1", "x4t2->x3t2", "x4t2->x3t3", "x4t2->x3t4", "x4t2->x3t5", "x4t2->x4t0", "x4t2->x4t1", "x4t2->x4t2", "x4t2->x4t3", "x4t2->x4t4", "x4t2->x4t5"],
#    ["x4t3->x1t0", "x4t3->x1t1", "x4t3->x1t2", "x4t3->x1t3", "x4t3->x1t4", "x4t3->x1t5", "x4t3->x2t0", "x4t3->x2t1", "x4t3->x2t2", "x4t3->x2t3", "x4t3->x2t4", "x4t3->x2t5", "x4t3->x3t0", "x4t3->x3t1", "x4t3->x3t2", "x4t3->x3t3", "x4t3->x3t4", "x4t3->x3t5", "x4t3->x4t0", "x4t3->x4t1", "x4t3->x4t2", "x4t3->x4t3", "x4t3->x4t4", "x4t3->x4t5"],
#    ["x4t4->x1t0", "x4t4->x1t1", "x4t4->x1t2", "x4t4->x1t3", "x4t4->x1t4", "x4t4->x1t5", "x4t4->x2t0", "x4t4->x2t1", "x4t4->x2t2", "x4t4->x2t3", "x4t4->x2t4", "x4t4->x2t5", "x4t4->x3t0", "x4t4->x3t1", "x4t4->x3t2", "x4t4->x3t3", "x4t4->x3t4", "x4t4->x3t5", "x4t4->x4t0", "x4t4->x4t1", "x4t4->x4t2", "x4t4->x4t3", "x4t4->x4t4", "x4t4->x4t5"],
#    ["x4t5->x1t0", "x4t5->x1t1", "x4t5->x1t2", "x4t5->x1t3", "x4t5->x1t4", "x4t5->x1t5", "x4t5->x2t0", "x4t5->x2t1", "x4t5->x2t2", "x4t5->x2t3", "x4t5->x2t4", "x4t5->x2t5", "x4t5->x3t0", "x4t5->x3t1", "x4t5->x3t2", "x4t5->x3t3", "x4t5->x3t4", "x4t5->x3t5", "x4t5->x4t0", "x4t5->x4t1", "x4t5->x4t2", "x4t5->x4t3", "x4t5->x4t4", "x4t5->x4t5"]
#    ])

if __name__ == "__main__":

    test_graph_tcp = np.array([ 
        [["aa1", "ba1"], # from node a to a with a time lag of 1, from b to a with a time lag of 1 (child node is at time 0)
        ["ab1", "bb1"]], 
        [["aa2", "ba2"],
        ["ab2", "bb2"]],
        [["aa3", "ba3"],
        ["ab3", "bb4"]]
    ])

    test_graph_pct = np.array([
        [["aa1", "aa2", "aa3"],
        ["ab1", "ab2", "ab3"]],
        [["ba1", "ba2", "ba3"],
        ["bb1", "bb2", "bb3"]]
    ])

    desired_output = np.array([
        ["aa1", '', '', "ab1", '', ''],
        ["aa2", '', '', "ab2", '', ''],
        ["aa3", '', '', "ab3", '', ''],
        ["ba1", '', '', "bb1", '', ''],
        ["ba2", '', '', "bb2", '', ''],
        ["ba3", '', '', "bb3", '', '']
    ])

    flat_tcp = flatten_temporal_adjacency_graph(shape="time_child_parent", graph=test_graph_tcp)
    print("flat_tcp:\n", flat_tcp)

    flat_pct = flatten_temporal_adjacency_graph(shape="parent_child_time", graph=test_graph_pct)
    print(" =============== ")
    print("flat_pct:\n", flat_pct)



