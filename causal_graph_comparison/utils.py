import numpy as np
import json
from causal_graph_comparison import CONFIGS_PATH
import logging
import pathlib


def get_timelag(node: str) -> int:
    """Extracts the time lag from a node with naming "NX, TX".

    Args:
        node: The node name in format "NX, TX".

    Returns:
        The time lag as an integer.
    """
    return int(node.split("T")[1])


def binarize_array(array: np.ndarray) -> np.ndarray:
    """Binarize an array.

    Args:
        array: The array to binarize.

    Returns:
        The binarized array.
    """
    return (array > 0).astype(np.int8)


def flatten_data_target(data, target, input_size, output_size):
    return data.view(-1, input_size), target.view(-1, output_size)


def get_json_config(json_config_file: str, logger=None) -> dict:
    """
    This function takes in the path, or name of the file if it can be found in the config/ folder, with of without the
    extension, and returns the values of the file in a dictionary format.

    Ex. For a file named app_config.json, directly in the config/ folder,
        the function could be called like so : `params = get_json_config('app_config')`

    Parameters
    ----------
    json_config_file
        Path to JSON config file. If config file is in the config folder,
    logger
        Logger to handle messaging, by default LOGGER

    Returns
    -------
    dict
        Dictionary of JSON configuration values
    """
    if logger is None:
        logger = logging.getLogger("utils")

    potential_paths = [
        pathlib.Path(json_config_file),
        CONFIGS_PATH / json_config_file,
        CONFIGS_PATH / f"{json_config_file}.json",
    ]

    config_filepath = None
    for path in potential_paths:
        if path.exists():
            config_filepath = path
            logger.info(f"JSON config file [{str(path)}] found.")
            break

    if not config_filepath:
        logger.error(f"JSON config file [{json_config_file}] not found.")
        return {}

    try:
        with config_filepath.open("r", encoding="UTF-8") as file:
            logger.info(f"Loading JSON config file [{config_filepath}].")
            return json.load(file)
    except json.JSONDecodeError as e:
        logger.warning(f"Error loading JSON file [{config_filepath}]: {e}")
        return {}

def flatten_temporal_adjacency_graph(graph: np.ndarray) -> np.ndarray:
    """Flatten a temporal adjacency graph.

    Args:
        graph: The temporal adjacency graph (Num_vars x Num_vars x Num_time_steps)

    Returns:
        The flattened temporal adjacency graph (Num_vars * Num_time_steps, Num_vars * Num_time_steps)
    """
    # print("Initial graph shape:", graph.shape)
    # print("Initial graph:\n", graph)
    # print("")
    # print(f"num_vars: {num_vars}")
    # print(f"time_steps: {time_steps}")

    num_vars, _, time_steps = graph.shape
    new_dims = num_vars * time_steps 

    # initialize new adjacency matrix with zeros
    flattened_graph = np.zeros((new_dims, new_dims))
    
    # populate new adjacency matrix with values from original graph

    for i in range(num_vars):
        for j in range(num_vars):
            for k in range(time_steps):
                # DBG:
                # print(f"i: {i}, j: {j}, k: {k}")
                # print(f"graph[i, j, k]: {graph[i, j, k]}")
                # print(f"new row: {i * time_steps + k}")
                # print(f"new col: {j * time_steps + k}")
                # print("=============")

                flattened_graph[i * time_steps + k, j * time_steps] = graph[i, j, k]

                # Convert to numpy array if not already
                if not isinstance(flattened_graph, np.ndarray):
                    print("Converting matrix to ndarray")
                    flattened_graph = np.array(flattened_graph)

    return flattened_graph

# temporal_test_graph = np.array([
#     [["x1t0->x1t0", "x1t1->x1t0", "x1t2->x1t0", "x1t3->x1t0", "x1t4->x1t0", "x1t5->x1t0"],
#      ["x1t0->x2t0", "x1t1->x2t0", "x1t2->x2t0", "x1t3->x2t0", "x1t4->x2t0", "x1t5->x2t0"],
#      ["x1t0->x3t0", "x1t1->x3t0", "x1t2->x3t0", "x1t3->x3t0", "x1t4->x3t0", "x1t5->x3t0"],
#      ["x1t0->x4t0", "x1t1->x4t0", "x1t2->x4t0", "x1t3->x4t0", "x1t4->x4t0", "x1t5->x4t0"]],
#     [["x2t0->x1t0", "x2t1->x1t0", "x2t2->x1t0", "x2t3->x1t0", "x2t4->x1t0", "x2t5->x1t0"],
#      ["x2t0->x2t0", "x2t1->x2t0", "x2t2->x2t0", "x2t3->x2t0", "x2t4->x2t0", "x2t5->x2t0"],
#      ["x2t0->x3t0", "x2t1->x3t0", "x2t2->x3t0", "x2t3->x3t0", "x2t4->x3t0", "x2t5->x3t0"],
#      ["x2t0->x4t0", "x2t1->x4t0", "x2t2->x4t0", "x2t3->x4t0", "x2t4->x4t0", "x2t5->x4t0"]],
#     [["x3t0->x1t0", "x3t1->x1t0", "x3t2->x1t0", "x3t3->x1t0", "x3t4->x1t0", "x3t5->x1t0"],
#      ["x3t0->x2t0", "x3t1->x2t0", "x3t2->x2t0", "x3t3->x2t0", "x3t4->x2t0", "x3t5->x2t0"],
#      ["x3t0->x3t0", "x3t1->x3t0", "x3t2->x3t0", "x3t3->x3t0", "x3t4->x3t0", "x3t5->x3t0"],
#      ["x3t0->x4t0", "x3t1->x4t0", "x3t2->x4t0", "x3t3->x4t0", "x3t4->x4t0", "x3t5->x4t0"]],
#     [["x4t0->x1t0", "x4t1->x1t0", "x4t2->x1t0", "x4t3->x1t0", "x4t4->x1t0", "x4t5->x1t0"],
#      ["x4t0->x2t0", "x4t1->x2t0", "x4t2->x2t0", "x4t3->x2t0", "x4t4->x2t0", "x4t5->x2t0"],
#      ["x4t0->x3t0", "x4t1->x3t0", "x4t2->x3t0", "x4t3->x3t0", "x4t4->x3t0", "x4t5->x3t0"],
#      ["x4t0->x4t0", "x4t1->x4t0", "x4t2->x4t0", "x4t3->x4t0", "x4t4->x4t0", "x4t5->x4t0"]],
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

