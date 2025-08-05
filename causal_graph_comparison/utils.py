import numpy as np
import json
from causal_graph_comparison import CONFIGS_DIR
import logging
import pathlib
from pathlib import Path
from causal_graph_comparison import SCRATCH_DIR

class PicabuResultsPaths:
    def __init__(self, exp_params, savar_params, model_name):
        
        self.model_name = model_name
        self.path_name = f"modes_{exp_params.d_z}-difficulty_{savar_params.difficulty}-seed_{exp_params.random_seed}"
        self.results_path = Path(f"{SCRATCH_DIR}/results/SAVAR_DATA_TEST/{self.model_name}-{self.path_name}")

        print(f"Loading learned temporal graph & other results from: {self.results_path} for model: {self.model_name}")
        self.plots_path = self.results_path / Path("plots")
        self.training_results_path = self.results_path / Path("training_results")

        self.graph_path = self.plots_path / Path("graphs.npy")
        self.graph = np.load(self.graph_path)
        self.w_decoder_path = self.plots_path / Path("w_decoder.npy")
        self.w_decoder = np.load(self.w_decoder_path)
        self.w_encoder_path = self.plots_path / Path("w_encoder.npy")
        self.w_encoder = np.load(self.w_encoder_path)

        # gt from savar
        self.savar_path = SCRATCH_DIR / Path("data/SAVAR_DATA_TEST") 
        self.modes_gt_path = self.savar_path / Path(f"{self.path_name}_mode_weights.npy")
        self.modes_gt = np.load(self.modes_gt_path)

def get_timelag(node: str) -> int:
    """Extracts the time lag from a node with naming "NX, TX".

    Args:
        node: The node name in format "NX, TX".

    Returns:
        The time lag as an integer.
    """
    return int(node.split("T")[1])


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
        CONFIGS_DIR / json_config_file,
        CONFIGS_DIR / f"{json_config_file}.json",
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

