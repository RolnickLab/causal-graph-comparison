from pathlib import Path
from dataclasses import dataclass
from causal_graph_comparison import *

@dataclass(init=False, frozen=True)
class PathsConfig:
    outputs_dir: Path = PROJECT_ROOT / "outputs" / "part-0"
    out_csv: Path = outputs_dir / "results_test.csv"

@dataclass(init=False, frozen=True)
class DataConfig:
    number_of_nodes = [4, 16, 64, 100] # [4, 16, 64, 100]
    difficulty = ["easy", "med_easy", "med_hard", "hard"]
    edge_modifications = ["delete_edges", "insert_edges", "change_edges", "randomly_modify_lag"]
    node_modifications = ["delete_nodes", "insert_nodes"]
    num_graph_seeds = 5
    num_mod_seeds = 5
