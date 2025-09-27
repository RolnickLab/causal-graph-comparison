# import numpy file
import numpy as np
import pathlib
import torch

from causal_graph_comparison.rollouts import run_rollouts

home_dir = pathlib.Path.home()
path_dir = home_dir.parent
path_dir = path_dir.parent

# load numpy file
# file path from home directory

models = ["mlp"] #["mlp", "lstm", "cnn", "vae"]
difficulties = ["hard"] #["easy", "med_easy", "med_hard", "hard"]
modes = [64] #[4, 16, 64]
seed = 1 

files = []

for mode in modes:
    for difficulty in difficulties:
        for model in models:
            file_path = path_dir / "Volumes" / "GEN-Z/outputs" / f"{model}-modes_{mode}-diff_{difficulty}-seed_{seed}-samples_1000-rollouts_20steps.npz"
            files.append(file_path)

for file in files:
    if file.exists():
        data = np.load(file)
        has_nans = np.isnan(data["outputs"]).any()
        if has_nans:
            print(f"{file} has nans: {has_nans}")
            trouble_file = file

data = np.load(trouble_file)

# print data
print(data["outputs"].shape)
print(data["outputs"])
