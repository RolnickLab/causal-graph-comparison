import argparse
from climatem.model.tsdcd_latent import LatentTSDCD
import numpy as np
import torch
import wandb
from math import sqrt


from causal_graph_comparison import *  # Directory paths
from causal_graph_comparison.causal_discovery import causal_discovery
from causal_graph_comparison.picabu import train_picabu
from causal_graph_comparison.picabu_helpers import load_picabu_config
from causal_graph_comparison.savar import generate_savar_data
from causal_graph_comparison.trainer import run_trainer
from causal_graph_comparison.utils import get_json_config
from causal_graph_comparison.vae import train_vae
from causal_graph_comparison.mlp import MLP
from causal_graph_comparison.lstm import LSTM
from causal_graph_comparison.cnn import CNN
from causal_graph_comparison.graph_utils import (
    binarize_array,
    flatten_temporal_adjacency_graph,
    permute_graph,
)
from causal_graph_comparison.rollouts import get_targets, run_rollouts
from causal_graph_comparison.interventions import intervention

import pickle




# --- PART 3: INTERVENTIONS

# 1) Perturb data
# Create dataset with perturbed data --> modify synthetic_data/savar.py _create_linear() data_field

# 2) Run picabu on all trained models with perturbed data
# train picabu on all models by passing perturbed dataloader

# 3) Run inference on all trained models w/ perturbed data (script generates targets from dataloader as well)
# Apply RMSE on inference results

# 5) Apply causal / structural metrics to the learned graphs vs gt (accessed via dataloader.gt_adj)


num_modes = 4
difficulty = "easy"
seed = 1
stdev = 1
tau_max = 5
resolution = 10
model = "mlp"

savar_name = f"modes_{num_modes}-diff_{difficulty}-seed_{seed}"
experiment_name = f"{model}-{savar_name}"


# ------ Run intervention ------

print(f"Running intervention on {model}: generating next step + targets")

intervened_modes = np.arange(num_modes)
intervened_ts = np.arange(1, tau_max + 1)
intervention_values = [-stdev, -stdev/2, stdev/2, stdev]

# Create 1000 random intervention samples
n_samples = 1000
intervention_samples = [
    (
        np.random.choice(intervened_modes),
        np.random.choice(intervened_ts), 
        np.random.choice(intervention_values)
    )
    for _ in range(n_samples)
]

mlp_intervention = intervention(model=model, experiment_name=experiment_name, test_loader=test_loader, device=device, intervention_samples=intervention_samples)

