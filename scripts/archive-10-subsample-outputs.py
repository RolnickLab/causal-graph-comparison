from causal_graph_comparison.dim_reduction import get_quadrant_centres, spatial_subsample
import numpy as np
from causal_graph_comparison import OUTPUTS_DIR

TIMESTAMP = "2025_07_10_00_04_54"
NUM_MODES = 4

mlp_output = np.load(f"{OUTPUTS_DIR}/test_results-{TIMESTAMP}.npz")

inputs = mlp_output["inputs"]
targets = mlp_output["target"]
outputs = mlp_output["outputs"]

print("Inputs shape: ", inputs.shape)
print("Targets shape: ", targets.shape)
print("Outputs shape: ", outputs.shape)

# Get the centres for visualization
centres = get_quadrant_centres(outputs, NUM_MODES)
print("Centres: ", centres)

# Subsample the outputs
subsampled_outputs = spatial_subsample(outputs, NUM_MODES)

print("Subsampled outputs shape: ", subsampled_outputs.shape)
print("First sample subsampled output:")
print(subsampled_outputs[0])