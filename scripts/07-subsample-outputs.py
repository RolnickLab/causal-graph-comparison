from causal_graph_comparison.subsample import get_quadrant_centres, subsample_outputs
import numpy as np

TIMESTAMP = "2025_07_08_18_53_00"
NUM_MODES = 4

mlp_output = np.load(f"../outputs/test_results-{TIMESTAMP}.npz")

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
subsampled_outputs = subsample_outputs(outputs, NUM_MODES)

print("Subsampled outputs shape: ", subsampled_outputs.shape)
print("First sample subsampled output:")
print(subsampled_outputs[0])