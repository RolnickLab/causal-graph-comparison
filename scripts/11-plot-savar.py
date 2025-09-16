import numpy as np 
import matplotlib.pyplot as plt
from causal_graph_comparison import *
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

modes = 4
difficulty = "med_easy"
seed = 1

experiment_name = f"modes_{modes}-diff_{difficulty}-seed_{seed}"

mlp_data_path = OUTPUTS_DIR / Path(f"mlp-{experiment_name}-samples_1000-rollouts_1steps.npz")
cnn_data_path = OUTPUTS_DIR / Path(f"cnn-{experiment_name}-samples_1000-rollouts_1steps.npz")
lstm_data_path = OUTPUTS_DIR / Path(f"lstm-{experiment_name}-samples_1000-rollouts_1steps.npz")
vae_data_path = OUTPUTS_DIR / Path(f"vae-{experiment_name}-linear-samples_1000-rollouts_1steps.npz")

mlp_data = np.load(mlp_data_path)
cnn_data = np.load(cnn_data_path)
lstm_data = np.load(lstm_data_path)
vae_data = np.load(vae_data_path)

print(mlp_data.keys())
print(mlp_data["targets"].shape)
print(mlp_data["inputs"].shape)
print(mlp_data["outputs"].shape)

print(vae_data.keys())
print(vae_data["targets"].shape)
print(vae_data["inputs"].shape)
print(vae_data["outputs"].shape)

# Check if first input sample is the same across all models
print("\nComparing first input sample across models:")
print("MLP vs CNN inputs match:", np.array_equal(mlp_data["inputs"][0], cnn_data["inputs"][0]))
print("MLP vs LSTM inputs match:", np.array_equal(mlp_data["inputs"][0], lstm_data["inputs"][0])) 
print("MLP vs VAE inputs match:", np.array_equal(mlp_data["inputs"][0], vae_data["inputs"][0]))

# Check if first target sample is the same across all models
print("\nComparing first target sample across models:")
print("MLP vs CNN targets match:", np.array_equal(mlp_data["targets"][0], cnn_data["targets"][0]))
print("MLP vs LSTM targets match:", np.array_equal(mlp_data["targets"][0], lstm_data["targets"][0]))
print("MLP vs VAE targets match:", np.array_equal(mlp_data["targets"][0], vae_data["targets"][0]))

# Plot the first 5 inputs, ground truth target, and outputs from all models
fig, axes = plt.subplots(2, 6, figsize=(24, 8))
fig.suptitle(f'SAVAR Inputs vs Outputs', fontsize=16)

# Get the first sample data
inputs = mlp_data["inputs"][0]  # Shape: (5, 400)
target = mlp_data["targets"][0]  # Shape: (1, 400)
mlp_output = mlp_data["outputs"][0]  # Shape: (1, 400)
cnn_output = cnn_data["outputs"][0]
lstm_output = lstm_data["outputs"][0]
vae_output = vae_data["outputs"][0]

# Reshape to square grids (400 = 20x20)
grid_size = int(math.sqrt(mlp_data["inputs"].shape[2]))
inputs_reshaped = inputs.reshape(5, grid_size, grid_size)
target_reshaped = target.reshape(grid_size, grid_size)
mlp_output_reshaped = mlp_output.reshape(grid_size, grid_size)
cnn_output_reshaped = cnn_output.reshape(grid_size, grid_size)
lstm_output_reshaped = lstm_output.reshape(grid_size, grid_size)
vae_output_reshaped = vae_output.reshape(grid_size, grid_size)

vae_inputs = vae_data["inputs"][0]
vae_targets = vae_data["targets"][0]

vae_inputs_reshaped = vae_inputs.reshape(5, grid_size, grid_size)
vae_targets_reshaped = vae_targets.reshape(grid_size, grid_size)

# Calculate separate color scales for inputs and outputs
inputs_data = inputs_reshaped.flatten()
outputs_data = np.concatenate([
    target_reshaped.flatten(),
    mlp_output_reshaped.flatten(),
    cnn_output_reshaped.flatten(),
    lstm_output_reshaped.flatten(),
    vae_output_reshaped.flatten()
])

inputs_and_target = np.concatenate([inputs_reshaped.flatten(), target_reshaped.flatten()])

vmin_inputs, vmax_inputs = inputs_and_target.min(), inputs_and_target.max()
vmin_outputs, vmax_outputs = -1.5, 3

# Plot first 5 inputs
for i in range(5):
    im = axes[0, i].imshow(inputs_reshaped[i], cmap='viridis', vmin=vmin_inputs, vmax=vmax_inputs)
    axes[0, i].set_title(f'Input time step {i+1}')
    axes[0, i].axis('off')

# Plot target and outputs in second row
im1 = axes[1, 0].imshow(target_reshaped, cmap='viridis', vmin=vmin_inputs, vmax=vmax_inputs)
axes[1, 0].set_title('Ground Truth Target')
axes[1, 0].axis('off')

im2 = axes[1, 1].imshow(mlp_output_reshaped, cmap='viridis', vmin=vmin_outputs, vmax=vmax_outputs)
axes[1, 1].set_title('MLP Output')
axes[1, 1].axis('off')

im3 = axes[1, 2].imshow(cnn_output_reshaped, cmap='viridis', vmin=vmin_outputs, vmax=vmax_outputs)
axes[1, 2].set_title('CNN Output')
axes[1, 2].axis('off')

im4 = axes[1, 3].imshow(lstm_output_reshaped, cmap='viridis', vmin=vmin_outputs, vmax=vmax_outputs)
axes[1, 3].set_title('LSTM Output')
axes[1, 3].axis('off')

im5 = axes[1, 4].imshow(vae_output_reshaped, cmap='viridis', vmin=vmin_outputs, vmax=vmax_outputs)
axes[1, 4].set_title('VAE Output')
axes[1, 4].axis('off')


# Add separate colorbars for inputs and outputs
axes[0, 5].axis('off')  # Hide the top right subplot
axes[1, 5].axis('off')  # Hide the bottom right subplot grid

# Colorbar for inputs (top row)
cbar_inputs = plt.colorbar(im, ax=axes[0, 5], fraction=1)
cbar_inputs.set_label('Input Values')

# Colorbar for outputs (bottom row)
cbar_outputs = plt.colorbar(im2, ax=axes[1, 5], fraction=1)
cbar_outputs.set_label('Output Values')

# plt.tight_layout()

# Save plot to outputs directory
plt.savefig(OUTPUTS_DIR / 'savar_inputs_outputs.png', dpi=300)
plt.show()

# Calculate statistics across all samples
print("\nData Statistics (All 998 samples):")
print("-" * 50)
print("Input/Target Statistics:")
all_inputs = mlp_data["inputs"].reshape(-1, 400)  # All samples flattened
all_targets = mlp_data["targets"].reshape(-1, 400)  # All samples flattened
all_inputs_and_targets = np.concatenate([all_inputs.flatten(), all_targets.flatten()])
print(f"Inputs+Target range: [{all_inputs_and_targets.min():.3f}, {all_inputs_and_targets.max():.3f}]")
print(f"Inputs+Target mean: {all_inputs_and_targets.mean():.3f}")
print(f"Inputs+Target std: {all_inputs_and_targets.std():.3f}")
print("-" * 50)
print("VAE Input/Target Statistics:")
all_vae_inputs = vae_data["inputs"].reshape(-1, 400)
all_vae_targets = vae_data["targets"].reshape(-1, 400)
all_vae_inputs_and_targets = np.concatenate([all_vae_inputs.flatten(), all_vae_targets.flatten()])
print(f"VAE Inputs+Target range: [{all_vae_inputs_and_targets.min():.3f}, {all_vae_inputs_and_targets.max():.3f}]")
print(f"VAE Inputs+Target mean: {all_vae_inputs_and_targets.mean():.3f}")
print(f"VAE Inputs+Target std: {all_vae_inputs_and_targets.std():.3f}")
print("-" * 50)

print("Model Output Statistics:")
all_mlp_outputs = mlp_data["outputs"].reshape(-1, 400)
all_cnn_outputs = cnn_data["outputs"].reshape(-1, 400)
all_lstm_outputs = lstm_data["outputs"].reshape(-1, 400)
all_vae_outputs = vae_data["outputs"].reshape(-1, 400)

print(f"MLP range: [{all_mlp_outputs.min():.3f}, {all_mlp_outputs.max():.3f}]")
print(f"MLP mean: {all_mlp_outputs.mean():.3f}")
print(f"MLP std: {all_mlp_outputs.std():.3f}")
print("-" * 50)
print(f"CNN range: [{all_cnn_outputs.min():.3f}, {all_cnn_outputs.max():.3f}]")
print(f"CNN mean: {all_cnn_outputs.mean():.3f}")
print(f"CNN std: {all_cnn_outputs.std():.3f}")
print("-" * 50)
print(f"LSTM range: [{all_lstm_outputs.min():.3f}, {all_lstm_outputs.max():.3f}]")
print(f"LSTM mean: {all_lstm_outputs.mean():.3f}")
print(f"LSTM std: {all_lstm_outputs.std():.3f}")
print("-" * 50)
print(f"VAE range: [{all_vae_outputs.min():.3f}, {all_vae_outputs.max():.3f}]")
print(f"VAE mean: {all_vae_outputs.mean():.3f}")
print(f"VAE std: {all_vae_outputs.std():.3f}")
print("-" * 50)



# Data Statistics (All 998 samples):
# --------------------------------------------------
# Input/Target Statistics:
# Inputs+Target range: [-15.702, 14.640]
# Inputs+Target mean: -0.005
# Inputs+Target std: 1.003
# --------------------------------------------------
# VAE Input/Target Statistics:
# VAE Inputs+Target range: [-14.626, 17.968]
# VAE Inputs+Target mean: 0.011
# VAE Inputs+Target std: 0.995
# --------------------------------------------------
# Model Output Statistics:
# MLP range: [-10.161, 7.804]
# MLP mean: -0.005
# MLP std: 0.533
# --------------------------------------------------
# CNN range: [-10.884, 8.531]
# CNN mean: -0.009
# CNN std: 0.587
# --------------------------------------------------
# LSTM range: [-4.941, 5.596]
# LSTM mean: -0.010
# LSTM std: 0.402
# --------------------------------------------------
# VAE range: [-8.821, 10.661]
# VAE mean: -0.002
# VAE std: 0.608