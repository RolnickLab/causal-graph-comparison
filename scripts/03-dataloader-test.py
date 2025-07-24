from climatem.data_loader.causal_datamodule import CausalClimateDataModule 
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from causal_graph_comparison import *
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])

# get root path
print(PROJECT_ROOT)

# Create output directory for plots
output_dir = SCRATCH_DIR / "data/SAVAR_DATA_TEST/plots"
os.makedirs(output_dir, exist_ok=True)
print(f"Plots will be saved in: {output_dir}")

# TODO check that the normalization works
# check where it gest loaded in main_picabu as dataloader - look at vars
# TODO global normalization

dl = CausalClimateDataModule(
    # Required parameters for ClimateDataModule
    in_var_ids=["savar"],
    out_var_ids=["savar"],  # Same as input for SAVAR
    train_years="2015-2100",
    train_historical_years="1950-2014",
    test_years="2015-2100",  # Same as train for SAVAR
    val_split=0.1,  # 10% validation split
    seq_to_seq=True,
    channels_last=False,
    train_scenarios=["savar"],
    test_scenarios=["savar"],
    train_models="savar",
    batch_size=32,
    eval_batch_size=64,
    num_workers=0,
    pin_memory=False,
    load_train_into_mem=True,
    load_test_into_mem=True,
    verbose=True,
    seed=42,
    seq_len=12,
    data_dir="",  # Not used for SAVAR
    output_save_dir=f"{SCRATCH_DIR}/data/SAVAR_DATA_TEST",
    num_ensembles=1,
    lon=40,
    lat=40,
    num_levels=1,
    global_normalization=True,
    seasonality_removal=False,
    reload_climate_set_data=True,
    # Required parameters for CausalClimateDataModule
    tau=5,
    future_timesteps=1,
    num_months_aggregated=1,
    train_val_interval_length=100,
    # SAVAR specific parameters
    time_len=10000,
    comp_size=10,
    noise_val=0.2,
    n_per_col=2,
    difficulty="med_easy",
    seasonality=False,
)

dl.setup()



# Access the datasets directly
train_dataset = dl._data_train
val_dataset = dl._data_val

# Let's inspect the first item
x, y = val_dataset[0]
print("\nInitial shapes:")
print("Input shape:", x.shape)
print("Target shape:", y.shape)

for i in range(5):
    plt.subplot(2,5,i+1)
    img = x[i,0,:].reshape(40,40)
    plt.imshow(img)
    plt.colorbar()
# plt.show()


plt.subplot(2,5,i+1+5)
img = y[0,0,:].reshape(40,40)
plt.imshow(img)
plt.colorbar()
plt.show()

# Print dataset sizes
print(f"\nDataset sizes:")
print(f"Train set size: {len(train_dataset)}")
print(f"Val set size: {len(val_dataset)}")

print("--------------------------------")

# Create simple dataloaders for inspection if needed
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = dl.val_dataloader()
test_train_dataloader = dl.train_dataloader(accelerator)

print(f"Train dataloader size: {len(train_loader)}")
print(f"Val dataloader size: {len(val_loader)}")
print(f"Test dataloader size: {len(test_dataloader)}")
print(f"Test train dataloader size: {len(test_train_dataloader)}")

# Inspect a batch
for batch in test_dataloader:
    x, y = batch
    print("TEST DATALOADER")
    print("\nBatch shapes:")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    break

print("--------------------------------")

for batch in val_loader:
    x, y = batch
    print("VAL DATALOADER")
    print("\nBatch shapes:")
    print("Input shape:", x.shape)
    print("Target shape:", y.shape)
    break

quit()


# Print some statistics about the data
print("\nData statistics:")
print("Input mean:", x.mean().item())
print("Input std:", x.std().item())
print("Target mean:", y.mean().item())
print("Target std:", y.std().item())

# Check temporal relationship between inputs and targets
print("\nChecking temporal relationship:")
# Get a few consecutive samples
samples = []
for i in range(5):  # Get 5 consecutive samples
    x, y = train_dataset[i]
    samples.append((x, y))

# Print shapes and some values to verify temporal relationship
print("\nSample 0:")
print("Input shape:", samples[0][0].shape)
print("Target shape:", samples[0][1].shape)

# Print the actual dimensions of the data
print("\nDetailed shape information:")
for i, (x, y) in enumerate(samples):
    print(f"\nSample {i}:")
    print(f"Input dimensions: {x.shape}")
    print(f"Target dimensions: {y.shape}")
    # Print values at a specific point (center of the grid)
    center_idx = 800  # Middle of the 1600 spatial points
    print(f"Input values at center point:")
    for t in range(x.shape[0]):  # For each timestep
        print(f"  Timestep {t}: {x[t,0,center_idx].item()}")
    print(f"Target value at center point: {y[0,0,center_idx].item()}")

# Plot temporal sequence for the center point
timesteps = range(5)
input_values = []
target_values = []  # Initialize the list

# Print what we're plotting
print("\nPlotting temporal sequence:")
print("For each sample:")
print("- Blue line: Last timestep of the input sequence (5 timesteps)")
print("- Red dashed line: Target value (should be the next timestep)")

for s in samples:
    # Get the last timestep of input and the target
    input_values.append(s[0][-1,0,center_idx].item())  # Last timestep of input
    target_values.append(s[1][0,0,center_idx].item())  # Target

# Print the values we're plotting
print("\nValues being plotted:")
print("Input values:", input_values)
print("Target values:", target_values)

plt.figure(figsize=(12, 6))
plt.plot(timesteps, input_values, 'b-o', label='Last timestep of input sequence')
plt.plot(timesteps, target_values, 'r--s', label='Target (next timestep)')
plt.title('Temporal Sequence at Center Point of Grid')
plt.xlabel('Sample Index (each sample contains 5 timesteps)')
plt.ylabel('Value at center point (40x40 grid flattened)')
plt.legend()
plt.grid(True)

plot_path = output_dir / 'temporal_sequence.png'
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
print(f"\nPlot saved to: {plot_path}")
plt.close()

# Check if target is indeed the next timestep
print("\nVerifying target is next timestep:")
for i in range(len(samples)-1):
    current_target = samples[i][1][0,0,center_idx].item()
    next_input = samples[i+1][0][0,0,center_idx].item()  # First timestep of next sample
    print(f"Sample {i}:")
    print(f"Current target: {current_target}")
    print(f"Next input first timestep: {next_input}")
    print(f"Difference: {abs(current_target - next_input)}")

# Collect statistics about the data
print("\nChecking data normalization:")
print("Collecting statistics from training data...")

# Initialize lists to store statistics
all_inputs = []
all_targets = []

# Collect data from first few batches
for i, batch in enumerate(train_loader):
    if i >= 100:  # Check first 100 batches
        break
    x, y = batch
    all_inputs.append(x)
    all_targets.append(y)

# Concatenate all batches
all_inputs = torch.cat(all_inputs, dim=0)
all_targets = torch.cat(all_targets, dim=0)

# Calculate statistics
print("\nData Statistics:")
print("Input data:")
print(f"  Mean: {all_inputs.mean().item():.4f}")
print(f"  Std: {all_inputs.std().item():.4f}")
print(f"  Min: {all_inputs.min().item():.4f}")
print(f"  Max: {all_inputs.max().item():.4f}")
print(f"  Range: {all_inputs.max().item() - all_inputs.min().item():.4f}")

print("\nTarget data:")
print(f"  Mean: {all_targets.mean().item():.4f}")
print(f"  Std: {all_targets.std().item():.4f}")
print(f"  Min: {all_targets.min().item():.4f}")
print(f"  Max: {all_targets.max().item():.4f}")
print(f"  Range: {all_targets.max().item() - all_targets.min().item():.4f}")

# Plot distribution of values
plt.figure(figsize=(12, 6))
plt.hist(all_inputs.flatten().numpy(), bins=50, alpha=0.5, label='Input')
plt.hist(all_targets.flatten().numpy(), bins=50, alpha=0.5, label='Target')
plt.title('Distribution of Input and Target Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)

plot_path = output_dir / 'value_distribution.png'
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
print(f"\nDistribution plot saved to: {plot_path}")
plt.close()

# Check if data appears to be normalized
print("\nNormalization Check:")
print("If the data is normalized, we would expect:")
print("- Mean close to 0 (typically between -0.1 and 0.1)")
print("- Standard deviation close to 1 (typically between 0.8 and 1.2)")
print("- Range typically between -5 and 5 (covering ~99% of normal distribution)")

# Check if the data matches these expectations
input_mean = all_inputs.mean().item()
input_std = all_inputs.std().item()
input_range = all_inputs.max().item() - all_inputs.min().item()

print("\nInput data normalization assessment:")
print(f"Mean is {'close to 0' if abs(input_mean) < 0.1 else 'not close to 0'}")
print(f"Std is {'close to 1' if 0.8 < input_std < 1.2 else 'not close to 1'}")
print(f"Range is {'typical for normalized data' if input_range < 10 else 'larger than expected for normalized data'}")


