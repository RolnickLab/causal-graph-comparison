import numpy as np
from causal_graph_comparison.dim_reduction import get_mean_modes
from causal_graph_comparison import OUTPUTS_DIR
import matplotlib.pyplot as plt

def get_all_targets(targets):
    """
    This function takes in the targets array from rollouts and returns concatenated targets.

    Args:
        targets: The targets array.

    Returns:
        The all targets array.
    """
    all_targets = targets[0, :, :]
    timesteps = targets.shape[1]
    samples = targets.shape[0]

    for i in range(timesteps, samples):
        all_targets = np.vstack([all_targets, targets[i, 0, :]])

    return all_targets

def power_spectral_density(rollouts_path, num_modes):
    try:
        data = np.load(rollouts_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {rollouts_path} not found")
    
    # get rollouts
    inputs = data['inputs']
    targets = data['targets']
    outputs = data['outputs']

    # dimensionality reduction to get mean for each mode
    inputs = get_mean_modes(inputs, num_modes)
    targets = get_mean_modes(targets, num_modes)
    outputs = get_mean_modes(outputs, num_modes)

    print("outputs.shape: ", outputs.shape)

    samples = targets.shape[0]
    timesteps = targets.shape[1]

    # concat targets into one 2d array (samples * timesteps) taking first timesteps from targets[0,:,:] then first target from each sample at target[timesteps,:,:] onwards
    # might not be necessary
    all_targets = get_all_targets(targets)
    print("all_targets.shape: ", all_targets.shape)

    # get fft coefficients
    fft_coeffs_rollouts = np.fft.rfft(outputs, axis = 1)
    print("fft_coeffs_rollouts.shape: ", fft_coeffs_rollouts.shape)
    fft_coeffs_savar = np.fft.rfft(targets, axis = 1)

    print("fft_coeffs_savar.shape: ", fft_coeffs_savar.shape)

    fft_coeffs_rollouts = fft_coeffs_rollouts.mean(0)
    fft_coeffs_savar = fft_coeffs_savar.mean(0)

    print("fft_coeffs_rollouts_mean.shape: ", fft_coeffs_rollouts.shape)
    print("fft_coeffs_savar_mean.shape: ", fft_coeffs_savar.shape)

    LSD = np.abs(fft_coeffs_rollouts - fft_coeffs_savar).mean()
    print("LSD: ", LSD)

    # take just the real part of the fft coefficients
    fft_coeffs_rollouts = fft_coeffs_rollouts.real
    fft_coeffs_savar = fft_coeffs_savar.real

    return LSD, fft_coeffs_rollouts, fft_coeffs_savar


if __name__ == "__main__":
    rollouts_path = f"{OUTPUTS_DIR}/cnn-modes_4-diff_easy-seed_1-samples_1000-rollouts_20steps.npz"
    LSD, fft_coeffs_rollouts, fft_coeffs_savar, fft_coeffs_savar_all = power_spectral_density(rollouts_path, 4)

    # ==== plot first 5 samples of targets

    # plt.figure(figsize=(10, 5))

    # # Find global min and max values across targets
    # all_values = []
    # for i in range(20):
    #     all_values.extend(targets[i, :, 0])
    # y_min, y_max = min(all_values), max(all_values)

    # for i in range(5):
    #     plt.subplot(5, 1, i+1)
    #     test_targets = targets[0+i, :, 0]

    #     # print(test_targets)
        
    #     # Create x-axis points for inputs and targets/outputs, shifted by i positions
    #     x_vals = np.arange(20)
        
    #     plt.plot(x_vals, test_targets, label="Target", marker='x')
    #     plt.ylabel(f"Sample {i}")
    #     plt.xlabel("Time steps")
    #     # plt.xticks(np.arange(50))  # Extended x-axis range to accommodate shifts
    #     plt.ylim(y_min, y_max)  # Set same y-axis limits for all subplots
    #     plt.title(f"Output time series for Quadrant 0")
    #     plt.legend()

    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # ==== plot all targets
    # plt.figure(figsize=(10, 5))

    # plt.plot(all_targets[:100, 0], label="Target", marker='x')
    # plt.ylabel(f"Target")
    # plt.xlabel("Time steps")
    # plt.title(f"Output time series for Quadrant 0")
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # ==== plot fft coefficients
    # plot on log scale
    plt.figure(figsize=(10, 5))
    # plt.plot(fft_coeffs_rollouts[:, 0], label="Rollouts", marker='x')
    # plt.plot(fft_coeffs_savar[:, 0], label="Savar", marker='o')
    plt.plot(fft_coeffs_savar_all[:, 0], label="Savar All", marker='s')
    plt.ylabel(f"FFT Coefficients")
    plt.xlabel("period")
    plt.title(f"FFT Coefficients for Quadrant 0")
    plt.legend()
    plt.show()

# ==== pseudocode 
# rollouts_array_modemean --> n_samples * n_timesteps * n_modes 
# savar_array_modemean --> n_timesteps * n_modes 
# #these two arrays are the rollouts/savar after taking the mean for each mode

# fft_coeffs_rollouts = np.fft.rfft(rollouts_array_modemean, axis = 1) --> this is n_samples * n_timesteps//2 * n_modes
# fft_coeffs_savar = np.fft.rfft(savar_array_modemean, axis = 0) --> this is n_timesteps//2 * n_modes

# fft_coeffs_rollouts_sample_mean = fft_coeffs_rollouts.mean(0)

# LSD = np.abs(fft_coeffs_rollouts_sample_mean - fft_coeffs_savar).mean()

    
# fft torch or numpy

# distance between coefficients (with 20 steps = 10 coefficients) log idstance
# log spectral distance

# autoregressive 20 steps --> fft
# get coefficients from fft (10 coefficients per mode)
# for each mode take mean of coefficients over the samples
# take absolute value of difference between log of coefficient means for autoregressive & gt savar (for savar it's not mean there arent samples)
# take mean over modes
# plot power spectral density
# plot log of coefficients
# lsd - closer to 0 is better

# torch.fft 

# rmse on next step prediction

# array shape is [num_samples, timesteps, dimensions]