import numpy as np
# 9) Plotting...
    # explore-mlp-output.ipynb
    # causal-discovery-mlp.ipynb
    # causal-discovery-groundtruth.ipynb
    # causal-discovery-mlp-1000samples.ipynb

modes = 4
model = "mlp"
difficulty = "easy"
seed = 1
steps = 20
next_step = 1
experiment_name = f"{model}-modes_{modes}-diff_{difficulty}-seed_{seed}"

# Plot time series rollout for 4 modes over 20 timesteps compared to targets for all models
# this is just illustrative example for methods illustration
rollouts_path = f"{experiment_name}-samples_1000-rollouts_{steps}steps.npz"
rollouts = np.load(rollouts_path)
rollouts = rollouts["rollouts"]
rollouts = rollouts.squeeze()
rollouts = rollouts.transpose(0, 1, 3, 2)
rollouts = rollouts.reshape(rollouts.shape[0], -1)
rollouts = rollouts.transpose(0, 1)

# Ground truth graph vs learned graph <-- a few examples
# use tigramite plotting tool for 1 example
# also plot adjacency matrices to show progresive difficulty + modes 

# Plot PSD (fft coefficients) for 4 modes over 20 timesteps compared to targets for all models

# Plot LSD vs (structural, causal) metrics for all models
# one plot per comparison, show that causal metric has best R^2 or something like that

# Plot RMSE vs (structural, causal) metrics for all models
# one plot per comparisonshow that causal metric has best R^2 or something like that

# Plot iRMSE vs (structural, causal) metrics for all models
# show that causal metric has best R^2 

# Confirm that RMSE correlates to other statistical metrics

# prev 5 time steps + 1 next time step for all models vs gt?
# actual data of savar 
# show what data looks like for savar 100 time steps 

# diagram of method

# Tables
# Hyperparameters?
