from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import math

from causal_graph_comparison import OUTPUTS_DIR
from climatem.synthetic_data.savar import dict_to_matrix

def create_intervention_samples(batch_size, intervened_modes, intervened_ts, intervention_values):
    # Create 1000 random intervention samples
    intervention_samples = [
        (
            np.random.choice(intervened_modes),
            np.random.choice(intervened_ts), 
            np.random.choice(intervention_values)
        )
        for _ in range(batch_size)
    ]
    return intervention_samples

def create_intervened_nextstep(mode_weights, datamodule, input_data, device, intervened_mode=None, intervention_value=None, intervened_t=None):
    """

    input_data are the tau timesteps that get intervened on 
    at mode intervened_mode, with value +intervention_value, at timestep intervened_t

    input_data is here of shape `self.spatial_resolution * self.time_length`.
    This is to keep the savar structure similar to the one of `self.data_field`
    """

    # print(" ======== IN INTERVENED NEXTSTEP ======== ")
    mode_weights = datamodule.savar_gt_modes_weights
    links_coeffs = datamodule.savar_links_coeffs
    gt_adj = dict_to_matrix(links_coeffs)
    tau_max = datamodule.train_val_input4mips.tau
    num_modes = datamodule.num_modes
    spatial_resolution = datamodule.spatial_resolution
    dimensions = datamodule.dimensions

    # print(f"Size of input_data: {input_data.shape}")

    # print("DBG links_coeffs: ", links_coeffs)
    # print("DBG mode_weights shape: ", mode_weights.shape)
    # print("DBG gt_adj shape: ", gt_adj.shape)

    weights = deepcopy(mode_weights.reshape(num_modes, -1))
    # print("DBG new reshapedweights shape: ", weights.shape)
    # weights_inv = np.linalg.pinv(weights)
    weights_inv = torch.Tensor(np.linalg.pinv(weights)).to(device=device)
    # print("DBG weights_inv shape: ", weights_inv.shape)
    weights = torch.Tensor(weights).to(device=device)
    # print("DBG weights shape: ", weights.shape)

    # phi = dict_to_matrix(self.links_coeffs)
    phi = torch.Tensor(gt_adj).to(device=device)
    # print("DBG phi shape: ", phi.shape)
    # data_field = deepcopy(self.data_field)
    next_step = torch.zeros(dimensions).to(device=device)
    # print("DBG next_step shape: ", next_step.shape)

    save_input_data = deepcopy(input_data)

    # print(f"Applying intervention for mode: {intervened_mode} at time: {intervened_t} with value: {intervention_value}")

    quadrant_row = intervened_mode // int(math.sqrt(num_modes))
    quadrant_col = intervened_mode % int(math.sqrt(num_modes))

    start_row = quadrant_row * spatial_resolution
    start_col = quadrant_col * spatial_resolution

    change_indices = []

    for i in range(spatial_resolution):
        for j in range(spatial_resolution):
            change_idx = (start_row + i) * int(math.sqrt(dimensions)) + (start_col + j)
            change_indices.append(change_idx)
    
    #perform intervention
    input_data[intervened_t, 0, change_indices] += intervention_value
    # print("DBG intervened_data shape: ", input_data.shape)
    # print("DBG changed indices: ", change_indices)
    # print("DBG saved_input_data: ", save_input_data[intervened_t, 0, change_indices])
    # print("===")
    # print("DBG intervened_data: ", input_data[intervened_t, 0, change_indices])

    # print("DBG input_data device: ", input_data.device)
    # print("DBG weights_inv device: ", weights_inv.device)
    # print("DBG phi device: ", phi.device)
    # print("DBG weights device: ", weights.device)

    intervened_data = input_data.squeeze()
    # print("DBG intervened_data shape: ", intervened_data.shape)
    # print("Is the data the same as the intervention data? ", np.array_equal(save_input_data, input_data))

    for i in range(tau_max):
        # print("===")
        # print(f"i: {i}")
        # print(f"DBG weights_inv: {weights_inv.shape}")
        # print(f"DBG phi[..., i]: {phi[..., i].shape}")
        # print(f"DBG weights: {weights.shape}")
        # print(f"DBG intervened_data[i, ...]: {intervened_data[i, ...].shape}")
        # print(f"DBG: next_step shape: {next_step.shape}")
        # print("===")
        next_step += weights_inv @ phi[..., i] @ weights @ intervened_data[i, ...]

    return next_step.cpu().numpy()

def intervention(model, experiment_name, test_loader, datamodule, device):

    print(f"Applying interventions to {experiment_name}...")

    save_path = OUTPUTS_DIR / f"{experiment_name}-interventions.npz"
    print("DBG save_path: ", save_path)

    # if Path(save_path).exists():
    #     print(f"Intervention already exist for {experiment_name}, skipping...")
    #     return save_path

    initial_data_list = []
    data_list = []
    output_list = []
    intervened_targets_list = []

    data, target = next(iter(test_loader))
    data = data.to(device)
    target = target.to(device)

    initial_data = np.array(data.cpu().numpy())
    initial_targets = np.array(target.cpu().numpy())

    intervened_modes = np.arange(datamodule.num_modes)
    intervened_ts = np.arange(0, datamodule.train_val_input4mips.tau)

    batch_size = data.shape[0]

    # get stdev of data
    stdev = math.ceil(data.std())
    intervention_values = [-stdev, -stdev/2, stdev/2, stdev]

    model.eval()

    with torch.no_grad():

        print("Creating intervention samples...")

        intervention_samples = create_intervention_samples(batch_size=batch_size, intervened_modes=intervened_modes, intervened_ts=intervened_ts, intervention_values=intervention_values)

        # print("DBG data shape: ", data.shape)
        # print("DBG target shape: ", target.shape)
        # print("DBG: range of data: ", data.min(), data.max())
        # print("DBG: mean of data: ", data.mean())
        # print("DBG: std of data: ", data.std())
        # save data before intervention

        intervened_targets = np.zeros((batch_size, datamodule.future_timesteps, datamodule.dimensions))

        # modify data and get next step
        for i in range(batch_size):
            intervened_mode, intervened_t, intervention_value = intervention_samples[i]

            # print("DBG intervened_mode: ", intervened_mode)
            # print("DBG intervened_t: ", intervened_t)
            # print("DBG intervention_value: ", intervention_value)
            intervened_targets[i, 0, :] = create_intervened_nextstep(mode_weights=datamodule.savar_gt_modes_weights, datamodule=datamodule, input_data=data[i], device=device, intervened_mode=intervened_mode, intervention_value=intervention_value, intervened_t=intervened_t)
        
        # save intervened input data
        for x in data:
            data_list.append(x.squeeze().cpu().numpy())

        if model.name == "vae":
            target = target[:,0]

        h0 = None
        c0 = None

        if model.name == "lstm":
            output, h0, c0 = model(data, h0=h0, c0=c0, return_hidden=True)
        
        elif model.name == "vae":
            output, _, _, _, _ = model.predict(data, target)
            output = output.unsqueeze(1)
        else:
            # for CNN & MLP
            output = model(data)

        output_list.append(output.squeeze().cpu().numpy())

    # Convert lists to arrays
    intervened_data = np.array(data_list)
    intervened_outputs = np.array(output_list)

    # switch order of dimensions to match n_samples, rollouts, dimensions
    intervened_outputs = np.moveaxis(intervened_outputs, 1, 0)

    initial_data = initial_data.squeeze()
    initial_targets = initial_targets.squeeze(axis=1)

    print("Initial data array shape: ", initial_data.shape) # inputs = n_samples, tau, dimensions
    print("Initial targets array shape: ", initial_targets.shape) # targets = n_samples, tau, dimensions
    print("Intervened data array shape: ", intervened_data.shape) # inputs = n_samples, tau, dimensions
    print("Intervened Target array shape: ", intervened_targets.shape) # targets = n_samples, rollouts, dimensions
    print("Output array shape: ", intervened_outputs.shape) # outputs = n_samples, rollouts, dimensions

    print(f"Saving interventions to {save_path}")
    np.savez(
        save_path,
        initial_inputs=initial_data,
        initial_targets=initial_targets,
        intervened_inputs=intervened_data,
        intervened_targets=intervened_targets,
        intervened_outputs=intervened_outputs,
    )
    return save_path