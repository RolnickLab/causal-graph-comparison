from pathlib import Path
import torch
import numpy as np

from causal_graph_comparison import OUTPUTS_DIR

def run_rollouts(model, device, test_loader, n_samples, rollouts, n_modes, difficulty, seed):

    save_path = OUTPUTS_DIR / f"{model.name}-modes_{n_modes}-diff_{difficulty}-seed_{seed}-samples_{n_samples}-rollouts_{rollouts}steps.npz"

    if Path(save_path).exists():
        print(f"Rollouts already exist for {model.name}, skipping...")
        return save_path

    n_samples = n_samples - 1 # for indexing starting at 0
    data_list = []
    output_list = []

    model.eval()

    with torch.no_grad():
        # get first batch
        data, target = next(iter(test_loader))

        for x in data:
            data_list.append(x.squeeze().cpu().numpy())
        data = data.to(device)
        target = target.to(device)
        if model.name == "vae":
            target = target[:,0]

        h0 = None
        c0 = None
        for step in range(rollouts):
            if model.name == "lstm":
                output, h0, c0 = model(data, h0=h0, c0=c0, return_hidden=True)
            
            elif model.name == "vae":
                print("DBG VAE target shape: ", target.shape)
                print("DBG VAE data shape: ", data.shape)
                output, _, _, _, _ = model.predict(data, target)
                output = output.unsqueeze(1)
                print("DBG VAE output shape: ", output.shape)
                print("==========")
            else:
                # for CNN & MLP
                output = model(data)

            output_list.append(output.squeeze().cpu().numpy())

            # roll data one timestep over
            data = torch.roll(data, shifts=-1, dims=1)

            # replace last timestep with output
            data[:, -1, :, :] = output[:,0,:,:]

        # get targets
        target_list = []
        for i in range(rollouts):
            target_list.append(target[i : n_samples - rollouts + i].squeeze().cpu().numpy())

    # print("DBG target_list: ", target_list[0])
    # print("DBG length of target_list: ", len(target_list))
    # for i in range(len(target_list)):
    #     print(f"DBG target_list[{i}].shape: ", target_list[i].shape)

    # Convert lists to arrays
    data_array = np.array(data_list)[: n_samples - rollouts]
    target_array = np.array(target_list)
    output_array = np.array(output_list)[:, : n_samples - rollouts]

    # switch order of dimensions to match n_samples, rollouts, dimensions
    target_array = np.moveaxis(target_array, 1, 0)
    output_array = np.moveaxis(output_array, 1, 0)

    print("Data array shape: ", data_array.shape) # inputs = n_samples, tau, dimensions
    print("Target array shape: ", target_array.shape) # targets = n_samples, rollouts, dimensions
    print("Output array shape: ", output_array.shape) # outputs = n_samples, rollouts, dimensions

    print(f"Saving rollouts to {save_path}")
    np.savez(
        save_path,
        inputs=data_array,
        targets=target_array,
        outputs=output_array,
    )
    return save_path

def get_targets(test_loader, n_samples, rollouts, n_modes, difficulty, seed):

    save_path = OUTPUTS_DIR / f"targets-modes_{n_modes}-diff_{difficulty}-seed_{seed}-samples_{n_samples}-rollouts_{rollouts}steps.npz"

    if Path(save_path).exists():
        print(f"Targets already exist, skipping...")
        return save_path

    n_samples = n_samples - 1 # for indexing starting at 0
    target_list = []

        # get first batch
    _, target = next(iter(test_loader))

    for i in range(rollouts):
        target_list.append(target[i : n_samples - rollouts + i].squeeze().cpu().numpy())
    
    # Convert lists to array
    target_array = np.array(target_list)

    # switch order of dimensions to match n_samples, rollouts, dimensions
    target_array = np.moveaxis(target_array, 1, 0)

    print("Target array shape: ", target_array.shape) # targets = n_samples, rollouts, dimensions

    print(f"Saving rollouts to {save_path}")
    np.savez(
        save_path,
        targets=target_array
    )
    return save_path