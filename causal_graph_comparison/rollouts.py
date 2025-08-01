import torch
import numpy as np

from causal_graph_comparison import OUTPUTS_DIR

def run_rollouts(model, device, test_loader, n_samples, rollouts, n_modes, difficulty, seed):

    data_list = []
    target_list = []
    output_list = []

    model.eval()

    with torch.no_grad():
        # get first batch
        data, target = next(iter(test_loader))

        for x in data:
            data_list.append(x.squeeze().cpu().numpy())
        data = data.to(device)

        h0 = None
        c0 = None
        for step in range(rollouts):
            print("Step: ", step)
            if model.name == "lstm":
                output, h0, c0 = model(data, h0=h0, c0=c0, return_hidden=True)
            
            elif model.name == "vae":
                target = target[:,0]
                output, _, _, _, _ = model.predict(data, target)
                output = output.unsqueeze(1)
            else:
                # for CNN & MLP
                output = model(data)

            output_list.append(output.squeeze().cpu().numpy())

            data = torch.roll(data, shifts=-1, dims=1)

            data[:, -1, :] = output

        target_list = []
        for i in range(rollouts):
            target_list.append(target[i : n_samples-1 - rollouts + i].squeeze().cpu().numpy())

    # Convert lists to arrays
    data_array = np.array(data_list)[: n_samples-1 - rollouts]
    target_array = np.array(target_list)
    output_array = np.array(output_list)[:, : n_samples-1 - rollouts]

    print("Data array shape: ", data_array.shape)
    print("Target array shape: ", target_array.shape)
    print("Output array shape: ", output_array.shape)

    np.savez(
        OUTPUTS_DIR / f"{model.name}-modes_{n_modes}-diff_{difficulty}-seed_{seed}-samples_{n_samples}-rollouts_{rollouts}steps.npz",
        inputs=data_array,
        target=target_array,
        outputs=output_array,
    )