from collections import OrderedDict
import torch
import torch.nn as nn

# TODO: make mlp accept (batch_size, 5, 1, 1600)


class Net(nn.Module):
    def __init__(self, input_size, output_size, layers):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.name = "mlp"

        activation = nn.LeakyReLU
        num_layers = len(self.layers)
        module_dict = OrderedDict()

        # Create model layer by layer
        module_dict["lin0"] = nn.Linear(self.input_size, self.layers[0])

        for layer in range(num_layers):
            in_features = self.layers[layer]
            out_features = self.layers[layer + 1] if layer < num_layers - 1 else self.output_size

            module_dict[f"nonlin{layer}"] = activation()
            module_dict[f"lin{layer+1}"] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x):
        # Flatten input
        # print(f"x shape: {x.shape}")
        flattened_x = x.view(x.size(0), -1)
        # print(f"flattened_x shape: {flattened_x.shape}")

        output = self.model(flattened_x)
        # print(f"output shape: {output.shape}")

        # reshape output to same shape as input (batch_size, num_timesteps, 1, dimensions)
        output = output.reshape(x.size(0), 1, -1, self.output_size)
        # print(f"reshaped output shape: {output.shape}")
        return output


def train_mlp(ds_train, ds_test, model_path):
    # TODO: run train_model from trainer.py
    return

if __name__ == "__main__":

    mlp = Net(input_size=2000, output_size=400, layers=[1000, 400, 200])
    x = torch.randn(32, 5, 1, 400)
    output = mlp(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")
    print(mlp)

    print("=" * 100)

