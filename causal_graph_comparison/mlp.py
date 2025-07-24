from collections import OrderedDict
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, output_size, layers):
        super().__init__()

        activation = nn.LeakyReLU
        num_layers = len(layers)
        module_dict = OrderedDict()

        # Create model layer by layer
        module_dict["lin0"] = nn.Linear(input_size, layers[0])

        for layer in range(num_layers):
            in_features = layers[layer]
            out_features = layers[layer + 1] if layer < num_layers - 1 else output_size

            module_dict[f"nonlin{layer}"] = activation()
            module_dict[f"lin{layer+1}"] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x):
        output = self.model(x)
        return output
