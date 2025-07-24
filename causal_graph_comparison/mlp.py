from collections import OrderedDict
import torch
import torch.nn as nn

# TODO: make mlp accept (batch_size, 5, 1, 1600) 

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

if __name__ == "__main__":

    mlp = Net(input_size=1600, output_size=1600, layers=[1600, 1600, 1600])
    input = torch.randn(32, 5, 1, 1600)
    output = mlp(input)
    print(f"input shape: {input.shape}")
    print(f"output shape: {output.shape}")

    print("=" * 100)