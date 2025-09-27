from collections import OrderedDict
import torch
import torch.nn as nn

class MLP(nn.Module):
    """MLP model for time series prediction.

    Args:
        input_size: Size of input features
        output_size: Size of output features 
        num_layers: Number of hidden layers

    The model takes a sequence of flattened frames and predicts the next frame.
    Input shape: (batch_size, seq_len, 1, input_size)
    Output shape: (batch_size, 1, 1, output_size)
    """
    def __init__(self, input_size, output_size, num_layers):
        super().__init__()

        self.name = "mlp"
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        # dynamically determine size of layers
        self.layers = []
        layer_size = self.input_size
        for layer in range(self.num_layers):
            layer_size = layer_size // 2
            self.layers.append(layer_size)

        activation = nn.LeakyReLU
        module_dict = OrderedDict()

        # Create model layer by layer
        module_dict["lin0"] = nn.Linear(self.input_size, self.layers[0])

        for layer in range(self.num_layers):
            in_features = self.layers[layer]
            out_features = self.layers[layer + 1] if layer < self.num_layers - 1 else self.output_size

            module_dict[f"nonlin{layer}"] = activation()
            module_dict[f"dropout{layer+1}"] = nn.Dropout(0.5)
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

if __name__ == "__main__":

    mlp = MLP(input_size=2000, output_size=400, layers=[1000, 400, 200])
    x = torch.randn(32, 5, 1, 400)
    output = mlp(x)
    print(f"input shape: {x.shape}")
    print(f"output shape: {output.shape}")
    print(mlp)

    print("=" * 100)

