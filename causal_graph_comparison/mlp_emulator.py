import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Union, Tuple
    
class MLP(nn.Module):
    """
    Simple MLP emulator to train on SAVAR data.
    """
    def __init__(self, hidden_sizes: Union[List[int], int], num_input: int, num_output: int):
        """
        Args:
            hidden_sizes: Either a list of integers specifying the size of each hidden layer,
                         or a single integer specifying the size of all hidden layers
            num_input: number of input units (should be tau * num_features)
            num_output: number of output units (should be num_features)
            
        """
        super().__init__()
        self.num_input = num_input
        self.num_output = num_output
        
        # Convert single int to list if needed
        if isinstance(hidden_sizes, int):
            self.hidden_sizes = [hidden_sizes]
        else:
            self.hidden_sizes = hidden_sizes
            
        self.num_layers = len(self.hidden_sizes)

        module_dict = OrderedDict()

        # create model layer by layer
        in_features = num_input
        out_features = self.hidden_sizes[0] if self.hidden_sizes else num_output

        module_dict["lin0"] = nn.Linear(in_features, out_features)

        for layer in range(self.num_layers):
            in_features = self.hidden_sizes[layer]
            out_features = self.hidden_sizes[layer + 1] if layer < self.num_layers - 1 else num_output

            module_dict[f"nonlin{layer}"] = nn.LeakyReLU()
            module_dict[f"lin{layer+1}"] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Input tensor of shape (batch_size, tau, num_features)
            
        Returns:
            Output tensor of shape (batch_size, num_features)
        """
        batch_size = x.shape[0]
        # Flatten the time dimension
        x_flat = x.reshape(batch_size, -1)
        return self.model(x_flat)
    
    def rollout(self, initial_sequence: torch.Tensor, num_steps: int) -> torch.Tensor:
        """
        Perform autoregressive rollout prediction.
        
        Args:
            initial_sequence: Initial sequence of shape (batch_size, tau, num_features)
            num_steps: Number of future steps to predict
            
        Returns:
            Tensor of shape (batch_size, num_steps, num_features) containing the predictions
        """
        batch_size = initial_sequence.shape[0]
        predictions = []
        current_sequence = initial_sequence.clone()
        
        for _ in range(num_steps):
            # Get prediction for next step
            next_step = self.forward(current_sequence)  # (batch_size, num_features)
            predictions.append(next_step)
            
            # Update sequence by removing oldest step and adding prediction
            current_sequence = torch.cat([
                current_sequence[:, 1:],  # Remove oldest step
                next_step.unsqueeze(1)    # Add prediction as new step
            ], dim=1)
            
        # Stack all predictions
        return torch.stack(predictions, dim=1)  # (batch_size, num_steps, num_features)