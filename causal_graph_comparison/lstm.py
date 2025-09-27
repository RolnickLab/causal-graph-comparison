from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class LSTM(nn.Module):
    """LSTM model for time series prediction.

    Args:
        input_size: Size of input features
        hidden_size: Number of hidden units in LSTM layer
        num_layers: Number of LSTM layers

    The model takes a sequence of flattened frames and predicts the next frame.
    Input shape: (batch_size, seq_len, 1, input_size)
    Output shape: (batch_size, 1, 1, input_size)
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.name = "lstm"
        # Input size is flattened 40x40=1600 dimensional vector
        # Hidden size can be adjusted as needed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers, dropout=0.3)

        # Output layer to predict next 40x40 frame
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.5),  
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=0.5),  
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x, h0=None, c0=None, return_hidden=False):
        # print(self)
        # Input shape: (batch_size, 5, 1, 1600) - 5 timesteps of 40x40 frames
        # print("input shape: ", x.shape)
        # reshape x to (batch_size, seq_len, input_size)
        x_reshaped = x.squeeze()
        # print("reshaped x shape: ", x_reshaped.shape)

        # Pass through LSTM
        # output shape: (batch_size, seq_len=5, hidden_size=800)
        if h0 is None or c0 is None:
            output, (hidden, cell) = self.rnn(x_reshaped)
        else:
            output, (hidden, cell) = self.rnn(x_reshaped, (h0, c0))
        # print("output shape: ", output.shape)

        # Take last output
        last_output = output[:, -1, :]
        # print("last_output shape: ", last_output.shape)

        # Decode to next frame
        next_frame = self.decoder(last_output)
        # print("next_frame shape: ", next_frame.shape)

        # reshape next_frame to (batch_size, 1, 1, dimensions)
        next_frame = next_frame.reshape(x.size(0), 1, 1, x.size(3))
        # print("reshaped next_frame shape: ", next_frame.shape)
        # print("hidden shape: ", hidden.shape)
        # print("cell shape: ", cell.shape)

        if return_hidden:
            return next_frame, hidden, cell
        else:
            return next_frame

if __name__ == "__main__":

    lstm = LSTM(input_size=1600, hidden_size=800, num_layers=2)
    x = torch.randn(32, 5, 1, 1600)
    output = lstm(x)
    next_frame = output[0]
    hidden = output[1]
    cell = output[2]
    print("next_frame shape: ", next_frame.shape)
    print("hidden shape: ", hidden.shape)
    print("cell shape: ", cell.shape)
