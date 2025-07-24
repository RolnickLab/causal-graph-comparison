from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# TODO: make lstm accept (batch_size, 5, 1, 1600)  


class Lstm(nn.Module):
    def __init__(self, input_size=1600, hidden_size=800, num_layers=3):
        super(Lstm, self).__init__()
        # Input size is flattened 40x40=1600 dimensional vector
        # Hidden size can be adjusted as needed
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, num_layers=num_layers)

        # Output layer to predict next 40x40 frame
        self.decoder = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, input, h0=None, c0=None):
        # print(self)
        # Input shape: (batch_size, 5, 1600) - 5 timesteps of 40x40 frames
        # print("input shape: ", input.shape)

        # Pass through LSTM
        # output shape: (batch_size, seq_len=5, hidden_size=800)
        if h0 is None or c0 is None:
            output, (hidden, cell) = self.rnn(input)
        else:
            output, (hidden, cell) = self.rnn(input, (h0, c0))
        # print("output shape: ", output.shape)

        # Take last output
        last_output = output[:, -1, :]
        # print("last_output shape: ", last_output.shape)

        # Decode to next frame
        next_frame = self.decoder(last_output)
        # print("next_frame shape: ", next_frame.shape)

        return next_frame, hidden, cell


class Lstm_sine(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3):
        super(Lstm_sine, self).__init__()
        # For 1D temporal data (like sine waves)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer for temporal sequence
        self.rnn = nn.LSTM(
            input_size=input_size,  # 1 for sine wave values
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Output layer to predict next value(s)
        self.fc = nn.Linear(hidden_size, 1)  # Predict 1 future value

    def forward(self, input, h0=None, c0=None):
        # Input shape: (batch_size, seq_len) - e.g., (batch, 5) for 5 timesteps
        batch_size = input.shape[0]
        seq_len = input.shape[1]

        # print("input shape: ", input.shape)

        # Pass through LSTM
        # output shape: (batch_size, seq_len, hidden_size)
        if h0 is None or c0 is None:
            output, (hidden, cell) = self.rnn(input)
        else:
            output, (hidden, cell) = self.rnn(input, (h0, c0))

        # print("output shape: ", output.shape)
        # print("hidden shape: ", hidden.shape)
        # print("cell shape: ", cell.shape)

        # Predict next value
        next_value = self.fc(output[:, -1, :])  # Shape: (batch_size, 1)

        # print("next_value shape: ", next_value.shape)

        return next_value, hidden, cell


if __name__ == "__main__":

    lstm_sine = Lstm_sine()
    input = torch.randn(32, 5, 1)
    output = lstm_sine(input)
    print(output.shape)

    print("=" * 100)

    lstm = Lstm()
    input = torch.randn(32, 5, 1600)
    output = lstm(input)
    print(output.shape)
