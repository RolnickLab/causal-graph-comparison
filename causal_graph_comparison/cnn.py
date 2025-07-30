import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO reduce dims to 2x2 at some point


class CNN(nn.Module):
    def __init__(self, input_channels=5, output_channels=1, image_size=20):
        """
        CNN for next-step prediction of 40x40 images.

        Args:
            input_channels: Number of input timesteps (default: 5)
            output_channels: Number of output timesteps (default: 1)
            image_size: Dimension of square input images (default: 20)
        """
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.image_size = image_size
        self.name = "cnn"

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # kernel size should be larger
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # kernel size should be larger
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Calculate the size after convolutions and pooling
        # After 2 pooling operations: 20 -> 10 -> 5
        conv_output_size = image_size // 4  # 20 / 4 = 5
        conv_output_features = 128 * conv_output_size * conv_output_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, image_size * image_size * output_channels)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_channels, height, width)
               where input_channels = 5 (timesteps), height=width=40

        Returns:
            Output tensor of shape (batch_size, output_channels, height, width)
        """
        # x shape: (batch_size, 5, 1, 1, image_size^2)
        # reshape to (batch_size, 5, image_size, image_size)
        out = x.view(x.size(0), x.size(1), self.image_size, self.image_size)

        # Convolutional layers with ReLU and pooling
        out = F.relu(self.conv1(out))  # (batch_size, 32, image_size, image_size)
        out = self.pool(out)  # (batch_size, 32, image_size/2, image_size/2)
        out = self.dropout1(out)

        out = F.relu(self.conv2(out))  # (batch_size, 64, image_size/2, image_size/2)
        out = self.pool(out)  # (batch_size, 64, image_size/4, image_size/4)
        out = self.dropout1(out)

        out = F.relu(self.conv3(out))  # (batch_size, 128, image_size/4, image_size/4)

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)  # (batch_size, 128*(image_size/4)*(image_size/4))

        # Fully connected layers
        out = F.relu(self.fc1(out))  # (batch_size, 512)
        out = self.dropout2(out)
        out = F.relu(self.fc2(out))  # (batch_size, 256)
        out = self.dropout2(out)
        out = self.fc3(out)  # (batch_size, image_size*image_size*1)

        # Reshape to output format
        out = out.reshape(out.size(0), self.output_channels, 1, self.image_size * self.image_size)
        print("output shape: ", out.shape)

        return out


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = CNN(input_channels=5, output_channels=1, image_size=20)

    batch_size = 32
    x = torch.randn(batch_size, 5, 1, 400)

    # Forward pass
    output = model(x)

    print(f"Batch size {batch_size}:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output shape: ({batch_size}, 1, 1, 400)")
    print()
        
    # Print model summary
    print("Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
