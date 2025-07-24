import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels=5, output_channels=1, image_size=40):
        """
        CNN for next-step prediction of 40x40 images.
        
        Args:
            input_channels: Number of input timesteps (default: 5)
            output_channels: Number of output timesteps (default: 1)
            image_size: Dimension of square input images (default: 40)
        """
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.image_size = image_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # After 2 pooling operations: 40 -> 20 -> 10
        conv_output_size = image_size // 4  # 40 / 4 = 10
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
        # x shape: (batch_size, 5, 40, 40)
        # x shape: (batch_size, 5, 1, 1, 1600)
        # reshape to (batch_size, 5, 40, 40)
        x = x.view(x.size(0), x.size(1), 40, 40)
        
        # Convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))  # (batch_size, 32, 40, 40)
        x = self.pool(x)           # (batch_size, 32, 20, 20)
        x = self.dropout1(x)
        
        x = F.relu(self.conv2(x))  # (batch_size, 64, 20, 20)
        x = self.pool(x)           # (batch_size, 64, 10, 10)
        x = self.dropout1(x)
        
        x = F.relu(self.conv3(x))  # (batch_size, 128, 10, 10)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 128*10*10)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))    # (batch_size, 512)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))    # (batch_size, 256)
        x = self.dropout2(x)
        x = self.fc3(x)            # (batch_size, 40*40*1)
        
        # Reshape to output format
        x = x.view(x.size(0), self.output_channels, 1, self.image_size * self.image_size)
        
        return x

# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = CNN(input_channels=5, output_channels=1, image_size=40)
    
    # Test with different batch sizes
    batch_sizes = [1, 16, 32, 64]
    
    for batch_size in batch_sizes:
        # Create dummy input: (batch_size, 5 timesteps, 40, 40)
        x = torch.randn(batch_size, 5, 1, 1600)
        
        # Forward pass
        output = model(x)
        
        print(f"Batch size {batch_size}:")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected output shape: ({batch_size}, 1, 1, 1600)")
        print()
    
    # Print model summary
    print("Model parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
