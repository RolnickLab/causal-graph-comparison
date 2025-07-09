import torch
import torch.nn as nn
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_filters: int = 32,
                 z_dim: int = 20):
        """Encoder with a CNN network
        Inputs:
            num_input_channels - Number of input channels of the image. For
                                 MNIST, this parameter is 1
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        self.net = nn.Sequential( # sequential puts input layer by layer
            # b/w image with 1 input layer
            # learn 32 things of size 3x3 in 1st layer
            # we have 32 filters, so 32 feature detectors ex. edge, corner, etc.
            nn.Conv2d(num_input_channels, num_filters, kernel_size=3, padding=1, stride=2),
            # output size = (input_size - kernel_size + 2*padding) / stride + 1
            # (28 - 3 + 2*1) / 2 + 1 = 14
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            # 14 x 14 x 32
            nn.GELU(),
            nn.Conv2d(num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2),
            # output size = (14 - 3 + 2*1) /2 + 1 = 7
            # 7x7x64 <- 64 filters
            nn.GELU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1),
            # increase number of feature detectors
            nn.GELU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1, stride=2),
            # final output size = 4x4 with 64 filters
            nn.GELU(),
            nn.Flatten(),  # Image grid to single feature vector
        )

        # predict mean & log stdev of where the latent distribution is encoded
        self.mu = nn.Linear(num_filters * 2 * 4 * 4, z_dim)
        self.log_std = nn.Linear(num_filters * 2 * 4 * 4, z_dim)

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,C,H,W] of type long with values between 0 and 15.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float() / 15 * 2.0 - 1.0  # normalize images between -1 and 1

        # pass x through the sequential network
        x = self.net(x)
        #print("x shape: ", x.shape) = batch dim, 4 * 4 * 64

        # pass x through the linear layers to output the mean & stdev
        mean = self.mu(x)
        #print("mean: ", mean)

        log_std = self.log_std(x)
        #print("log_std: ", log_std)
        return mean, log_std


class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 16, num_filters: int = 32,
                 z_dim: int = 20):
        """Decoder with a CNN network.
        Inputs:
            num_input_channels - Number of channels of the image to
                                 reconstruct. For a 4-bit MNIST, this parameter is 16
            num_filters - Number of filters we use in the last convolutional
                          layers. Early layers might use a duplicate of it.
            z_dim - Dimensionality of latent representation z
        """
        super().__init__()

        # For an intial architecture, you can use the decoder of Tutorial 9.
        # Feel free to experiment with the architecture yourself, but the one specified here is
        # sufficient for the assignment.
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.linear = nn.Sequential(
            nn.Linear(z_dim, 2 * 4 * 4 * num_filters), # pass through sample from 20-dim latent space, convert to something that can be scaled up to img
            nn.GELU() # gaussian error linear unit, retains vals for -ve vals close to 0
        )
        self.net = nn.Sequential(
            # ouput_padding=0 so that the output size is 7x7
            # conv transpose inverse of convolution - scales up
            nn.ConvTranspose2d(2 * num_filters, 2 * num_filters, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * num_filters, num_filters, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(num_filters, num_input_channels, kernel_size=3, output_padding=1, padding=3, stride=2),

            # output 1 x 28 x 28 (1 input colour greyscale channel)
        )

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a softmax applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        x = self.linear(z)

        x = x.reshape(x.shape[0], -1, 4, 4) # reshape to (batch_size, num_filters*2, 4, 4) unflatten the vector

        x = self.net(x)
        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device