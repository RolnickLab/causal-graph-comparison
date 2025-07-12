
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from causal_graph_comparison.encoder_decoder import CNNEncoder, CNNDecoder


class VAE(nn.Module):

    def __init__(self, num_filters, z_dim, lr):
        """
        Module for training a VAE.
        Inputs:
            num_filters - Number of channels to use in a CNN encoder/decoder
            z_dim - Dimensionality of latent space
            lr - Learning rate to use for the optimizer
        """
        super().__init__()
        # self.save_hyperparameters()

        self.encode = CNNEncoder(z_dim=z_dim, num_filters=num_filters)
        self.decode = CNNDecoder(z_dim=z_dim, num_filters=num_filters)

    def forward(self, x):
        """
        The forward function to calculate the VAE-loss for a given batch of 2D spatial data.
        Inputs:
            x - Batch of 2D spatial data of shape [B,H,W].
                   The input data is converted to 4-bit, i.e. integers between 0 and 15.
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """

        # data dimensions:


        # encode the data
        mu, log_std = self.encode(x)
        # print("mu: ", mu)
        # print("log_std: ", log_std)
        # print("")

        # reparameterize the latent space (convert mean & stdev to z)
        z = self.reparameterize(mu, log_std)
        # print("z: ", z)
        # print("")

        # decode the data
        recon_x = self.decode(z)
        # print("recon_x: ", recon_x.shape)
        # print("")

        # calculate the reconstruction loss l_rec
        # use reduction='mean' to get a single value across all pixels
        # x.view reshapes the data to be 2D
        l_rec = F.cross_entropy(recon_x, x.view(-1, 28, 28), reduction='sum') / x.shape[0]
        # print("l_rec: ", l_rec)
        # print("")
        # print("l_rec: ", l_rec.shape)

        # calculate the regularization loss l_reg
        # convert l_reg from tensor.shape([4]) to scalar
        l_reg = self.KLD(mu, log_std).mean()
        # print("l_reg: ", l_reg)
        # print("")

        # calculate elbo (sum of above losses)
        elbo = l_rec + l_reg
        # print("elbo: ", elbo)
        # print("")

        # calculate bpd (bits per dimension)
        bpd = self.elbo_to_bpd(elbo, x.shape) # normalize elbo to image shape to remove sensitivity to img size
        # print("bpd: ", bpd)
        # print("======================")
        # print("")

        return l_rec, l_reg, bpd, recon_x

    def reparametrize(self, mu, log_std):
        """
        Perform the reparameterization trick to sample from a distribution with the given mean and std
        Inputs:
            mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
            std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
        Outputs:
            z - A sample of the distributions, with gradient support for both mean and std.
                The tensor should have the same shape as the mean and std input tensors.
        """
        epsilon = torch.randn_like(log_std)
        z = mu + epsilon * log_std.exp()
        return z
    
    def KLD(self, mu, log_std):
        """
        Calculate the Kullback-Leibler divergence between the given distribution and a standard normal distribution
        Inputs:
            mu - Tensor of arbitrary shape and range, denoting the mean of the distributions
            log_std - Tensor of arbitrary shape with strictly positive values. Denotes the log of the standard deviation
              of the distribution
        Outputs:
            kld - The Kullback-Leibler divergence between the given distribution and a standard normal distribution
        """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + log_std - mu.pow(2) - log_std.exp())
        return KLD

    def elbo_to_bpd(elbo, img_shape):
        """
        Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
        Inputs:
            elbo - Tensor of shape [batch_size]
            img_shape - Shape of the input images, representing [batch, channels, height, width]
        Outputs:
            bpd - The negative log likelihood in bits per dimension for the given image.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # bpd = bits per dimension

        # calculate total number of pixels
        num_pixels = img_shape[1] * img_shape[2] * img_shape[3]

        # divide by number of pixels and log(2) to get bpd
        # use np.log(2.0) to convert to base 2 (bits are in base 2)
        # eqn from the assignment pdf
        bpd = elbo * torch.log2(torch.exp(torch.tensor(1.0))) / num_pixels

        #######################
        # END OF YOUR CODE    #
        #######################
        return bpd