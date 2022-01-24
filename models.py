from torch import nn
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class beta_VAE(nn.Module):
  # Simple beta_vae implementation for 2dshape, according to original paper
  def __init__(self, d, latents):
    super().__init__()
    self.latents = latents
    self.encoder = nn.Sequential(nn.Linear(64*64, 1200),
                                 nn.ReLU(),
                                 nn.Linear(1200, 1200),
                                 nn.ReLU(),
                                 nn.Linear(1200, latents*2))

    self.decoder = nn.Sequential(
                                 nn.Linear(latents, 1200),
                                 nn.Tanh(),
                                 nn.Linear(1200, 1200),
                                 nn.Tanh(),
                                 nn.Linear(1200, 1200),
                                 nn.Tanh(),
                                 nn.Linear(1200, 64*64),
                                 nn.Sigmoid())
    
  def loss(self, x, x_hat, mu,logvar, beta):
    batch_size = x.shape[0]
    x = torch.squeeze(x)
    x_hat = torch.squeeze(x_hat)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    rec_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    total_loss = (rec_loss + beta*kl_div).div(batch_size)
    return total_loss

  def re_param(self, logvar, mu):
    std = torch.exp(0.5*logvar)
    epsilon = torch.randn_like(std)
    z = mu + std*epsilon
    return z
    
  def forward(self, x):
    x = torch.squeeze(x)
    x = torch.flatten(x, start_dim=1)
    params = self.encoder(x)
    mu = params[:,:self.latents]
    logvar = params[:,self.latents:]
    z = self.re_param(logvar, mu)
    x_hat = self.decoder(z)
    return x, x_hat, mu, logvar


