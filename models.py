from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
#from skimage import io, transform
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

class beta_VAE(nn.Module):
  # Simple beta_vae implementation for 2dshape, according to original paper
  def __init__(self, size, d, latents, beta):
    super().__init__()
    self.beta = beta
    self.d = d
    self.size = size
    self.latents = latents
    self.encoder = nn.Sequential(nn.Linear(self.size*self.size*self.d, 1200),
                                 nn.ReLU(),
                                 nn.Linear(1200, 1200),
                                 nn.ReLU(),
                                 nn.Linear(1200, self.latents*2))

    self.decoder = nn.Sequential(
                                 nn.Linear(self.latents, 1200),
                                 nn.Tanh(),
                                 nn.Linear(1200, 1200),
                                 nn.Tanh(),
                                 nn.Linear(1200, 1200),
                                 nn.Tanh(),
                                 nn.Linear(1200, self.size*self.size*self.d),
                                 nn.Sigmoid())

  def loss(self, x, x_hat, mu,logvar):
    batch_size = x.shape[0]
    x = torch.squeeze(x)
    x_hat = torch.squeeze(x_hat)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    rec_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    total_loss = (rec_loss + self.beta*kl_div).div(batch_size)
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

  def _train(self, n_epoch, loader, device):
    import torch.optim as optim
    optimizer = torch.optim.Adagrad(self.parameters(),
                             lr = 0.01,
                             weight_decay = 1e-8)
    for epoch in range(n_epoch):  # loop over the dataset multiple times
      running_loss = 0.0
      for i, data in enumerate(loader, 0):
          data, _ = data
          optimizer.zero_grad()
          x = data.to(device)
          x, x_hat, mu, logvar = self.forward(x)
          loss = self.loss(x, x_hat, mu, logvar)
          loss.backward()
          optimizer.step()

          # print statistics
          running_loss += loss.item()
          if i % 200 == 0:    # print every 200 mini-batches
              print('[%d, %5d] loss: %.6f' %
                    (epoch + 1, i + 1, running_loss / 200))
              running_loss = 0.0
    return self, x_hat, x
