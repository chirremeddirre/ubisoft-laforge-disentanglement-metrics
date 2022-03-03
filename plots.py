#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch
import numpy as np
import model_utils

def comp_color_img(x,x_hat, n=4):
  plt.figure()
  plt.box(False)
  f, axarr = plt.subplots(n,2)
  f.set_size_inches(12, 6)
  f.set(dpi=100)
  for i in range(n):
    x_py = torch.reshape(x[i], (3,64,64))
    x_hat_py = torch.reshape(x_hat[i], (3,64,64))
    x_py = x_py.permute(1,2,0).to("cpu").detach().numpy()
    x_hat_py = x_hat_py.permute(1,2,0).to("cpu").detach().numpy()
    axarr[i,0].imshow(x_hat_py)
    axarr[i,0].axis('off')
    axarr[i,1].imshow(x_py)
    axarr[i,1].axis('off')

  axarr[0][0].set_title("x\u0302")
  axarr[0][1].set_title("x")
  plt.subplots_adjust(wspace=-0.85, hspace=0.2)

  plt.savefig('comp_c.png', bbox_inches='tight',pad_inches = 0.2)
  plt.savefig('comp_c.pdf', bbox_inches='tight',pad_inches = 0.2)
  plt.show()

def comp_bw_img(x,x_hat, n=4):
  plt.figure()
  f, axarr = plt.subplots(n,2)
  f.set_size_inches(12, 6)
  f.set(dpi=100)
  for i in range(n):
    x_py = torch.reshape(x[i], (64,64))
    x_hat_py = torch.reshape(x_hat[i], (64,64))
    x_py = x_py.to("cpu").detach().numpy()
    x_hat_py = x_hat_py.to("cpu").detach().numpy()
    axarr[i,0].imshow(x_hat_py, cmap="gray")
    axarr[i,0].axis('off')
    axarr[i,1].imshow(x_py, cmap="gray")
    axarr[i,1].axis('off')

  axarr[0][0].set_title("x\u0302")
  axarr[0][1].set_title("x")
  plt.subplots_adjust(wspace=-0.85, hspace=0.2)
  plt.savefig('comp_bw.png', bbox_inches='tight',pad_inches = 0.2)
  plt.savefig('comp_bw.pdf', bbox_inches='tight',pad_inches = 0.2)
  plt.show()


def comp_img(x,x_hat,d,n=4):
  print(x.shape)
  if d > 1:
    comp_color_img(x,x_hat, n)
  else:
    comp_bw_img(x,x_hat, n)



def beta_comp_color(models, n, dataset_path="../dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", device="cpu"):
  # Generate [n, len(models)] grid of images from saved models with varying beta factors
  loader = model_utils.dsprites_loader(dataset_path, n)
  plt.figure()
  plt.box(False)
  f, axarr = plt.subplots(n,len(models))
  f.set_size_inches(n*2, len(models)*2)
  f.set(dpi=100)
  for data in loader:
    with torch.no_grad():
      x, latents = data
      x = x.to(device)
      for i, model in enumerate(models,0): # For all model paths
        net = model_utils.load_model("beta-vae", 64, 10, model, device)
        x, x_hat, _, _ = net(x)
        for j in range(n):
          if i == 0:
            x_py = torch.reshape(x[j], (3,64,64))
            x_py = x_py.permute(1,2,0).to("cpu").detach().numpy()
            axarr[j,0].imshow(x_py)
            axarr[j,0].axis('off')
            axarr[0][0].set_title("x")
          x_hat_py = torch.reshape(x_hat[j], (3,64,64))
          x_hat_py = x_hat_py.permute(1,2,0).to("cpu").detach().numpy()
          axarr[j,i+1].imshow(x_hat_py)
          axarr[j,i+1].axis('off')
        axarr[0][i+1].set_title(f"x\u0302;\u03B2 = {net.beta}")
      break
  plt.subplots_adjust(wspace=-0.85, hspace=0.2)
  plt.savefig('beta_comp_c.png', bbox_inches='tight',pad_inches = 0.2)
  plt.savefig('beta_comp_c.pdf', bbox_inches='tight',pad_inches = 0.2)
  plt.show()


def beta_comp_bw(models, n, dataset_path="../dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz", device="cpu"):
  # Generate [n, len(models)] grid of images from saved models with varying beta factors
  loader = model_utils.dsprites_loader(dataset_path, n)
  #loader = DataLoader(training_data, batch_size=n, shuffle=True)
  plt.figure()
  plt.box(False)
  f, axarr = plt.subplots(n,len(models)+1)
  f.set_size_inches(n*2, len(models)*2)
  f.set(dpi=100)
  for data in loader:
    with torch.no_grad():
      x, latents = data
      x = x.to(device)
      for i, model in enumerate(models,0): # For all model paths
        net = model_utils.load_model("beta-vae", 64, 10, model, device)
        x, x_hat, _, _ = net(x)
        for j in range(n):
          if i == 0:
            x_py = torch.reshape(x[j], (64,64))
            x_py = x_py.to("cpu").detach().numpy()
            axarr[j,0].imshow(x_py, cmap="gray")
            axarr[j,0].axis('off')
            axarr[0][0].set_title("x")
          x_hat_py = torch.reshape(x_hat[j], (64,64))
          x_hat_py = x_hat_py.to("cpu").detach().numpy()
          axarr[j,i+1].imshow(x_hat_py, cmap="gray")
          axarr[j,i+1].axis('off')
        axarr[0][i+1].set_title(f"x\u0302;\u03B2 = {net.beta}")
      break
  plt.subplots_adjust(wspace=-0.60, hspace=0.2)
  plt.savefig('beta_comp_c.png', bbox_inches='tight',pad_inches = 0.2)
  plt.savefig('beta_comp_c.pdf', bbox_inches='tight',pad_inches = 0.2)
  plt.show()
