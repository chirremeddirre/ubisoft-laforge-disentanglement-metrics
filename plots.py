#!/usr/bin/env python

import matplotlib.pyplot as plt
import torch
import numpy as np

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
