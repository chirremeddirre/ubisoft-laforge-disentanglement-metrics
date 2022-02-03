from torch.utils.data import Dataset, DataLoader
import numpy as np
import os.path
import torchvision.transforms as transforms
import torch
import pandas as pd
import src.metrics.z_diff as z_diff
import src.metrics.dci as dci
import src.metrics.dcimig as dcimig
import src.metrics.mig as mig
import src.metrics.irs as irs
import src.metrics.sap as sap
import src.metrics.z_max_var as z_max_var
import src.metrics.z_min_var as z_min_var
import src.metrics.explicitness as explicitness

import models

class dSpritesDataset(Dataset):
    def __init__(self, ds, transform = None):
        self.data = ds['imgs']
        self.latent_values = ds['latents_values']
        if transform == None:
          self.transforms = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])
        else:
          self.transforms = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        img = np.expand_dims(self.data[idx], axis=0)
        img = torch.from_numpy(img).float()
        #img = self.transforms(img) #
        latent = torch.from_numpy(self.latent_values[idx])
        latent = latent[1:] # Drop color col
        return img, latent


def dsprites_loader(path, batch_size):
   dataset = np.load(path) 
   dataset = dSpritesDataset(dataset)
   return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def load_dsprites(path):
    dataset = np.load(path)
    return dSpritesDataset(dataset)

def split_batch(batch):
    split_size = batch.size(dim=0)//2
    x1, x2 = torch.split(batch, split_size)
    return x1, x2

def save_model(model, path):
    full_path = path + "model"
    num = 0
    while os.path.exists(full_path):
        full_path = full_path + num
        num += 1
    torch.save(model.state_dict(), full_path)

def load_model(model_type, im_size, latents, path, device):
    if model_type == "beta-vae":
        model = models.beta_VAE(im_size, latents)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
    return model

def extract_metrics(metrics):
    funs = []
    if "z_diff" in metrics or "all" in metrics:
        funs.append(z_diff.z_diff)
    if "dci" in metrics or "all" in metrics:
        funs.append(dci.dci)
    if "sap" in metrics or "all" in metrics:
        funs.append(sap.sap)
    if "irs" in metrics or "all" in metrics:
        funs.append(irs.irs)
    if "mig" in metrics or "all" in metrics:
        funs.append(mig.mig)

    return funs

def init_res_table(metrics):
    table = {metric:[] for metric in metrics}
    return table

def res_table_to_tex(res_table):
    for k in res_table:
       res_table[k] = [np.mean(res_table[k]), np.var(res_table[k])]

    df = pd.DataFrame.from_dict(res_table)
    df.index = ["Mean", "Variance"]
    df.to_latex(buf="disentanglement_results")

def parse_metric_names(metric_names):
    metric_names = metric_names.lower().split(" ")
    if "all" in metric_names:
        metric_names = ["z_diff", "dci", "sap", "irs", "mig"]
    return metric_names

def eval(model, model_type, dataset, metric_names, eval_batch_size, eval_iters ,device):
    loader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=True)
    metric_names = parse_metric_names(metric_names)
    metrics = extract_metrics(metric_names)
    res_table = init_res_table(metric_names)
    n_metrics = len(metric_names)
    for i, data in enumerate(loader, 0):
        if i >= eval_iters:
            break
        x, latents = data
        x = x.to(device)
        _, _, _, z = model(x)
        for name, metric in zip(metric_names, metrics):
            score = metric(latents.detach().numpy(), z.to("cpu").detach().numpy())
            if type(score) == tuple:
                res_table[name].append(score[0])
            else:
                res_table[name].append(score)
    print(res_table["z_diff"])
    res_table_to_tex(res_table)
    return res_table
