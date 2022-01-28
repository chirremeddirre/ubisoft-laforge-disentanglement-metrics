from torch.utils.data import Dataset, DataLoader
import numpy as np
import os.path
import torchvision.transforms as transforms
import torch
import src.metrics.z_diff as z_diff
import src.metrics.dci as dci


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
    torch.save(model, full_path)


def extract_metrics(metrics):
    funs = []
    if "z_diff" in metrics or "all" in metrics:
        funs.append(z_diff.z_diff)
    if "dci" in metrics or "all" in metrics:
        funs.append(dci.dci)
    return funs

def init_res_table(metrics):
    table = {metric:0 for metric in metrics}
    return table

def parse_metric_names(metric_names):
    metric_names = metric_names.lower().split(" ")
    if "all" in metric_names:
        metric_names = ["z_diff", "dci"]
    return metric_names

def eval(model, model_type, dataset, metric_names, device):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    cum_score = 0
    score = 0
    metric_names = parse_metric_names(metric_names)
    metrics = extract_metrics(metric_names)
    res_table = init_res_table(metric_names)
    print(res_table)
    n_metrics = len(metric_names)
    for i, data in enumerate(loader, 0):
        x, latents = data
        x = x.to(device)
        _, _, _, z = model(x)
        for i, metric in enumerate(metrics, 0):
            score = metric(z.to("cpu").detach().numpy(), latents.detach().numpy())
            if type(score) == tuple:
                res_table[metric_names[i%n_metrics]] += score[0]
            else:
                res_table[metric_names[i%n_metrics]] += score
        #print(f"Running score: {cum_score/(i+1)}")
        #print(f"Latest score: {score}")
    #print(f"Total score: {score}")
    return score/i
