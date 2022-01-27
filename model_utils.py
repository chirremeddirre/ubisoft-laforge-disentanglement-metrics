from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import torch
import z_diff
import dci

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


def load_dsprites(path, batch_size):
   dataset = np.load(path) 
   dataset = dSpritesDataset(dataset)
   return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def split_batch(batch):
    split_size = batch.size(dim=0)//2
    x1, x2 = torch.split(batch, split_size)
    return x1, x2


def extract_metrics(metrics):
    funs = []
    metrics = list(map(str.lower, metrics))
    if "z_diff" in metrics:
        funs.append(z_diff.z_diff)
    if "dci" in metrics:
        funs.append(dci.dci)


def eval(model, model_type, dataset, metrics):
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    if model_type == "beta_vae":
      cum_score = 0
    for i, data in enumerate(loader, 0):
        x, latents = data
        x = x.to(device)
        _, _, _, z = net(x)
        score = z_diff.z_diff(z.to("cpu").detach().numpy(), latents.detach().numpy())
        cum_score += score
        print(f"Running score: {cum_score/(i+1)}")
        print(f"Latest score: {score}")
    print(f"Total score: {score}")
    return score/i
