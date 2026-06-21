from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2

CLASSES=["akiec","bcc","bkl","df","mel","nv","vasc"]

def transforms(size:int, train:bool):
    if train:
        return v2.Compose([v2.RandomResizedCrop(size,scale=(.7,1.0)),v2.RandomHorizontalFlip(),v2.RandomVerticalFlip(),v2.RandomRotation(30),v2.ColorJitter(.15,.15,.1,.03),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])
    return v2.Compose([v2.Resize((size,size)),v2.ToImage(),v2.ToDtype(torch.float32,scale=True),v2.Normalize([.485,.456,.406],[.229,.224,.225])])

class HAMDataset(Dataset):
    def __init__(self, frame:pd.DataFrame,size:int,train=False,two_views=False):
        self.df=frame.reset_index(drop=True); self.tf=transforms(size,train); self.two_views=two_views
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        with Image.open(r.path) as im: image=im.convert("RGB")
        x=self.tf(image); y=CLASSES.index(r.dx)
        if self.two_views: x=torch.stack([x,self.tf(image)])
        return x,y,r.image_id

def make_loaders(csv_path:str,size:int,batch:int,workers:int,prefetch:int,balance:str,two_views:bool):
    df=pd.read_csv(csv_path); out={}
    kwargs=dict(num_workers=workers,pin_memory=True,persistent_workers=workers>0)
    if workers>0: kwargs["prefetch_factor"]=prefetch
    tr=df[df.split=="train"]; ds=HAMDataset(tr,size,True,two_views)
    sampler=None
    if balance=="weighted_sampler":
        counts=tr.dx.value_counts(); weights=tr.dx.map(lambda x:1.0/counts[x]).to_numpy()
        sampler=WeightedRandomSampler(torch.as_tensor(weights,dtype=torch.double),len(weights),replacement=True)
    out["train"]=DataLoader(ds,batch_size=batch,shuffle=sampler is None,sampler=sampler,drop_last=two_views,**kwargs)
    for split in ("val","test"):
        out[split]=DataLoader(HAMDataset(df[df.split==split],size),batch_size=batch*2,shuffle=False,**kwargs)
    return out,df

