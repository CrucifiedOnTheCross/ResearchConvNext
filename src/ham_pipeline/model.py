from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import timm

class ConvNeXtMetric(nn.Module):
    def __init__(self,name:str,n_classes=7,embedding_dim=256,dropout=.2,pretrained=True):
        super().__init__(); self.backbone=timm.create_model(name,pretrained=pretrained,num_classes=0,global_pool="avg")
        d=self.backbone.num_features
        self.projector=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Dropout(dropout),nn.Linear(d,embedding_dim))
        self.classifier=nn.Linear(embedding_dim,n_classes)
        self.proxies=nn.Parameter(torch.randn(n_classes,embedding_dim)*.02)
    def forward(self,x):
        f=self.backbone(x); z=F.normalize(self.projector(f),dim=1); return self.classifier(z),z
    def set_checkpointing(self,enabled=True):
        if hasattr(self.backbone,"set_grad_checkpointing"): self.backbone.set_grad_checkpointing(enabled)

