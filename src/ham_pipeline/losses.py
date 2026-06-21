from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def pair_masks(y):
    same=y[:,None].eq(y[None,:]); eye=torch.eye(len(y),device=y.device,dtype=torch.bool)
    return same & ~eye, ~same

class SupCon(nn.Module):
    def __init__(self,t=.1): super().__init__(); self.t=t
    def forward(self,z,y,**_):
        sim=z@z.T/self.t; pos,neg=pair_masks(y); sim=sim-sim.max(1,keepdim=True).values.detach()
        logp=sim-torch.log((torch.exp(sim)*(~torch.eye(len(y),device=y.device,dtype=torch.bool))).sum(1,keepdim=True).clamp_min(1e-12))
        valid=pos.sum(1)>0
        return -(logp*pos).sum(1)[valid].div(pos.sum(1)[valid]).mean() if valid.any() else z.sum()*0

class BatchHardTriplet(nn.Module):
    def __init__(self,m=.2): super().__init__(); self.m=m
    def forward(self,z,y,**_):
        d=torch.cdist(z,z); pos,neg=pair_masks(y); hp=d.masked_fill(~pos,-1).max(1).values; hn=d.masked_fill(~neg,10).min(1).values
        valid=pos.any(1)&neg.any(1); return F.relu(hp[valid]-hn[valid]+self.m).mean()

class MultiSimilarity(nn.Module):
    def __init__(self,m=.5,a=2,b=50): super().__init__(); self.m=m; self.a=a; self.b=b
    def forward(self,z,y,**_):
        s=z@z.T; pos,neg=pair_masks(y)
        lp=torch.log1p((torch.exp(-self.a*(s-self.m))*pos).sum(1))/self.a
        ln=torch.log1p((torch.exp(self.b*(s-self.m))*neg).sum(1))/self.b
        return (lp+ln).mean()

class CircleLoss(nn.Module):
    def __init__(self,m=.25,g=64): super().__init__(); self.m=m; self.g=g
    def forward(self,z,y,**_):
        s=z@z.T; pos,neg=pair_masks(y); ap=(-s.detach()+1+self.m).clamp_min(0); an=(s.detach()+self.m).clamp_min(0)
        lp=-self.g*ap*(s-(1-self.m)); ln=self.g*an*(s-self.m)
        return F.softplus(torch.logsumexp(lp.masked_fill(~pos,-1e4),1)+torch.logsumexp(ln.masked_fill(~neg,-1e4),1)).mean()

class ProxyAnchor(nn.Module):
    def __init__(self,m=.1,a=32): super().__init__(); self.m=m; self.a=a
    def forward(self,z,y,proxies,**_):
        s=z@F.normalize(proxies,dim=1).T; oh=F.one_hot(y,proxies.shape[0]).bool()
        return (torch.log1p((torch.exp(-self.a*(s-self.m))*oh).sum(0)).mean()+torch.log1p((torch.exp(self.a*(s+self.m))*(~oh)).sum(0)).mean())

class AngularMargin(nn.Module):
    def __init__(self,m=.2,s=30,kind="arcface"): super().__init__(); self.m=m; self.s=s; self.kind=kind
    def forward(self,z,y,proxies,**_):
        c=(z@F.normalize(proxies,dim=1).T).clamp(-1+1e-5,1-1e-5)
        target=torch.cos(torch.acos(c)+self.m) if self.kind=="arcface" else c-self.m
        logits=torch.where(F.one_hot(y,c.shape[1]).bool(),target,c)*self.s
        return F.cross_entropy(logits,y)

class CenterLoss(nn.Module):
    def forward(self,z,y,proxies,**_): return ((z-F.normalize(proxies,dim=1)[y])**2).sum(1).mean()

class PaCo(nn.Module):
    def __init__(self,t=.1): super().__init__(); self.sup=SupCon(t); self.t=t
    def forward(self,z,y,proxies,**_): return self.sup(z,y)+F.cross_entropy(z@F.normalize(proxies,dim=1).T/self.t,y)

class BalancedContrastive(nn.Module):
    def __init__(self,t=.1): super().__init__(); self.t=t
    def forward(self,z,y,class_counts=None,**_):
        sim=z@z.T/self.t; pos,_=pair_masks(y); eye=torch.eye(len(y),device=y.device,dtype=torch.bool)
        counts=torch.bincount(y,minlength=int(y.max())+1).float().clamp_min(1); denom=(torch.exp(sim)*(~eye)/counts[y][None,:]).sum(1)
        lp=sim-torch.log(denom[:,None].clamp_min(1e-12)); valid=pos.any(1)
        return -(lp*pos).sum(1)[valid].div(pos.sum(1)[valid]).mean()

class PrototypeContrastive(nn.Module):
    def __init__(self,t=.1): super().__init__(); self.t=t
    def forward(self,z,y,proxies,**_): return F.cross_entropy(z@F.normalize(proxies,dim=1).T/self.t,y)

def build_metric_loss(name,t=.1,m=.2):
    table={"supcon":SupCon(t),"triplet":BatchHardTriplet(m),"n_pairs":SupCon(t),"multi_similarity":MultiSimilarity(),"circle":CircleLoss(),"proxy_anchor":ProxyAnchor(m),"arcface":AngularMargin(m,kind="arcface"),"cosface":AngularMargin(m,kind="cosface"),"center":CenterLoss(),"paco":PaCo(t),"bcl":BalancedContrastive(t),"sbcl":BalancedContrastive(t),"prototype":PrototypeContrastive(t),"meta_prototype":PrototypeContrastive(t)}
    if name not in table: raise ValueError(f"Unknown method {name}; choose {sorted(table)}")
    return table[name]

