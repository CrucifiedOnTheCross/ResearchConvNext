from __future__ import annotations
import argparse, math, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np, torch, yaml
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ham_pipeline.data import CLASSES,make_loaders
from ham_pipeline.losses import build_metric_loss
from ham_pipeline.metrics import evaluate_arrays,save_plots
from ham_pipeline.model import ConvNeXtMetric
from ham_pipeline.utils import *

def deep_set(d,key,value):
    cur=d
    bits=key.split(".")
    for k in bits[:-1]: cur=cur[k]
    try: value=yaml.safe_load(value)
    except Exception: pass
    cur[bits[-1]]=value

def autocast_ctx(precision):
    dtype=torch.bfloat16 if precision=="bf16" else torch.float16
    return torch.amp.autocast("cuda",dtype=dtype,enabled=precision in ("bf16","fp16"))

def build_model(cfg,device):
    m=ConvNeXtMetric(cfg["name"],len(CLASSES),cfg["embedding_dim"],cfg["dropout"],cfg["pretrained"]); m.set_checkpointing(cfg.get("gradient_checkpointing",False)); return m.to(device)

def find_batch(cfg,device,logger):
    tc=cfg["training"]; mc=cfg["model"]; size=cfg["data"]["image_size"]
    if tc["batch_size"]!="auto": return int(tc["batch_size"])
    for b in tc["batch_candidates"]:
        try:
            m=build_model(mc,device); m.train(); x=torch.randn(b*2,3,size,size,device=device)
            if tc["channels_last"]: m=m.to(memory_format=torch.channels_last); x=x.to(memory_format=torch.channels_last)
            probe_opt=torch.optim.AdamW(m.parameters(),lr=1e-4)
            with autocast_ctx(tc["precision"]): logits,z=m(x); loss=logits.mean()+z.mean()
            loss.backward(); probe_opt.step()  # materialize AdamW states before accepting the batch
            del probe_opt,m,x,logits,z,loss; torch.cuda.empty_cache(); logger.info("Auto batch selected: %s",b); return b
        except torch.cuda.OutOfMemoryError:
            logger.info("Batch %s OOM",b); torch.cuda.empty_cache()
    raise RuntimeError("No batch candidate fits VRAM")

def run_epoch(model,loader,optimizer,metric_loss,scaler,device,cfg,train=True):
    tc=cfg["training"]; model.train(train); total=correct=n=0; optimizer.zero_grad(set_to_none=True) if train else None
    accum=max(1,math.ceil(tc["effective_batch_size"]/loader.batch_size)); start=time.perf_counter()
    context=torch.enable_grad if train else torch.inference_mode
    with context():
      for step,(x,y,_) in enumerate(loader):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        if x.ndim==5: x=x.flatten(0,1); y=y.repeat_interleave(2)
        if tc["channels_last"]: x=x.to(memory_format=torch.channels_last)
        with autocast_ctx(tc["precision"]):
            logits,z=model(x); ce=F.cross_entropy(logits,y,label_smoothing=tc["label_smoothing"]); ml=metric_loss(z,y,proxies=model._orig_mod.proxies if hasattr(model,"_orig_mod") else model.proxies); loss=(tc["ce_weight"]*ce+tc["metric_weight"]*ml)/accum
        if train:
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()
            if (step+1)%accum==0 or step+1==len(loader):
                if scaler: scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),tc["grad_clip"])
                if scaler: scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        total+=float(loss.detach())*accum*len(y); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
    return {"loss":total/n,"accuracy":correct/n,"images_per_sec":n/(time.perf_counter()-start)}

@torch.inference_mode()
def predict(model,loader,device,cfg):
    model.eval(); ls=[]; ys=[]; zs=[]; ids=[]
    for x,y,i in loader:
        x=x.to(device,non_blocking=True)
        if cfg["training"]["channels_last"]: x=x.to(memory_format=torch.channels_last)
        with autocast_ctx(cfg["training"]["precision"]): l,z=model(x)
        ls.append(l.float().cpu()); ys.append(y); zs.append(z.float().cpu()); ids.extend(i)
    return torch.cat(ls).numpy(),torch.cat(ys).numpy(),torch.cat(zs).numpy(),ids

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--set",action="append",default=[]); a=p.parse_args()
    cfg=yaml.safe_load(Path(a.config).read_text()); [deep_set(cfg,*x.split("=",1)) for x in a.set]
    seed_everything(cfg["seed"]); setup_runtime(); run=make_run_dir(cfg["output"]["root"],cfg["training"]["method"]); logger=configure_logging(run); writer=SummaryWriter(run/"tensorboard")
    (run/"config.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False)); save_json(system_info(),run/"system.json")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required")
    device=torch.device("cuda"); batch=find_batch(cfg,device,logger); loaders,frame=make_loaders(cfg["data"]["splits"],cfg["data"]["image_size"],batch,cfg["training"]["workers"],cfg["training"]["prefetch_factor"],cfg["training"]["class_balance"],True)
    model=build_model(cfg["model"],device)
    if cfg["training"]["channels_last"]: model=model.to(memory_format=torch.channels_last)
    raw=model; 
    if cfg["model"]["compile"]:
        try: model=torch.compile(model,mode="max-autotune-no-cudagraphs"); logger.info("torch.compile enabled")
        except Exception as e: logger.warning("compile unavailable: %s",e)
    tc=cfg["training"]; backbone=[p for n,p in raw.named_parameters() if n.startswith("backbone")]; head=[p for n,p in raw.named_parameters() if not n.startswith("backbone")]
    opt=torch.optim.AdamW([{"params":backbone,"lr":tc["learning_rate"]},{"params":head,"lr":tc["head_learning_rate"]}],weight_decay=tc["weight_decay"])
    warm=max(1,tc["warmup_epochs"]); epochs=tc["epochs"]
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda e:(e+1)/warm if e<warm else .5*(1+math.cos(math.pi*(e-warm)/max(1,epochs-warm))))
    scaler=torch.amp.GradScaler("cuda") if tc["precision"]=="fp16" else None; mloss=build_metric_loss(tc["method"],tc["temperature"],tc["margin"])
    best=-1; bad=0; history=[]
    for epoch in range(epochs):
        tr=run_epoch(model,loaders["train"],opt,mloss,scaler,device,cfg,True); logits,y,z,_=predict(model,loaders["val"],device,cfg); vm,_,_,_=evaluate_arrays(logits,y,z,CLASSES); sched.step()
        row={"epoch":epoch+1,**{f"train_{k}":v for k,v in tr.items()},"val_macro_f1":vm["macro_f1"],"val_balanced_accuracy":vm["balanced_accuracy"],"lr":opt.param_groups[0]["lr"]}; history.append(row); logger.info("epoch %d: %s",epoch+1,row)
        for k,v in row.items():
            if isinstance(v,(int,float)): writer.add_scalar(k,v,epoch+1)
        score=row[tc["monitor"]]
        if score>best: best=score; bad=0; torch.save({"model":raw.state_dict(),"config":cfg,"epoch":epoch+1,"score":best},run/"best.pt")
        else: bad+=1
        save_json(history,run/"history.json")
        if bad>=tc["early_stopping"]: logger.info("Early stopping"); break
    ck=torch.load(run/"best.pt",map_location=device,weights_only=False); raw.load_state_dict(ck["model"])
    calibrated_temperature=None
    for split in ("val","test"):
        logits,y,z,ids=predict(raw,loaders[split],device,cfg); metrics,probs,pred,used_temperature=evaluate_arrays(logits,y,z,CLASSES,temperature=calibrated_temperature,fit_temperature=split=="val")
        if split=="val": calibrated_temperature=used_temperature
        save_json(metrics,run/f"{split}_metrics.json")
        np.savez_compressed(run/f"{split}_predictions.npz",ids=np.array(ids),y=y,logits=logits,probs=probs,embeddings=z)
        if cfg["output"]["visualizations"]: save_plots(y,pred,z,CLASSES,run,prefix=split)
    writer.close(); logger.info("Finished. Artifacts: %s",run.resolve())
if __name__=="__main__": main()
