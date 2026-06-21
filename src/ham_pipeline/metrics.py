from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,matthews_corrcoef,classification_report,confusion_matrix,roc_auc_score,average_precision_score,log_loss,brier_score_loss,silhouette_score)

def ece(probs,y,bins=15):
    conf=probs.max(1); pred=probs.argmax(1); edges=np.linspace(0,1,bins+1); score=0.
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(conf>lo)&(conf<=hi)
        if m.any(): score+=m.mean()*abs((pred[m]==y[m]).mean()-conf[m].mean())
    return float(score)

def temperature_scale(logits,y):
    l=torch.tensor(logits,dtype=torch.float64); t=torch.ones(1,dtype=torch.float64,requires_grad=True); target=torch.tensor(y)
    opt=torch.optim.LBFGS([t],lr=.05,max_iter=80,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); loss=F.cross_entropy(l/t.clamp(.05,10),target); loss.backward(); return loss
    opt.step(closure); return float(t.detach().clamp(.05,10))

def calibration(probs,y):
    onehot=np.eye(probs.shape[1])[y]
    return {"ece":ece(probs,y),"brier":float(np.mean(np.sum((probs-onehot)**2,axis=1))),"nll":float(log_loss(y,probs,labels=list(range(probs.shape[1]))))}

def evaluate_arrays(logits,y,z,classes,temperature=None,fit_temperature=False):
    probs=torch.softmax(torch.tensor(logits),1).numpy(); pred=probs.argmax(1)
    report=classification_report(y,pred,target_names=classes,output_dict=True,zero_division=0)
    out={"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),"mcc":matthews_corrcoef(y,pred),"per_class":{c:{k:report[c][k] for k in ("precision","recall","f1-score","support")} for c in classes}}
    try:
        out["roc_auc_macro_ovr"]=roc_auc_score(y,probs,multi_class="ovr",average="macro")
        out["pr_auc_macro"]=average_precision_score(np.eye(len(classes))[y],probs,average="macro")
        out["per_class_auc"]={c:{"roc_auc":roc_auc_score(y==i,probs[:,i]),"pr_auc":average_precision_score(y==i,probs[:,i])} for i,c in enumerate(classes)}
    except ValueError: pass
    malignant=np.isin(y,[classes.index(c) for c in ("mel","bcc","akiec")]); pmal=probs[:,[classes.index(c) for c in ("mel","bcc","akiec")]].sum(1); pm=pmal>=.5
    tn,fp,fn,tp=confusion_matrix(malignant,pm,labels=[False,True]).ravel()
    out["binary_malignant"]={"sensitivity":tp/max(tp+fn,1),"specificity":tn/max(tn+fp,1),"f1":f1_score(malignant,pm),"mcc":matthews_corrcoef(malignant,pm),"roc_auc":roc_auc_score(malignant,pmal),"pr_auc":average_precision_score(malignant,pmal)}
    out["calibration_before"]=calibration(probs,y)
    temp=temperature_scale(logits,y) if fit_temperature else (temperature or 1.0)
    scaled=torch.softmax(torch.tensor(logits)/temp,1).numpy(); out["temperature_from_validation"]=temp; out["calibration_after"]=calibration(scaled,y)
    if len(y)>2 and len(y)<=10000:
        out["embedding"]={"silhouette":float(silhouette_score(z,y,sample_size=min(5000,len(y)),random_state=42))}
    centers=np.stack([z[y==i].mean(0) for i in range(len(classes))]); intra=np.mean([np.linalg.norm(z[y==i]-centers[i],axis=1).mean() for i in range(len(classes))]); inter=np.linalg.norm(centers[:,None]-centers[None,:],axis=2); inter=inter[np.triu_indices(len(classes),1)].mean()
    out.setdefault("embedding",{}).update(intra_class_distance=float(intra),inter_class_distance=float(inter),inter_intra_ratio=float(inter/max(intra,1e-12)))
    return out,probs,pred,temp

def save_plots(y,pred,z,classes,outdir:Path,prefix="test"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    cm=confusion_matrix(y,pred,normalize="true"); plt.figure(figsize=(8,7)); sns.heatmap(cm,annot=True,fmt=".2f",xticklabels=classes,yticklabels=classes,cmap="Blues"); plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(outdir/f"{prefix}_confusion_matrix.png",dpi=180); plt.close()
    try:
        import umap
        emb=umap.UMAP(n_neighbors=20,min_dist=.1,metric="cosine",random_state=42).fit_transform(z)
        plt.figure(figsize=(9,7));
        for i,c in enumerate(classes): plt.scatter(*emb[y==i].T,s=8,alpha=.65,label=c)
        plt.legend(markerscale=2); plt.tight_layout(); plt.savefig(outdir/f"{prefix}_umap.png",dpi=180); plt.close(); np.save(outdir/f"{prefix}_umap.npy",emb)
        from sklearn.manifold import TSNE
        ts=TSNE(n_components=2,init="pca",learning_rate="auto",perplexity=min(30,max(5,len(y)//20)),random_state=42).fit_transform(z)
        plt.figure(figsize=(9,7))
        for i,c in enumerate(classes): plt.scatter(*ts[y==i].T,s=8,alpha=.65,label=c)
        plt.legend(markerscale=2); plt.tight_layout(); plt.savefig(outdir/f"{prefix}_tsne.png",dpi=180); plt.close(); np.save(outdir/f"{prefix}_tsne.npy",ts)
    except Exception as e: (outdir/f"{prefix}_embedding_plot_error.txt").write_text(str(e))
