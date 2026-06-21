from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

def locate_images(root: Path) -> dict[str, Path]:
    return {p.stem:p for p in root.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="data/ham10000"); p.add_argument("--size",type=int,default=224); p.add_argument("--seed",type=int,default=42); p.add_argument("--cache",action="store_true")
    a=p.parse_args(); root=Path(a.root); df=pd.read_csv(root/"HAM10000_metadata.csv")
    if df.image_id.duplicated().any(): raise ValueError("Duplicate image_id")
    # Outer group split: 15% test; inner group split: 15/85 validation from remaining.
    outer=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=a.seed)
    trainval_idx,test_idx=next(outer.split(df,df.dx,groups=df.lesion_id))
    tv=df.iloc[trainval_idx]
    inner=StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=a.seed+1)
    train_rel,val_rel=next(inner.split(tv,tv.dx,groups=tv.lesion_id))
    df["split"]="test"; df.loc[tv.index[train_rel],"split"]="train"; df.loc[tv.index[val_rel],"split"]="val"
    leaked=df.groupby("lesion_id").split.nunique().gt(1)
    if leaked.any(): raise RuntimeError(f"lesion_id leakage: {leaked.sum()}")
    images=locate_images(root); missing=set(df.image_id)-images.keys()
    if missing: raise FileNotFoundError(f"Missing {len(missing)} images")
    df["source_path"]=df.image_id.map(lambda x:str(images[x].resolve()))
    if a.cache:
        cache=root/f"cache_{a.size}"; cache.mkdir(exist_ok=True)
        for image_id,src in tqdm(zip(df.image_id,df.source_path),total=len(df),desc="Caching"):
            dst=cache/f"{image_id}.jpg"
            if not dst.exists():
                with Image.open(src) as im:
                    im=ImageOps.exif_transpose(im).convert("RGB").resize((a.size,a.size),Image.Resampling.LANCZOS)
                    im.save(dst,"JPEG",quality=95,optimize=True)
        df["path"]=df.image_id.map(lambda x:str((cache/f"{x}.jpg").resolve()))
    else: df["path"]=df.source_path
    df.to_csv(root/"splits.csv",index=False)
    summary={s:{"images":len(g),"lesions":g.lesion_id.nunique(),"classes":g.dx.value_counts().to_dict()} for s,g in df.groupby("split")}
    (root/"split_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); print("No lesion leakage; wrote",root/"splits.csv")
if __name__ == "__main__": main()

