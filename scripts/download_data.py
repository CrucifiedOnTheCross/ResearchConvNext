from __future__ import annotations
import argparse, shutil, subprocess, zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root", default="data/ham10000"); p.add_argument("--archive")
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True, exist_ok=True)
    if (root/"HAM10000_metadata.csv").exists(): print("Dataset already present"); return
    if a.archive:
        archives=[Path(a.archive)]
    else:
        subprocess.run(["kaggle","datasets","download","-d","kmader/skin-cancer-mnist-ham10000","-p",str(root)], check=True)
        archives=list(root.glob("*.zip"))
    if not archives: raise FileNotFoundError("HAM10000 archive was not downloaded")
    for z in archives:
        print(f"Extracting {z}")
        with zipfile.ZipFile(z) as f: f.extractall(root)
    if not (root/"HAM10000_metadata.csv").exists():
        found=next(root.rglob("HAM10000_metadata.csv"), None)
        if found:
            for item in found.parent.iterdir(): shutil.move(str(item), root/item.name)
    print("HAM10000 ready at", root.resolve())
if __name__ == "__main__": main()

