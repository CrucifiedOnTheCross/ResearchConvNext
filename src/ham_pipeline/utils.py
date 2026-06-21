from __future__ import annotations
import json, logging, os, random, subprocess
from pathlib import Path
import numpy as np
import torch

def to_builtin_types(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin_types(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        value = obj.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    return obj

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def setup_runtime() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def make_run_dir(root: str, method: str) -> Path:
    from datetime import datetime
    p = Path(root) / f"{datetime.now():%Y%m%d_%H%M%S}_{method}"
    p.mkdir(parents=True, exist_ok=False)
    return p

def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("ham"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(), logging.FileHandler(run_dir / "train.log", encoding="utf-8")):
        h.setFormatter(fmt); logger.addHandler(h)
    return logger

def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(to_builtin_types(obj), indent=2, ensure_ascii=False), encoding="utf-8")

def system_info() -> dict:
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, vram_gib=round(p.total_memory/2**30,2), capability=f"{p.major}.{p.minor}")
    try: info["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: info["git_commit"] = None
    return info
