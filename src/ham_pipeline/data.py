from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2


CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_TO_IDX = {class_name: i for i, class_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: class_name for class_name, i in CLASS_TO_IDX.items()}

CLASS_BALANCE_MODES = {
    "none",
    "weighted_sampler",
}


def seed_worker(worker_id: int) -> None:
    """
    Makes numpy/random augmentations more reproducible inside DataLoader workers.

    torch.initial_seed() is already derived from DataLoader generator.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_split_frame(df: pd.DataFrame) -> None:
    required_columns = {
        "image_id",
        "dx",
        "path",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"splits.csv is missing required columns: {sorted(missing)}"
        )

    unknown_classes = sorted(set(df["dx"]) - set(CLASSES))

    if unknown_classes:
        raise ValueError(
            f"Unknown classes in splits.csv: {unknown_classes}. "
            f"Expected: {CLASSES}"
        )

    split_values = set(df["split"])
    required_splits = {"train", "val", "test"}
    missing_splits = required_splits - split_values

    if missing_splits:
        raise ValueError(
            f"splits.csv is missing required splits: {sorted(missing_splits)}"
        )

    if "lesion_id" in df.columns:
        leaked = df.groupby("lesion_id")["split"].nunique().gt(1)
        if leaked.any():
            raise ValueError(f"lesion_id leakage detected for {int(leaked.sum())} lesions")

    missing_paths = [
        path for path in df["path"].drop_duplicates()
        if not Path(path).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Some image paths from splits.csv do not exist. "
            f"First missing examples: {missing_paths[:5]}"
        )


def transforms(size: int, train: bool):
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(
                    size,
                    scale=(0.7, 1.0),
                    ratio=(0.9, 1.1),
                ),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(30),
                v2.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.03,
                ),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    return v2.Compose(
        [
            v2.Resize((size, size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class HAMDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        size: int,
        train: bool = False,
        two_views: bool = False,
    ):
        self.df = frame.reset_index(drop=True).copy()
        self.tf = transforms(size, train)
        self.train = train
        self.two_views = two_views

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        image_path = row.path

        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")

            if self.two_views:
                x1 = self.tf(image)
                x2 = self.tf(image)
                x = torch.stack([x1, x2], dim=0)
            else:
                x = self.tf(image)

        y = CLASS_TO_IDX[row.dx]

        return x, y, row.image_id


def make_weighted_sampler(
    train_frame: pd.DataFrame,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    counts = train_frame["dx"].value_counts()

    weights = train_frame["dx"].map(
        lambda class_name: 1.0 / counts[class_name]
    ).to_numpy(dtype=np.float64)

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def make_train_sampler(
    train_frame: pd.DataFrame,
    balance: str,
    generator: torch.Generator | None = None,
):
    if balance not in CLASS_BALANCE_MODES:
        raise ValueError(
            f"Unknown class_balance={balance!r}. "
            f"Choose one of {sorted(CLASS_BALANCE_MODES)}"
        )

    if balance == "none":
        return None

    if balance == "weighted_sampler":
        return make_weighted_sampler(train_frame, generator=generator)

    raise AssertionError(f"Unhandled class balance mode: {balance}")


def make_loader_kwargs(
    workers: int,
    prefetch: int,
    generator: torch.Generator | None = None,
) -> dict:
    kwargs = {
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
        "worker_init_fn": seed_worker if workers > 0 else None,
        "generator": generator,
    }

    if workers > 0:
        kwargs["prefetch_factor"] = prefetch

    return kwargs


def make_loaders(
    csv_path: str,
    size: int,
    batch: int,
    workers: int,
    prefetch: int,
    balance: str,
    two_views: bool,
    seed: int = 42,
):
    """
    Builds train/val/test DataLoaders.

    balance:
    - none: regular shuffled train DataLoader.
    - weighted_sampler: inverse-frequency sampling over train split.

    two_views:
    - True for SupCon/BCL/PaCo-like methods.
    - False for CE/Focal/ArcFace/Center/ProxyAnchor baselines.
    """
    df = pd.read_csv(csv_path)
    check_split_frame(df)

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader_kwargs = make_loader_kwargs(
        workers=workers,
        prefetch=prefetch,
        generator=generator,
    )

    train_frame = df[df.split == "train"].reset_index(drop=True)
    val_frame = df[df.split == "val"].reset_index(drop=True)
    test_frame = df[df.split == "test"].reset_index(drop=True)

    train_dataset = HAMDataset(
        train_frame,
        size=size,
        train=True,
        two_views=two_views,
    )

    sampler = make_train_sampler(
        train_frame,
        balance=balance,
        generator=generator,
    )

    loaders = {}

    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=two_views,
        **loader_kwargs,
    )

    eval_batch = max(batch * 2, batch)

    loaders["val"] = DataLoader(
        HAMDataset(
            val_frame,
            size=size,
            train=False,
            two_views=False,
        ),
        batch_size=eval_batch,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    loaders["test"] = DataLoader(
        HAMDataset(
            test_frame,
            size=size,
            train=False,
            two_views=False,
        ),
        batch_size=eval_batch,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return loaders, df
