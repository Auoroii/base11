from __future__ import annotations

import os
import random
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedShuffleSplit


LABEL_MAP = {"ang": 0, "neu": 1, "sad": 2, "hap": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def load_yaml(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_metadata_df(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metadata_csv not found: {path}")
    return pd.read_csv(path)


def split_by_speaker(
    df: pd.DataFrame, fold_speaker: str, val_split: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_df = df[df["speaker_id"] == fold_speaker].reset_index(drop=True)
    train_full = df[df["speaker_id"] != fold_speaker].reset_index(drop=True)

    if val_split <= 0.0:
        return train_full, train_full.copy(), test_df

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=seed
    )
    try:
        train_idx, val_idx = next(
            splitter.split(train_full, train_full["label_id"])
        )
        train_df = train_full.iloc[train_idx].reset_index(drop=True)
        val_df = train_full.iloc[val_idx].reset_index(drop=True)
    except ValueError:
        train_df = train_full.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        val_size = max(1, int(len(train_df) * val_split))
        val_df = train_df[:val_size].reset_index(drop=True)
        train_df = train_df[val_size:].reset_index(drop=True)

    return train_df, val_df, test_df


def _stratified_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()
    test_size = float(max(0.0, min(0.99, test_size)))
    if test_size <= 0.0:
        return df.copy().reset_index(drop=True), df.iloc[:0].copy().reset_index(drop=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        train_idx, test_idx = next(splitter.split(df, df["label_id"]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
    except ValueError:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test_n = max(1, int(round(len(shuffled) * test_size)))
        test_df = shuffled.iloc[:test_n].reset_index(drop=True)
        train_df = shuffled.iloc[test_n:].reset_index(drop=True)
    return train_df, test_df


def split_train_val_test(
    df: pd.DataFrame,
    *,
    split_mode: str,
    seed: int,
    fold_speaker: Optional[str] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode = str(split_mode).strip().lower()
    val_ratio = float(max(0.0, min(0.99, val_ratio)))
    test_ratio = float(max(0.0, min(0.99, test_ratio)))

    if mode in {"loso", "speaker", "leave_one_speaker_out"}:
        if not fold_speaker:
            raise ValueError("fold_speaker is required when split_mode=loso")
        return split_by_speaker(
            df=df,
            fold_speaker=str(fold_speaker),
            val_split=val_ratio,
            seed=seed,
        )

    if mode not in {"random", "random_811", "random_stratified"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    train_val_df, test_df = _stratified_split(df, test_size=test_ratio, seed=seed)
    if val_ratio <= 0.0:
        return (
            train_val_df.reset_index(drop=True),
            train_val_df.copy().reset_index(drop=True),
            test_df.reset_index(drop=True),
        )

    remaining = max(1e-8, 1.0 - test_ratio)
    val_over_train = min(0.99, max(1e-8, val_ratio / remaining))
    train_df, val_df = _stratified_split(
        train_val_df, test_size=val_over_train, seed=seed + 1
    )
    return train_df, val_df, test_df.reset_index(drop=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
