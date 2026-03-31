from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


MetadataType = Union[str, pd.DataFrame, Sequence[Dict[str, str]]]


class IemocapDataset(Dataset):
    def __init__(
        self,
        metadata: MetadataType,
        sample_rate: int = 16000,
        max_seconds: Optional[float] = None,
        is_train: bool = True,
        eval_clip: bool = True,
    ) -> None:
        if isinstance(metadata, str):
            if not os.path.isfile(metadata):
                raise FileNotFoundError(f"metadata_csv not found: {metadata}")
            df = pd.read_csv(metadata)
        elif isinstance(metadata, pd.DataFrame):
            df = metadata.copy()
        else:
            df = pd.DataFrame(list(metadata))

        self.records = df.to_dict("records")
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.is_train = is_train
        self.eval_clip = eval_clip

    def __len__(self) -> int:
        return len(self.records)

    def _load_wav(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav.contiguous()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        row = self.records[idx]
        wav = self._load_wav(row["path"]).float()

        if self.max_seconds is not None:
            max_len = int(self.sample_rate * self.max_seconds)
            if wav.numel() > max_len:
                if self.is_train:
                    start = random.randint(0, wav.numel() - max_len)
                    wav = wav[start : start + max_len]
                elif self.eval_clip:
                    wav = wav[:max_len]

        attention_mask = torch.ones(wav.shape[0], dtype=torch.long)
        return wav, attention_mask, int(row["label_id"]), str(row["speaker_id"]), str(
            row["utt_id"]
        )

    @staticmethod
    def collate_fn(
        batch: List[Tuple[torch.Tensor, torch.Tensor, int, str, str]]
    ) -> Dict[str, torch.Tensor | List[str]]:
        waveforms, _, labels, speaker_ids, utt_ids = zip(*batch)
        lengths = [w.shape[0] for w in waveforms]
        max_len = max(lengths)

        padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, w in enumerate(waveforms):
            L = w.shape[0]
            padded[i, :L] = w
            attention_mask[i, :L] = 1

        return {
            "input_values": padded,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_ids": list(speaker_ids),
            "utt_ids": list(utt_ids),
        }
