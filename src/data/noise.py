from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Sequence

import torch
import torchaudio
from torch.utils.data import Dataset


def _rand_choice(rng: random.Random, items: Sequence[str]) -> str:
    if hasattr(rng, "choice"):
        return rng.choice(items)
    return random.choice(items)


def _randint(rng: random.Random, low: int, high: int) -> int:
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high + 1))
    return int(rng.randint(low, high))


class NoiseAdder:
    def __init__(
        self,
        noise_root: str,
        noise_types: Sequence[str],
        snr_levels: Sequence[int],
        sample_rate: int = 16000,
        clip: bool = True,
    ) -> None:
        if not os.path.isdir(noise_root):
            raise FileNotFoundError(f"noise_root not found: {noise_root}")
        self.noise_root = noise_root
        self.noise_types = list(noise_types)
        self.snr_levels = list(snr_levels)
        self.sample_rate = sample_rate
        self.clip = clip
        self.noise_files = self._index_noise_files()
        self._noise_cache: Dict[str, torch.Tensor] = {}

    def _index_noise_files(self) -> Dict[str, List[str]]:
        all_wavs: List[str] = []
        for root, _, files in os.walk(self.noise_root):
            for name in files:
                if name.lower().endswith(".wav"):
                    all_wavs.append(os.path.join(root, name))
        all_wavs.sort()

        if not all_wavs:
            raise FileNotFoundError(f"No wav files found in {self.noise_root}")

        noise_files: Dict[str, List[str]] = {}
        for ntype in self.noise_types:
            matched = [p for p in all_wavs if ntype.lower() in os.path.basename(p).lower()]
            if not matched:
                raise FileNotFoundError(
                    f"No wav files found for noise_type={ntype} in {self.noise_root}"
                )
            noise_files[ntype] = sorted(matched)
        return noise_files

    def _load_noise(self, path: str) -> torch.Tensor:
        cached = self._noise_cache.get(path)
        if cached is not None:
            return cached

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.contiguous()
        self._noise_cache[path] = wav
        return wav

    def ensure_length(self, wav: torch.Tensor, target_length: int) -> torch.Tensor:
        target_length = int(target_length)
        if wav.numel() == target_length:
            return wav
        if wav.numel() > target_length:
            return wav[:target_length]
        pad = torch.zeros(target_length - wav.numel(), dtype=wav.dtype)
        return torch.cat([wav, pad], dim=0)

    def _extract_noise_segment(
        self, noise: torch.Tensor, target_length: int, noise_offset: int
    ) -> torch.Tensor:
        target_length = int(target_length)
        noise_offset = int(max(0, noise_offset))

        if noise.numel() >= target_length:
            max_start = noise.numel() - target_length
            start = min(noise_offset, max_start)
            return noise[start : start + target_length]

        if noise.numel() <= 0:
            return torch.zeros(target_length, dtype=noise.dtype)

        start = noise_offset % noise.numel()
        repeats = math.ceil((start + target_length) / noise.numel())
        tiled = noise.repeat(repeats)
        return tiled[start : start + target_length]

    def _mix_clean_with_noise(
        self, clean_wav: torch.Tensor, noise_segment: torch.Tensor, snr_db: float
    ) -> torch.Tensor:
        sig_power = clean_wav.pow(2).mean().clamp_min(1e-12)
        noise_power = noise_segment.pow(2).mean().clamp_min(1e-12)
        scale = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10))))
        noisy = clean_wav + noise_segment * scale
        if self.clip:
            noisy = torch.clamp(noisy, -1.0, 1.0)
        return noisy

    def mix_with_fixed_noise(
        self,
        clean_wav: torch.Tensor,
        snr_db: float,
        noise_path: str,
        noise_offset: int = 0,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        clean_len = int(target_length) if target_length is not None else int(clean_wav.shape[0])
        clean = self.ensure_length(clean_wav, clean_len)
        noise = self._load_noise(noise_path)
        noise_segment = self._extract_noise_segment(noise, clean_len, noise_offset)
        return self._mix_clean_with_noise(clean, noise_segment, float(snr_db))

    def __call__(
        self,
        clean_wav: torch.Tensor,
        snr_db: float,
        noise_type: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = random

        if noise_type is None:
            noise_type = _rand_choice(rng, self.noise_types)

        noise_paths = self.noise_files[noise_type]
        noise_path = _rand_choice(rng, noise_paths)
        noise = self._load_noise(noise_path)

        clean_len = clean_wav.shape[0]
        max_start = max(0, noise.numel() - clean_len)
        start = _randint(rng, 0, max_start) if max_start > 0 else 0
        noise_segment = self._extract_noise_segment(noise, clean_len, start)
        return self._mix_clean_with_noise(clean_wav, noise_segment, float(snr_db))


class NoisyPairDataset(Dataset):
    def __init__(
        self,
        clean_dataset: Dataset,
        noise_adder: NoiseAdder,
        is_train: bool = True,
        fixed_noise_type: Optional[str] = None,
        fixed_snr: Optional[float] = None,
        rng: Optional[random.Random] = None,
        deterministic_eval: bool = False,
        eval_seed: int = 0,
    ) -> None:
        self.clean_dataset = clean_dataset
        self.noise_adder = noise_adder
        self.is_train = is_train
        self.fixed_noise_type = fixed_noise_type
        self.fixed_snr = fixed_snr
        self.rng = rng or random.Random()
        self.deterministic_eval = deterministic_eval
        self.eval_seed = int(eval_seed)
        self.noise_to_idx = {
            str(ntype): idx for idx, ntype in enumerate(self.noise_adder.noise_types)
        }
        self.snr_to_idx = {
            float(snr): idx for idx, snr in enumerate(self.noise_adder.snr_levels)
        }

    def __len__(self) -> int:
        return len(self.clean_dataset)

    def __getitem__(self, idx: int):
        clean_wav, attention_mask, label, speaker_id, utt_id = self.clean_dataset[idx]
        if self.is_train:
            noise_type = _rand_choice(self.rng, self.noise_adder.noise_types)
            snr_db = _rand_choice(self.rng, self.noise_adder.snr_levels)
            sample_rng = self.rng
        else:
            noise_type = self.fixed_noise_type or self.noise_adder.noise_types[0]
            snr_db = self.fixed_snr if self.fixed_snr is not None else self.noise_adder.snr_levels[0]
            if self.deterministic_eval:
                sample_rng = random.Random(
                    f"{self.eval_seed}|{idx}|{noise_type}|{float(snr_db):.4f}"
                )
            else:
                sample_rng = self.rng

        noisy_wav = self.noise_adder(clean_wav, snr_db=snr_db, noise_type=noise_type, rng=sample_rng)
        return (
            clean_wav,
            noisy_wav,
            attention_mask,
            label,
            speaker_id,
            utt_id,
            noise_type,
            float(snr_db),
            int(self.noise_to_idx[str(noise_type)]),
            int(self.snr_to_idx[float(snr_db)]),
        )


class ManifestNoisyPairDataset(Dataset):
    def __init__(
        self,
        clean_dataset: Dataset,
        noise_adder: NoiseAdder,
        manifest_entries: Sequence[Dict],
    ) -> None:
        self.clean_dataset = clean_dataset
        self.noise_adder = noise_adder
        self.manifest_entries = list(manifest_entries)
        self.noise_to_idx = {
            str(ntype): idx for idx, ntype in enumerate(self.noise_adder.noise_types)
        }
        self.snr_to_idx = {
            float(snr): idx for idx, snr in enumerate(self.noise_adder.snr_levels)
        }

    def __len__(self) -> int:
        return len(self.manifest_entries)

    def __getitem__(self, idx: int):
        entry = self.manifest_entries[idx]
        clean_index = int(entry["clean_index"])
        clean_wav, _, label, speaker_id, utt_id = self.clean_dataset[clean_index]

        target_length = int(entry.get("target_length", clean_wav.shape[0]))
        clean_wav = self.noise_adder.ensure_length(clean_wav, target_length)
        attention_mask = torch.ones(clean_wav.shape[0], dtype=torch.long)

        noise_type = str(entry["noise_type"])
        snr_db = float(entry["snr_db"])
        noise_path = str(entry["noise_source_path"])
        noise_offset = int(entry.get("noise_offset", 0))

        noisy_wav = self.noise_adder.mix_with_fixed_noise(
            clean_wav=clean_wav,
            snr_db=snr_db,
            noise_path=noise_path,
            noise_offset=noise_offset,
            target_length=target_length,
        )
        return (
            clean_wav,
            noisy_wav,
            attention_mask,
            label,
            speaker_id,
            utt_id,
            noise_type,
            snr_db,
            int(self.noise_to_idx[str(noise_type)]),
            int(self.snr_to_idx[float(snr_db)]),
        )


def noisy_collate_fn(batch):
    (
        clean_list,
        noisy_list,
        _,
        labels,
        speaker_ids,
        utt_ids,
        noise_types,
        snr_db,
        noise_type_ids,
        snr_ids,
    ) = zip(*batch)
    lengths = [w.shape[0] for w in clean_list]
    max_len = max(lengths)

    clean_padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    noisy_padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (c, n) in enumerate(zip(clean_list, noisy_list)):
        L = c.shape[0]
        clean_padded[i, :L] = c
        noisy_padded[i, :L] = n
        attention_mask[i, :L] = 1

    return {
        "clean_input_values": clean_padded,
        "noisy_input_values": noisy_padded,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.long),
        "speaker_ids": list(speaker_ids),
        "utt_ids": list(utt_ids),
        "noise_types": list(noise_types),
        "snr_db": list(snr_db),
        "noise_type_ids": torch.tensor(noise_type_ids, dtype=torch.long),
        "snr_ids": torch.tensor(snr_ids, dtype=torch.long),
        "snr_values": torch.tensor(snr_db, dtype=torch.float32),
    }
