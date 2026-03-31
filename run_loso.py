from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from transformers import Wav2Vec2Config

from src.data.iemocap_dataset import IemocapDataset
from src.data.noise import NoiseAdder, NoisyPairDataset, noisy_collate_fn
from src.models.ser_models import Wav2Vec2SERStudent
from src.training.metrics import evaluate_classifier
from src.training.trainers import train_student_mlkd, train_teacher
from src.training.utils import get_device, load_metadata_df, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    metadata_csv = config["paths"]["metadata_csv"]
    df = load_metadata_df(metadata_csv)
    speakers = sorted(df["speaker_id"].unique().tolist())

    exp_root = config["paths"]["exp_root"]
    os.makedirs(exp_root, exist_ok=True)

    results = []
    audio_cfg = config["audio"]
    device = get_device()

    for speaker in speakers:
        fold_dir = os.path.join(exp_root, f"fold_{speaker}")
        teacher_dir = os.path.join(fold_dir, "teacher")
        student_dir = os.path.join(fold_dir, "student")
        os.makedirs(teacher_dir, exist_ok=True)
        os.makedirs(student_dir, exist_ok=True)

        print(f"=== LOSO fold: {speaker} ===")
        teacher_ckpt = train_teacher(
            metadata_csv=metadata_csv,
            output_dir=teacher_dir,
            fold_speaker=speaker,
            config=config,
        )

        student_ckpt = train_student_mlkd(
            metadata_csv=metadata_csv,
            teacher_ckpt=teacher_ckpt,
            output_dir=student_dir,
            fold_speaker=speaker,
            config=config,
        )

        test_df = df[df["speaker_id"] == speaker].reset_index(drop=True)
        clean_ds = IemocapDataset(
            test_df,
            sample_rate=audio_cfg["sample_rate"],
            max_seconds=audio_cfg["max_seconds"],
            is_train=False,
            eval_clip=audio_cfg.get("eval_clip", True),
        )

        ckpt = torch.load(student_ckpt, map_location="cpu")
        model_cfg = Wav2Vec2Config(**ckpt["model_config"])
        model_meta = ckpt.get("model_meta", {})
        if not isinstance(model_meta, dict):
            model_meta = {}
        model = Wav2Vec2SERStudent(config=model_cfg, num_classes=4, model_args=model_meta)
        load_result = model.load_state_dict(ckpt["model_state"], strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing or unexpected:
            print(
                f"LOSO load_state_dict(non-strict) | missing={len(missing)} unexpected={len(unexpected)}"
            )
        model.to(device)

        clean_loader = torch.utils.data.DataLoader(
            clean_ds,
            batch_size=config["train"]["batch_size"],
            shuffle=False,
            num_workers=config["train"]["num_workers"],
            collate_fn=clean_ds.collate_fn,
        )
        clean_metrics = evaluate_classifier(
            model, clean_loader, device, num_classes=4, input_key="input_values"
        )

        results.append(
            {
                "speaker": speaker,
                "noise_type": "clean",
                "snr_db": "clean",
                "ua": clean_metrics["ua"],
            }
        )

        noise_adder = NoiseAdder(
            noise_root=config["paths"]["noise_root"],
            noise_types=config["noise"]["types"],
            snr_levels=config["noise"]["snrs"],
            sample_rate=audio_cfg["sample_rate"],
        )

        for ntype in config["noise"]["types"]:
            for snr in config["noise"]["snrs"]:
                noisy_ds = NoisyPairDataset(
                    clean_ds,
                    noise_adder,
                    is_train=False,
                    fixed_noise_type=ntype,
                    fixed_snr=snr,
                )
                noisy_loader = torch.utils.data.DataLoader(
                    noisy_ds,
                    batch_size=config["train"]["batch_size"],
                    shuffle=False,
                    num_workers=config["train"]["num_workers"],
                    collate_fn=noisy_collate_fn,
                )
                noisy_metrics = evaluate_classifier(
                    model,
                    noisy_loader,
                    device,
                    num_classes=4,
                    input_key="noisy_input_values",
                )
                results.append(
                    {
                        "speaker": speaker,
                        "noise_type": ntype,
                        "snr_db": snr,
                        "ua": noisy_metrics["ua"],
                    }
                )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(results)
    summary = (
        results_df.groupby(["noise_type", "snr_db"])["ua"]
        .mean()
        .reset_index()
        .rename(columns={"ua": "ua_mean"})
    )

    os.makedirs(exp_root, exist_ok=True)
    results_df.to_csv(os.path.join(exp_root, "loso_results.csv"), index=False)
    summary.to_csv(os.path.join(exp_root, "loso_summary.csv"), index=False)
    print(f"Saved LOSO results to {exp_root}")


if __name__ == "__main__":
    main()
