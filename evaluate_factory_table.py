from __future__ import annotations

import argparse
import os
import time
from typing import List

import torch
from transformers import Wav2Vec2Config

from src.data.iemocap_dataset import IemocapDataset
from src.data.noise import NoiseAdder, NoisyPairDataset, noisy_collate_fn
from src.models.ser_models import Wav2Vec2SERTeacher, Wav2Vec2SERStudent
from src.training.metrics import evaluate_classifier
from src.training.utils import get_device, load_metadata_df, load_yaml


def _load_model(ckpt_path: str, model_type: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if model_type == "teacher":
        model = Wav2Vec2SERTeacher(
            num_classes=4,
            backbone_name=ckpt.get("model_meta", {}).get("teacher_backbone_name")
            if isinstance(ckpt.get("model_meta"), dict)
            else None,
        )
    else:
        model_cfg = Wav2Vec2Config(**ckpt["model_config"])
        model_meta = ckpt.get("model_meta", {})
        if not isinstance(model_meta, dict):
            model_meta = {}
        model = Wav2Vec2SERStudent(config=model_cfg, num_classes=4, model_args=model_meta)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def _eval_clean(model, test_df, audio_cfg, config, device) -> tuple[float, float]:
    dataset = IemocapDataset(
        test_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=False,
        eval_clip=audio_cfg.get("eval_clip", True),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=dataset.collate_fn,
    )
    start = time.time()
    metrics = evaluate_classifier(model, loader, device, num_classes=4)
    elapsed = time.time() - start
    return metrics["ua"] * 100.0, elapsed


def _eval_factory(
    model, test_df, audio_cfg, config, device, snr_db: float, noise_type: str
) -> tuple[float, float]:
    noise_adder = NoiseAdder(
        noise_root=config["paths"]["noise_root"],
        noise_types=config["noise"]["types"],
        snr_levels=config["noise"]["snrs"],
        sample_rate=audio_cfg["sample_rate"],
    )
    dataset = NoisyPairDataset(
        IemocapDataset(
            test_df,
            sample_rate=audio_cfg["sample_rate"],
            max_seconds=audio_cfg["max_seconds"],
            is_train=False,
            eval_clip=audio_cfg.get("eval_clip", True),
        ),
        noise_adder,
        is_train=False,
        fixed_noise_type=noise_type,
        fixed_snr=snr_db,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=noisy_collate_fn,
    )
    start = time.time()
    metrics = evaluate_classifier(
        model, loader, device, num_classes=4, input_key="noisy_input_values"
    )
    elapsed = time.time() - start
    return metrics["ua"] * 100.0, elapsed


def main() -> None:
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--student_ckpt", type=str, required=True)
    parser.add_argument("--fold_speaker", type=str, required=True)
    parser.add_argument("--noise_type", type=str, default="factory")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    metadata_csv = args.metadata_csv or config["paths"]["metadata_csv"]
    df = load_metadata_df(metadata_csv)
    test_df = df[df["speaker_id"] == args.fold_speaker].reset_index(drop=True)

    audio_cfg = config["audio"]
    device = get_device()

    teacher = _load_model(args.teacher_ckpt, "teacher", device)
    student = _load_model(args.student_ckpt, "student", device)

    snrs: List[float] = list(config["noise"]["snrs"])

    teacher_scores = []
    student_scores = []
    for snr in snrs:
        t_ua, t_time = _eval_factory(
            teacher, test_df, audio_cfg, config, device, snr, args.noise_type
        )
        s_ua, s_time = _eval_factory(
            student, test_df, audio_cfg, config, device, snr, args.noise_type
        )
        teacher_scores.append((t_ua, t_time))
        student_scores.append((s_ua, s_time))

    teacher_clean, teacher_clean_time = _eval_clean(
        teacher, test_df, audio_cfg, config, device
    )
    student_clean, student_clean_time = _eval_clean(
        student, test_df, audio_cfg, config, device
    )

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.student_ckpt), "eval")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "factory_table.csv")
    md_path = os.path.join(output_dir, "factory_table.md")

    headers = ["Models"] + [f"SNR={int(s)}dB" for s in snrs] + ["clean"]
    def _format_cell(ua: float, sec: float) -> str:
        return f"{ua:.2f}({sec:.1f}s)"

    rows = [
        ["Teacher"]
        + [_format_cell(ua, sec) for ua, sec in teacher_scores]
        + [_format_cell(teacher_clean, teacher_clean_time)],
        ["Student with supervision"]
        + [_format_cell(ua, sec) for ua, sec in student_scores]
        + [_format_cell(student_clean, student_clean_time)],
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
        f.write(f"EvalTime(s),{time.time() - start_time:.1f}\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
        f.write("| EvalTime(s) | " + f"{time.time() - start_time:.1f}" + " |\n")

    print("Factory table (paper-style, markdown):")
    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            print(line.rstrip())
    print(f"Saved table to {out_path}")
    print(f"Saved markdown table to {md_path}")


if __name__ == "__main__":
    main()
