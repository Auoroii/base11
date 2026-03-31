from __future__ import annotations

import argparse
import os

from src.training.trainers import train_teacher
from src.training.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--fold_speaker",
        type=str,
        default=None,
        help="Required for split_mode=loso; ignored for split_mode=random.",
    )
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    metadata_csv = args.metadata_csv or config["paths"]["metadata_csv"]
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = train_teacher(
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        fold_speaker=args.fold_speaker,
        config=config,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume_ckpt=args.resume_ckpt,
    )
    print(f"Best teacher checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
