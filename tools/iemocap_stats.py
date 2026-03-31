from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, type=str)
    args = parser.parse_args()

    if not os.path.isfile(args.metadata_csv):
        raise FileNotFoundError(f"metadata_csv not found: {args.metadata_csv}")

    df = pd.read_csv(args.metadata_csv)

    total = len(df)
    print(f"Total samples: {total}")

    if total == 0:
        return

    print("\nLabel distribution:")
    print(df["label_str"].value_counts().sort_index())

    print("\nSpeaker distribution:")
    print(df["speaker_id"].value_counts().sort_index())

    print("\nSession distribution:")
    print(df["session_id"].value_counts().sort_index())


if __name__ == "__main__":
    main()