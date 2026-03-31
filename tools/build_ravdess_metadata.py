from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional


# RAVDESS filename format:
# 03-01-EMO-INT-STM-REP-ACTOR.wav
# EMO codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
EMO_CODE_TO_LABEL = {
    "01": "neu",
    "02": "neu",  # calm merged into neutral by default
    "03": "hap",
    "04": "sad",
    "05": "ang",
}
LABEL_TO_ID = {"ang": 0, "neu": 1, "sad": 2, "hap": 3}


def _parse_ravdess_filename(filename: str) -> Optional[Dict[str, str]]:
    stem, ext = os.path.splitext(filename)
    if ext.lower() != ".wav":
        return None
    parts = stem.split("-")
    if len(parts) != 7:
        return None
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion_code": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor_id": parts[6],
        "utt_id": stem,
    }


def build_metadata(ravdess_root: str, include_calm_as_neu: bool = True) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for root, _, files in os.walk(ravdess_root):
        for name in sorted(files):
            parsed = _parse_ravdess_filename(name)
            if parsed is None:
                continue
            emo = parsed["emotion_code"]
            if emo == "02" and not include_calm_as_neu:
                continue
            label_str = EMO_CODE_TO_LABEL.get(emo)
            if label_str is None:
                continue

            speaker_id = f"Actor_{parsed['actor_id']}"
            wav_path = os.path.abspath(os.path.join(root, name))
            rows.append(
                {
                    "path": wav_path,
                    "label_str": label_str,
                    "label_id": LABEL_TO_ID[label_str],
                    "speaker_id": speaker_id,
                    "session_id": speaker_id,
                    "utt_id": parsed["utt_id"],
                }
            )
    rows.sort(key=lambda x: x["path"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess_root", required=True, type=str)
    parser.add_argument("--output_csv", default="data/ravdess_metadata.csv", type=str)
    parser.add_argument(
        "--exclude_calm",
        action="store_true",
        help="Exclude calm(02). By default calm is merged into neu.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.ravdess_root):
        raise FileNotFoundError(f"ravdess_root not found: {args.ravdess_root}")

    rows = build_metadata(
        args.ravdess_root,
        include_calm_as_neu=not bool(args.exclude_calm),
    )
    if not rows:
        raise RuntimeError("No valid RAVDESS utterances found. Check root path and filenames.")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    fieldnames = ["path", "label_str", "label_id", "speaker_id", "session_id", "utt_id"]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
