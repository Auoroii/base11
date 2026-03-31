from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

LABEL_MAP: Dict[str, int] = {"ang": 0, "neu": 1, "sad": 2, "hap": 3}
ALLOWED = set(LABEL_MAP.keys()) | {"exc"}


def parse_label(label: str) -> Optional[str]:
    label = label.lower()
    if label == "exc":
        return "hap"
    if label in LABEL_MAP:
        return label
    return None


def parse_eval_line(line: str) -> Optional[Tuple[str, str]]:
    # Typical format:
    # [start - end]\tSes01F_impro01_F000\tneu\t[...]
    # Or sometimes without brackets. We search for utt_id and label tokens.
    utt_match = re.search(r"(Ses\d{2}[FM]_[A-Za-z0-9_]+)", line)
    if not utt_match:
        return None
    utt_id = utt_match.group(1)

    tokens = re.split(r"[\t\s]+", line.strip())
    label = None
    for tok in tokens:
        lab = parse_label(tok)
        if lab is not None:
            label = lab
            break
    if label is None:
        return None
    return utt_id, label


def extract_speaker_id(utt_id: str) -> Optional[str]:
    match = re.match(r"(Ses\d{2}[FM])", utt_id)
    if not match:
        return None
    return match.group(1)


def build_metadata(iemocap_root: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for session_idx in range(1, 6):
        session_name = f"Session{session_idx}"
        session_dir = os.path.join(iemocap_root, session_name)
        eval_dir = os.path.join(session_dir, "dialog", "EmoEvaluation")
        if not os.path.isdir(eval_dir):
            print(f"Warning: missing {eval_dir}", file=sys.stderr)
            continue
        eval_files = sorted(glob.glob(os.path.join(eval_dir, "*.txt")))
        for eval_file in eval_files:
            with open(eval_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parsed = parse_eval_line(line)
                    if not parsed:
                        continue
                    utt_id, label = parsed
                    speaker_id = extract_speaker_id(utt_id)
                    if speaker_id is None:
                        continue
                    # Use full dialog folder name (e.g., Ses01F_script01_1 or Ses01F_impro01).
                    dialog_id = "_".join(utt_id.split("_")[:-1])
                    wav_path = os.path.join(
                        session_dir, "sentences", "wav", dialog_id, f"{utt_id}.wav"
                    )
                    if not os.path.isfile(wav_path):
                        print(f"Warning: missing wav {wav_path}", file=sys.stderr)
                        continue
                    rows.append(
                        {
                            "path": wav_path,
                            "label_str": label,
                            "label_id": str(LABEL_MAP[label]),
                            "speaker_id": speaker_id,
                            "session_id": session_name,
                            "utt_id": utt_id,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iemocap_root", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.iemocap_root):
        raise FileNotFoundError(f"iemocap_root not found: {args.iemocap_root}")

    rows = build_metadata(args.iemocap_root)
    if not rows:
        raise RuntimeError("No valid utterances found. Check paths and labels.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "label_str",
                "label_id",
                "speaker_id",
                "session_id",
                "utt_id",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
