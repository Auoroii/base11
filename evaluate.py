from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import Wav2Vec2Config

from src.data.iemocap_dataset import IemocapDataset
from src.data.noise import NoiseAdder, NoisyPairDataset, noisy_collate_fn
from src.models.ser_models import Wav2Vec2SERTeacher, Wav2Vec2SERStudent
from src.training.metrics import evaluate_classifier, unweighted_accuracy
from src.training.utils import (
    get_device,
    load_metadata_df,
    load_yaml,
    split_train_val_test,
)


def save_results(output_dir: str, ua: float, cm: np.ndarray) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "metrics.json")
    csv_path = os.path.join(output_dir, "metrics.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"ua": ua, "confusion_matrix": cm.tolist()},
            f,
            indent=2,
        )

    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        recall = float(cm[i, i] / denom) if denom > 0 else 0.0
        recalls.append(recall)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("class_id,recall\n")
        for i, r in enumerate(recalls):
            f.write(f"{i},{r:.6f}\n")


def _build_clean_dataset(test_df, audio_cfg: Dict[str, Any]) -> IemocapDataset:
    return IemocapDataset(
        test_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=False,
        eval_clip=audio_cfg.get("eval_clip", True),
    )


def _evaluate_all_conditions_once(
    *,
    model: torch.nn.Module,
    test_df,
    config: Dict[str, Any],
    audio_cfg: Dict[str, Any],
    device: torch.device,
    eval_seed: int,
    pass_condition_labels: bool,
    condition_source_override: Optional[str],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    clean_ds = _build_clean_dataset(test_df, audio_cfg)
    clean_loader = torch.utils.data.DataLoader(
        clean_ds,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=config["train"]["num_workers"],
        collate_fn=clean_ds.collate_fn,
    )
    clean_start = time.time()
    clean_metrics = evaluate_classifier(
        model,
        clean_loader,
        device,
        num_classes=4,
        pass_condition_labels=pass_condition_labels,
        condition_source_override=condition_source_override,
    )
    clean_time = time.time() - clean_start
    results.append(
        {
            "noise_type": "clean",
            "snr_db": "clean",
            "ua": float(clean_metrics["ua"]),
            "eval_time": float(clean_time),
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
            dataset = NoisyPairDataset(
                _build_clean_dataset(test_df, audio_cfg),
                noise_adder,
                is_train=False,
                fixed_noise_type=ntype,
                fixed_snr=snr,
                deterministic_eval=True,
                eval_seed=eval_seed,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config["train"]["batch_size"],
                shuffle=False,
                num_workers=config["train"]["num_workers"],
                collate_fn=noisy_collate_fn,
            )
            cond_start = time.time()
            metrics = evaluate_classifier(
                model,
                loader,
                device,
                num_classes=4,
                input_key="noisy_input_values",
                pass_condition_labels=pass_condition_labels,
                condition_source_override=condition_source_override,
            )
            cond_time = time.time() - cond_start
            results.append(
                {
                    "noise_type": str(ntype),
                    "snr_db": float(snr),
                    "ua": float(metrics["ua"]),
                    "eval_time": float(cond_time),
                }
            )
    return results


def _summarize_all_condition_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    noisy_rows = [r for r in results if r["noise_type"] != "clean"]
    clean_row = next((r for r in results if r["noise_type"] == "clean"), None)
    overall_noisy_ua = (
        float(sum(r["ua"] for r in noisy_rows) / max(1, len(noisy_rows))) if noisy_rows else 0.0
    )
    overall_all_ua = float(sum(r["ua"] for r in results) / max(1, len(results))) if results else 0.0

    by_noise: Dict[str, List[float]] = {}
    by_snr: Dict[str, List[float]] = {}
    for row in noisy_rows:
        by_noise.setdefault(str(row["noise_type"]), []).append(float(row["ua"]))
        by_snr.setdefault(str(row["snr_db"]), []).append(float(row["ua"]))
    by_noise_ua = {k: float(sum(v) / max(1, len(v))) for k, v in by_noise.items()}
    by_snr_ua = {k: float(sum(v) / max(1, len(v))) for k, v in by_snr.items()}

    return {
        "clean_ua": float(clean_row["ua"]) if clean_row is not None else None,
        "overall_noisy_ua": overall_noisy_ua,
        "overall_all_ua": overall_all_ua,
        "by_noise_type_ua": by_noise_ua,
        "by_snr_ua": by_snr_ua,
    }


def _write_all_conditions_table(
    output_dir: str,
    results: List[Dict[str, Any]],
    noise_types: List[str],
    snrs: List[float],
    total_eval_time_s: float,
) -> Tuple[str, str]:
    display_map = {
        "babble": "Babble",
        "f16": "F16",
        "factory": "Factory",
        "hf": "HF-channel",
        "volvo": "Volvo",
    }
    headers = ["snr_db"] + [f"{display_map.get(ntype, ntype)}_MLKD" for ntype in noise_types]
    table_path = os.path.join(output_dir, "mlkd_table.csv")
    md_path = os.path.join(output_dir, "mlkd_table.md")

    rows = []
    for snr in snrs:
        row_vals = [str(snr)]
        for ntype in noise_types:
            match = next(r for r in results if r["snr_db"] == snr and r["noise_type"] == ntype)
            row_vals.append(f"{match['ua'] * 100.0:.2f}({match['eval_time']:.1f}s)")
        rows.append(row_vals)
    mean_vals = ["mean"]
    for ntype in noise_types:
        vals = [r["ua"] * 100.0 for r in results if r["noise_type"] == ntype]
        mean_vals.append(f"{sum(vals)/len(vals):.2f}")
    rows.append(mean_vals)

    with open(table_path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row_vals in rows:
            f.write(",".join(row_vals) + "\n")
        f.write(f"EvalTime(s),{total_eval_time_s:.1f}\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row_vals in rows:
            f.write("| " + " | ".join(row_vals) + " |\n")
        f.write("| EvalTime(s) | " + f"{total_eval_time_s:.1f}" + " |\n")
    return table_path, md_path


def main() -> None:
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["teacher", "student"], required=True)
    parser.add_argument(
        "--fold_speaker",
        type=str,
        default=None,
        help="Required for split_mode=loso; ignored for split_mode=random.",
    )
    parser.add_argument("--noise_type", type=str, default=None)
    parser.add_argument("--snr_db", type=float, default=None)
    parser.add_argument("--all_conditions", action="store_true")
    parser.add_argument("--all-conditions-repeats", type=int, default=1)
    parser.add_argument("--all-conditions-seed", type=int, default=1234)
    parser.add_argument("--table_only", action="store_true")
    parser.add_argument(
        "--blind_noise_labels",
        action="store_true",
        help="Do not pass snr/noise condition labels to student during evaluation.",
    )
    parser.add_argument(
        "--eval_condition_mode",
        type=str,
        choices=["deploy", "oracle"],
        default=None,
        help="deploy: gate cannot use oracle labels; oracle: gate can use oracle labels.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    metadata_csv = args.metadata_csv or config["paths"]["metadata_csv"]
    df = load_metadata_df(metadata_csv)
    data_cfg = dict(config.get("data", {}))
    split_mode = str(data_cfg.get("split_mode", "loso")).strip().lower()
    val_ratio = float(data_cfg.get("val_ratio", config.get("train", {}).get("val_split", 0.1)))
    default_test_ratio = 0.1 if split_mode in {"random", "random_811", "random_stratified"} else 0.0
    test_ratio = float(data_cfg.get("test_ratio", default_test_ratio))
    if split_mode in {"loso", "speaker", "leave_one_speaker_out"}:
        if not args.fold_speaker:
            raise ValueError("fold_speaker is required when split_mode=loso")
        test_df = df[df["speaker_id"] == args.fold_speaker].reset_index(drop=True)
    else:
        _, _, test_df = split_train_val_test(
            df,
            split_mode=split_mode,
            seed=int(config.get("train", {}).get("seed", 1234)),
            fold_speaker=args.fold_speaker,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    audio_cfg = config["audio"]
    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if args.model_type == "teacher":
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

    load_result = model.load_state_dict(ckpt["model_state"], strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"Checkpoint load_state_dict(non-strict) | missing={len(missing)} unexpected={len(unexpected)}"
        )
    model.to(device)

    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    os.makedirs(output_dir, exist_ok=True)

    eval_condition_mode = str(
        args.eval_condition_mode
        or config.get("training", {}).get("eval_condition_mode", "deploy")
    ).strip().lower()
    if eval_condition_mode not in {"deploy", "oracle"}:
        eval_condition_mode = "deploy"

    condition_source_override: Optional[str] = None
    if args.model_type == "student":
        condition_source_override = "oracle" if eval_condition_mode == "oracle" else "predicted"

    pass_condition_labels = not args.blind_noise_labels
    if args.model_type == "student":
        pass_condition_labels = eval_condition_mode == "oracle"
    if args.blind_noise_labels:
        pass_condition_labels = False

    if args.model_type == "student":
        print(
            f"Eval mode | eval_condition_mode={eval_condition_mode} "
            f"condition_source={condition_source_override} "
            f"pass_condition_labels={int(pass_condition_labels)}"
        )

    results: List[Dict[str, Any]] = []
    if args.all_conditions:
        repeats = max(1, int(args.all_conditions_repeats))
        if repeats == 1:
            results = _evaluate_all_conditions_once(
                model=model,
                test_df=test_df,
                config=config,
                audio_cfg=audio_cfg,
                device=device,
                eval_seed=int(args.all_conditions_seed),
                pass_condition_labels=pass_condition_labels,
                condition_source_override=condition_source_override,
            )
            csv_path = os.path.join(output_dir, "all_conditions.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("noise_type,snr_db,ua,eval_time_s\n")
                for row in results:
                    f.write(
                        f"{row['noise_type']},{row['snr_db']},{row['ua']:.6f},{row['eval_time']:.3f}\n"
                    )
            print(f"Saved results to {csv_path}")

            summary = _summarize_all_condition_results(results)
            summary_path = os.path.join(output_dir, "ua_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"Saved UA summary to {summary_path}")

            table_path, md_path = _write_all_conditions_table(
                output_dir=output_dir,
                results=results,
                noise_types=list(config["noise"]["types"]),
                snrs=list(config["noise"]["snrs"]),
                total_eval_time_s=(time.time() - start_time),
            )
            print(f"Saved MLKD table to {table_path}")
            print(f"Saved MLKD markdown table to {md_path}")
            print("MLKD table (paper-style, markdown):")
            with open(md_path, "r", encoding="utf-8") as f:
                for line in f:
                    print(line.rstrip())
        else:
            all_repeat_rows: List[Dict[str, Any]] = []
            repeat_summaries: List[Dict[str, Any]] = []
            for rep in range(repeats):
                rep_seed = int(args.all_conditions_seed) + rep
                rep_results = _evaluate_all_conditions_once(
                    model=model,
                    test_df=test_df,
                    config=config,
                    audio_cfg=audio_cfg,
                    device=device,
                    eval_seed=rep_seed,
                    pass_condition_labels=pass_condition_labels,
                    condition_source_override=condition_source_override,
                )
                rep_summary = _summarize_all_condition_results(rep_results)
                repeat_summaries.append(
                    {
                        "repeat": rep + 1,
                        "eval_seed": rep_seed,
                        "clean_ua": rep_summary["clean_ua"],
                        "mean_noisy_ua": rep_summary["overall_noisy_ua"],
                        "mean_all_ua": rep_summary["overall_all_ua"],
                    }
                )
                print(
                    f"Repeat {rep + 1}/{repeats} | "
                    f"mean_noisy_UA={rep_summary['overall_noisy_ua']:.4f} "
                    f"mean_all_UA={rep_summary['overall_all_ua']:.4f}"
                )
                for row in rep_results:
                    all_repeat_rows.append(
                        {
                            "repeat": rep + 1,
                            "eval_seed": rep_seed,
                            "noise_type": row["noise_type"],
                            "snr_db": row["snr_db"],
                            "ua": float(row["ua"]),
                            "eval_time": float(row["eval_time"]),
                        }
                    )

            repeat_csv = os.path.join(output_dir, "all_conditions_repeats.csv")
            with open(repeat_csv, "w", encoding="utf-8") as f:
                f.write("repeat,eval_seed,noise_type,snr_db,ua,eval_time_s\n")
                for row in all_repeat_rows:
                    f.write(
                        f"{row['repeat']},{row['eval_seed']},{row['noise_type']},{row['snr_db']},"
                        f"{row['ua']:.6f},{row['eval_time']:.3f}\n"
                    )
            print(f"Saved repeated all-condition results to {repeat_csv}")

            repeat_mean_csv = os.path.join(output_dir, "repeat_mean_ua.csv")
            with open(repeat_mean_csv, "w", encoding="utf-8") as f:
                f.write("repeat,eval_seed,clean_ua,mean_noisy_ua,mean_all_ua\n")
                for row in repeat_summaries:
                    clean_txt = "" if row["clean_ua"] is None else f"{float(row['clean_ua']):.6f}"
                    f.write(
                        f"{row['repeat']},{row['eval_seed']},{clean_txt},"
                        f"{float(row['mean_noisy_ua']):.6f},{float(row['mean_all_ua']):.6f}\n"
                    )
            print(f"Saved per-repeat mean UA to {repeat_mean_csv}")

            grouped: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
            for row in all_repeat_rows:
                key = (str(row["noise_type"]), row["snr_db"])
                grouped.setdefault(key, []).append(row)

            mean_results: List[Dict[str, Any]] = []
            mean_csv = os.path.join(output_dir, "all_conditions_mean_over_repeats.csv")
            with open(mean_csv, "w", encoding="utf-8") as f:
                f.write("noise_type,snr_db,ua_mean,ua_std,eval_time_mean_s\n")
                for noise_type, snr_db in sorted(
                    grouped.keys(), key=lambda x: (x[0] != "clean", x[0], str(x[1]))
                ):
                    rows = grouped[(noise_type, snr_db)]
                    uas = [float(x["ua"]) for x in rows]
                    times = [float(x["eval_time"]) for x in rows]
                    ua_mean = float(np.mean(uas))
                    ua_std = float(np.std(uas))
                    t_mean = float(np.mean(times))
                    f.write(f"{noise_type},{snr_db},{ua_mean:.6f},{ua_std:.6f},{t_mean:.3f}\n")
                    mean_results.append(
                        {
                            "noise_type": noise_type,
                            "snr_db": snr_db,
                            "ua": ua_mean,
                            "eval_time": t_mean,
                        }
                    )
            print(f"Saved mean UA over repeats to {mean_csv}")

            compat_csv = os.path.join(output_dir, "all_conditions.csv")
            with open(compat_csv, "w", encoding="utf-8") as f:
                f.write("noise_type,snr_db,ua,eval_time_s\n")
                for row in mean_results:
                    f.write(
                        f"{row['noise_type']},{row['snr_db']},{row['ua']:.6f},{row['eval_time']:.3f}\n"
                    )
            print(f"Saved compatibility all_conditions file to {compat_csv}")

            mean_noisy_list = [float(x["mean_noisy_ua"]) for x in repeat_summaries]
            mean_all_list = [float(x["mean_all_ua"]) for x in repeat_summaries]
            summary_payload = {
                "repeats": repeats,
                "base_eval_seed": int(args.all_conditions_seed),
                "repeat_mean_ua": repeat_summaries,
                "mean_of_repeat_mean_noisy_ua": float(np.mean(mean_noisy_list)),
                "std_of_repeat_mean_noisy_ua": float(np.std(mean_noisy_list)),
                "mean_of_repeat_mean_all_ua": float(np.mean(mean_all_list)),
                "std_of_repeat_mean_all_ua": float(np.std(mean_all_list)),
            }
            summary_path = os.path.join(output_dir, "ua_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_payload, f, indent=2)
            print(f"Saved UA summary to {summary_path}")

            table_path, md_path = _write_all_conditions_table(
                output_dir=output_dir,
                results=mean_results,
                noise_types=list(config["noise"]["types"]),
                snrs=list(config["noise"]["snrs"]),
                total_eval_time_s=(time.time() - start_time),
            )
            print(f"Saved mean MLKD table to {table_path}")
            print(f"Saved mean MLKD markdown table to {md_path}")
            print("Mean MLKD table over repeats (markdown):")
            with open(md_path, "r", encoding="utf-8") as f:
                for line in f:
                    print(line.rstrip())
    else:
        if args.noise_type is None and args.snr_db is None:
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
            metrics = evaluate_classifier(
                model,
                loader,
                device,
                num_classes=4,
                pass_condition_labels=pass_condition_labels,
                condition_source_override=condition_source_override,
            )
        else:
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
                fixed_noise_type=args.noise_type or config["noise"]["types"][0],
                fixed_snr=args.snr_db
                if args.snr_db is not None
                else config["noise"]["snrs"][0],
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config["train"]["batch_size"],
                shuffle=False,
                num_workers=config["train"]["num_workers"],
                collate_fn=noisy_collate_fn,
            )
            metrics = evaluate_classifier(
                model,
                loader,
                device,
                num_classes=4,
                input_key="noisy_input_values",
                pass_condition_labels=pass_condition_labels,
                condition_source_override=condition_source_override,
            )

        ua = metrics["ua"]
        cm = metrics["confusion_matrix"]
        print(f"UA: {ua:.4f}")
        save_results(output_dir, ua, cm)
        print(f"Saved results to {output_dir}")

    elapsed = time.time() - start_time
    print(f"Total eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
