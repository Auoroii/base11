from __future__ import annotations

import argparse
import os

from src.training.trainers import train_student_mlkd, rerank_student_topk_checkpoints
from src.training.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--metadata_csv", type=str, default=None)
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--fold_speaker",
        type=str,
        default=None,
        help="Required for split_mode=loso; ignored for split_mode=random.",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["fixed", "cosine", "plateau"],
        default=None,
    )
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--warmup-start-lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument(
        "--monitor",
        type=str,
        choices=["full_mean_UA", "proxy_mean_UA"],
        default=None,
    )
    parser.add_argument("--full-val-every", type=int, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--loss-schedule",
        type=str,
        choices=["fixed", "staged"],
        default=None,
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save-best-full", dest="save_best_full", action="store_true")
    parser.add_argument("--no-save-best-full", dest="save_best_full", action="store_false")
    parser.add_argument("--save-best-proxy", dest="save_best_proxy", action="store_true")
    parser.add_argument("--no-save-best-proxy", dest="save_best_proxy", action="store_false")
    parser.set_defaults(save_best_full=None, save_best_proxy=None)
    parser.add_argument("--rerank_topk_only", action="store_true")
    parser.add_argument(
        "--variant",
        type=str,
        choices=["mlkd_baseline", "adaptive_dualpath_mlkd"],
        default=None,
    )
    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
        help=(
            "baseline|enhancement_only|dual_path_no_gate|dual_path_gate_oracle|"
            "dual_path_gate_pred|dual_path_gate_pred_reg|dual_path_gate_hybrid"
        ),
    )
    parser.add_argument(
        "--gate-condition-mode",
        dest="gate_condition_mode",
        type=str,
        choices=["none", "oracle", "predicted", "hybrid"],
        default=None,
    )
    parser.add_argument(
        "--eval-condition-mode",
        dest="eval_condition_mode",
        type=str,
        choices=["deploy", "oracle"],
        default=None,
    )
    parser.add_argument("--condition-feat-dim", type=int, default=None)
    parser.add_argument("--hybrid-warmup-epochs", type=int, default=None)
    parser.add_argument("--hybrid-transition-epochs", type=int, default=None)
    parser.add_argument("--lambda-snr-reg", type=float, default=None)
    parser.add_argument("--use-snr-regression", dest="use_snr_regression", action="store_true")
    parser.add_argument("--no-use-snr-regression", dest="use_snr_regression", action="store_false")
    parser.set_defaults(use_snr_regression=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if "model" not in config:
        config["model"] = {}
    if args.variant:
        config["model"]["variant"] = args.variant
    if args.ablation:
        config["model"]["ablation"] = args.ablation
    if args.gate_condition_mode is not None:
        config["model"]["gate_condition_mode"] = str(args.gate_condition_mode)
    if args.condition_feat_dim is not None:
        config["model"]["condition_feat_dim"] = int(args.condition_feat_dim)
    if args.use_snr_regression is not None:
        config["model"]["use_snr_regression"] = bool(args.use_snr_regression)

    if "training" not in config:
        config["training"] = {}
    if "loss" not in config:
        config["loss"] = {}
    if "student_val" not in config:
        config["student_val"] = {}

    max_epochs = args.max_epochs if args.max_epochs is not None else args.num_epochs
    if max_epochs is not None:
        config["training"]["epochs"] = int(max_epochs)
    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        config["training"]["lr"] = float(args.lr)
    if args.scheduler is not None:
        config["training"]["scheduler"] = str(args.scheduler)
    if args.warmup_epochs is not None:
        config["training"]["warmup_epochs"] = int(args.warmup_epochs)
    if args.warmup_start_lr is not None:
        config["training"]["warmup_start_lr"] = float(args.warmup_start_lr)
    if args.min_lr is not None:
        config["training"]["min_lr"] = float(args.min_lr)
    if args.early_stop_patience is not None:
        config["training"]["early_stop_patience"] = int(args.early_stop_patience)
        config["training"]["early_stopping"] = int(args.early_stop_patience)
    if args.monitor is not None:
        config["training"]["monitor"] = str(args.monitor)
    if args.eval_condition_mode is not None:
        config["training"]["eval_condition_mode"] = str(args.eval_condition_mode)
    if args.hybrid_warmup_epochs is not None:
        config["training"]["hybrid_warmup_epochs"] = int(args.hybrid_warmup_epochs)
    if args.hybrid_transition_epochs is not None:
        config["training"]["hybrid_transition_epochs"] = int(args.hybrid_transition_epochs)
    if args.grad_clip is not None:
        config["training"]["grad_clip"] = float(args.grad_clip)

    if args.loss_schedule is not None:
        config["loss"]["schedule"] = str(args.loss_schedule)
    if args.lambda_snr_reg is not None:
        config["loss"]["lambda_snr_reg"] = float(args.lambda_snr_reg)

    if args.full_val_every is not None:
        config["student_val"]["full_eval_every"] = int(args.full_val_every)
    if args.save_best_full is not None:
        config["training"]["save_best_full"] = bool(args.save_best_full)
    if args.save_best_proxy is not None:
        config["training"]["save_best_proxy"] = bool(args.save_best_proxy)

    resume_spec = args.resume if args.resume is not None else args.resume_ckpt

    metadata_csv = args.metadata_csv or config["paths"]["metadata_csv"]
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.rerank_topk_only:
        ckpt_path = rerank_student_topk_checkpoints(
            metadata_csv=metadata_csv,
            teacher_ckpt=args.teacher_ckpt,
            output_dir=output_dir,
            fold_speaker=args.fold_speaker,
            config=config,
            batch_size=args.batch_size,
        )
    else:
        ckpt_path = train_student_mlkd(
            metadata_csv=metadata_csv,
            teacher_ckpt=args.teacher_ckpt,
            output_dir=output_dir,
            fold_speaker=args.fold_speaker,
            config=config,
            epochs=max_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_ckpt=resume_spec,
        )
    print(f"Best student checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
