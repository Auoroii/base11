from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import os
import random
import csv
import json
import datetime as dt
import time
import math
from collections import Counter

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from transformers import Wav2Vec2Config
from tqdm import tqdm
from src.data.iemocap_dataset import IemocapDataset
from src.data.noise import (
    NoiseAdder,
    NoisyPairDataset,
    ManifestNoisyPairDataset,
    noisy_collate_fn,
)
from src.models.ser_models import Wav2Vec2SERTeacher, Wav2Vec2SERStudent
from src.training.losses import compute_adaptive_mlkd_loss
from src.training.metrics import evaluate_classifier
from src.training.utils import (
    ensure_dir,
    get_device,
    load_metadata_df,
    seed_worker,
    set_seed,
    split_train_val_test,
)


def _resolve_runtime_train_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = dict(config.get("train", {}))
    training_cfg = dict(config.get("training", {}))
    merged = dict(train_cfg)
    merged.update(training_cfg)
    if "seed" not in merged:
        merged["seed"] = train_cfg.get("seed", 1234)
    if "amp" not in merged:
        merged["amp"] = train_cfg.get("amp", False)
    if "val_split" not in merged:
        merged["val_split"] = train_cfg.get("val_split", 0.1)
    if "num_workers" not in merged:
        merged["num_workers"] = train_cfg.get("num_workers", 4)
    if "batch_size" not in merged:
        merged["batch_size"] = train_cfg.get("batch_size", 8)
    if "lr" not in merged:
        merged["lr"] = train_cfg.get("lr", 5e-5)
    if "epochs" not in merged:
        merged["epochs"] = train_cfg.get("epochs", 200)
    merged.setdefault("freeze_teacher", True)
    merged.setdefault("freeze_feature_encoder_epochs", 0)
    merged.setdefault("grad_clip", 1.0)
    merged.setdefault("scheduler", "fixed")
    merged.setdefault("warmup_epochs", 0)
    merged.setdefault("warmup_start_lr", merged.get("lr", 5e-5))
    merged.setdefault("min_lr", merged.get("lr", 5e-5))
    merged.setdefault("plateau_factor", 0.5)
    merged.setdefault("plateau_patience", 3)
    merged.setdefault("plateau_threshold", 1e-4)
    merged.setdefault("monitor", "full_mean_UA")
    merged.setdefault("eval_condition_mode", "deploy")
    merged.setdefault("hybrid_warmup_epochs", 0)
    merged.setdefault("hybrid_transition_epochs", 10)
    if "early_stop_patience" not in merged:
        merged["early_stop_patience"] = int(merged.get("early_stopping", 0) or 0)
    merged.setdefault("early_stopping", merged["early_stop_patience"])
    merged.setdefault("save_best_full", True)
    merged.setdefault("save_best_proxy", True)
    return merged


def _resolve_loss_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    kd_cfg = dict(config.get("kd", {}))
    alpha = float(kd_cfg.get("alpha", 0.7))
    loss_cfg = dict(config.get("loss", {}))
    resolved = {
        "temperature": float(loss_cfg.get("temperature", kd_cfg.get("temperature", 7.0))),
        "lambda_ce": float(loss_cfg.get("lambda_ce", 1.0 - alpha)),
        "lambda_kl": float(loss_cfg.get("lambda_kl", alpha)),
        "lambda_mse": float(loss_cfg.get("lambda_mse", 1.0)),
        "lambda_enh": float(loss_cfg.get("lambda_enh", 0.0)),
        "lambda_snr": float(loss_cfg.get("lambda_snr", 0.0)),
        "lambda_noise": float(loss_cfg.get("lambda_noise", 0.0)),
        "lambda_snr_reg": float(loss_cfg.get("lambda_snr_reg", 0.0)),
        "snr_reg_loss_type": str(loss_cfg.get("snr_reg_loss_type", "smooth_l1")),
        "w_wav": float(loss_cfg.get("w_wav", 1.0)),
        "w_stft": float(loss_cfg.get("w_stft", 1.0)),
        "distill_layers": str(loss_cfg.get("distill_layers", "original_even_mapping")),
    }
    return resolved


def _default_staged_loss_config() -> List[Dict[str, Any]]:
    return [
        {
            "name": "stage_a",
            "start_epoch": 1,
            "end_epoch": 20,
            "lambda_ce": 1.0,
            "lambda_kl": 0.7,
            "lambda_mse": 1.0,
            "lambda_enh": 0.2,
            "lambda_snr": 0.05,
            "lambda_noise": 0.05,
            "lambda_snr_reg": 0.0,
        },
        {
            "name": "stage_b",
            "start_epoch": 21,
            "end_epoch": 40,
            "lambda_ce": 1.0,
            "lambda_kl": 0.7,
            "lambda_mse": 0.5,
            "lambda_enh": 0.1,
            "lambda_snr": 0.05,
            "lambda_noise": 0.05,
            "lambda_snr_reg": 0.0,
        },
        {
            "name": "stage_c",
            "start_epoch": 41,
            "end_epoch": None,
            "lambda_ce": 1.0,
            "lambda_kl": 0.7,
            "lambda_mse": 0.2,
            "lambda_enh": 0.05,
            "lambda_snr": 0.05,
            "lambda_noise": 0.05,
            "lambda_snr_reg": 0.0,
        },
    ]


def _resolve_loss_schedule_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    loss_cfg = dict(config.get("loss", {}))
    schedule_cfg = dict(config.get("loss_schedule", {}))
    mode = str(schedule_cfg.get("mode", loss_cfg.get("schedule", "fixed"))).strip().lower()
    if mode not in {"fixed", "staged"}:
        raise ValueError(f"Unsupported loss schedule mode: {mode}")

    stages_raw = schedule_cfg.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        stages_raw = _default_staged_loss_config()

    parsed_stages: List[Dict[str, Any]] = []
    for idx, stage in enumerate(stages_raw):
        if not isinstance(stage, dict):
            continue
        start_epoch = max(1, int(stage.get("start_epoch", 1)))
        end_epoch_raw = stage.get("end_epoch")
        if end_epoch_raw is None or end_epoch_raw == "":
            end_epoch = None
        else:
            end_epoch_int = int(end_epoch_raw)
            end_epoch = None if end_epoch_int < 0 else max(start_epoch, end_epoch_int)
        parsed = {
            "name": str(stage.get("name", f"stage_{idx + 1}")),
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
        }
        for key in (
            "lambda_ce",
            "lambda_kl",
            "lambda_mse",
            "lambda_enh",
            "lambda_snr",
            "lambda_noise",
            "lambda_snr_reg",
        ):
            if key in stage:
                parsed[key] = float(stage[key])
        parsed_stages.append(parsed)

    if not parsed_stages:
        parsed_stages = _default_staged_loss_config()

    parsed_stages = sorted(parsed_stages, key=lambda x: int(x["start_epoch"]))
    return {"mode": mode, "stages": parsed_stages}


def get_loss_weights(epoch: int, config: Dict[str, Any]) -> Dict[str, Any]:
    base_cfg = _resolve_loss_cfg(config)
    schedule_cfg = _resolve_loss_schedule_cfg(config)
    resolved = dict(base_cfg)

    if schedule_cfg["mode"] != "staged":
        resolved["loss_stage"] = "fixed"
        return resolved

    selected_stage = schedule_cfg["stages"][-1]
    for stage in schedule_cfg["stages"]:
        start_epoch = int(stage["start_epoch"])
        end_epoch = stage["end_epoch"]
        if epoch >= start_epoch and (end_epoch is None or epoch <= int(end_epoch)):
            selected_stage = stage
            break

    for key in (
        "lambda_ce",
        "lambda_kl",
        "lambda_mse",
        "lambda_enh",
        "lambda_snr",
        "lambda_noise",
        "lambda_snr_reg",
    ):
        if key in selected_stage:
            resolved[key] = float(selected_stage[key])
    resolved["loss_stage"] = str(selected_stage.get("name", "staged"))
    return resolved


def _resolve_scheduler_cfg(train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    scheduler_mode = str(train_cfg.get("scheduler", "fixed")).strip().lower()
    if scheduler_mode not in {"fixed", "cosine", "plateau"}:
        raise ValueError(f"Unsupported scheduler: {scheduler_mode}")
    return {
        "mode": scheduler_mode,
        "warmup_epochs": max(0, int(train_cfg.get("warmup_epochs", 0))),
        "warmup_start_lr": float(train_cfg.get("warmup_start_lr", train_cfg.get("lr", 5e-5))),
        "min_lr": float(train_cfg.get("min_lr", train_cfg.get("lr", 5e-5))),
        "plateau_factor": float(train_cfg.get("plateau_factor", 0.5)),
        "plateau_patience": max(1, int(train_cfg.get("plateau_patience", 3))),
        "plateau_threshold": float(train_cfg.get("plateau_threshold", 1e-4)),
    }


def _resolve_data_split_cfg(config: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    split_mode = str(data_cfg.get("split_mode", "loso")).strip().lower()
    val_ratio = float(data_cfg.get("val_ratio", train_cfg.get("val_split", 0.1)))
    default_test = 0.1 if split_mode in {"random", "random_811", "random_stratified"} else 0.0
    test_ratio = float(data_cfg.get("test_ratio", default_test))
    return {
        "split_mode": split_mode,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def _compute_warmup_cosine_lr(
    *,
    epoch: int,
    total_epochs: int,
    target_lr: float,
    warmup_epochs: int,
    warmup_start_lr: float,
    min_lr: float,
) -> float:
    if total_epochs <= 1:
        return float(target_lr)

    warmup_epochs = max(0, int(warmup_epochs))
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        if warmup_epochs == 1:
            return float(target_lr)
        ratio = float(epoch - 1) / float(max(1, warmup_epochs - 1))
        return float(warmup_start_lr + ratio * (target_lr - warmup_start_lr))

    decay_steps = max(1, total_epochs - warmup_epochs)
    decay_pos = max(0.0, float(epoch - warmup_epochs - 1))
    decay_ratio = min(1.0, decay_pos / float(max(1, decay_steps - 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return float(min_lr + (target_lr - min_lr) * cosine)


def _resolve_resume_checkpoint_path(
    resume_spec: Optional[str],
    output_dir: str,
) -> Optional[str]:
    if not resume_spec:
        return None
    spec = str(resume_spec).strip()
    if not spec:
        return None

    lower = spec.lower()
    keyword_map = {
        "last": "last.pt",
        "last.pt": "last.pt",
        "best_full": "best_full.pt",
        "best_full.pt": "best_full.pt",
        "best_proxy": "best_proxy.pt",
        "best_proxy.pt": "best_proxy.pt",
    }
    mapped = keyword_map.get(lower)
    if mapped is not None:
        return os.path.join(output_dir, mapped)

    if os.path.isfile(spec):
        return spec
    return os.path.join(output_dir, spec)


def _resolve_monitor_value(
    monitor_name: str,
    *,
    full_metrics: Optional[Dict[str, Any]],
    proxy_metrics: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], str]:
    monitor = str(monitor_name or "full_mean_UA").strip().lower()
    if monitor in {"full", "full_mean_ua"}:
        if full_metrics is not None:
            return float(full_metrics["mean_ua"]), "full_mean_UA"
        if proxy_metrics is not None:
            return float(proxy_metrics["mean_ua"]), "proxy_mean_UA"
        return None, ""
    if monitor in {"proxy", "proxy_mean_ua"}:
        if proxy_metrics is not None:
            return float(proxy_metrics["mean_ua"]), "proxy_mean_UA"
        if full_metrics is not None:
            return float(full_metrics["mean_ua"]), "full_mean_UA"
        return None, ""
    raise ValueError(f"Unsupported monitor metric: {monitor_name}")


def _resolve_model_cfg(config: Dict[str, Any], noise_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    model_cfg = dict(config.get("model", {}))
    variant = str(model_cfg.get("variant", "mlkd_baseline"))
    use_dual_default = variant == "adaptive_dualpath_mlkd"
    use_enh_default = variant == "adaptive_dualpath_mlkd"

    ablation = str(model_cfg.get("ablation", "")).strip().lower()
    default_gate_condition_mode = str(
        model_cfg.get(
            "gate_condition_mode",
            "oracle"
            if bool(model_cfg.get("use_oracle_snr", False) or model_cfg.get("use_oracle_noise", False))
            else "none",
        )
    ).strip().lower()
    if default_gate_condition_mode not in {"none", "oracle", "predicted", "hybrid"}:
        default_gate_condition_mode = "none"

    resolved = {
        "variant": variant,
        "teacher_backbone_name": model_cfg.get("teacher_backbone_name"),
        "student_backbone_name": model_cfg.get("student_backbone_name"),
        "student_num_hidden_layers": model_cfg.get(
            "student_num_hidden_layers", None if variant == "adaptive_dualpath_mlkd" else 6
        ),
        "use_dual_path": bool(model_cfg.get("use_dual_path", use_dual_default)),
        "use_enhancement": bool(model_cfg.get("use_enhancement", use_enh_default)),
        "enhancement_type": str(model_cfg.get("enhancement_type", "resunet")),
        "enhancement_channels": int(model_cfg.get("enhancement_channels", 32)),
        "separate_branch_backbones": bool(model_cfg.get("separate_branch_backbones", False)),
        "gate_type": str(model_cfg.get("gate_type", "estimated")),
        "gate_condition_mode": default_gate_condition_mode,
        "use_oracle_snr": bool(model_cfg.get("use_oracle_snr", False)),
        "use_oracle_noise": bool(model_cfg.get("use_oracle_noise", False)),
        "use_aux_snr": bool(model_cfg.get("use_aux_snr", False)),
        "use_aux_noise": bool(model_cfg.get("use_aux_noise", False)),
        "use_snr_regression": bool(model_cfg.get("use_snr_regression", False)),
        "condition_feat_dim": int(model_cfg.get("condition_feat_dim", 64)),
        "snr_gate_temperature": float(model_cfg.get("snr_gate_temperature", 10.0)),
        "snr_gate_bias_strength": float(model_cfg.get("snr_gate_bias_strength", 0.75)),
    }

    if ablation:
        if ablation in {"baseline", "mlkd_baseline"}:
            resolved["variant"] = "mlkd_baseline"
            resolved["use_dual_path"] = False
            resolved["use_enhancement"] = False
            resolved["gate_type"] = "none"
            resolved["gate_condition_mode"] = "none"
            resolved["use_aux_snr"] = False
            resolved["use_aux_noise"] = False
            resolved["use_snr_regression"] = False
        elif ablation in {"enhancement_only", "mlkd_plus_enhancement_only"}:
            resolved["variant"] = "adaptive_dualpath_mlkd"
            resolved["use_dual_path"] = False
            resolved["use_enhancement"] = True
            resolved["gate_type"] = "none"
            resolved["gate_condition_mode"] = "none"
            resolved["use_aux_snr"] = False
            resolved["use_aux_noise"] = False
            resolved["use_snr_regression"] = False
        elif ablation in {"dual_path_no_gate", "mlkd_plus_dualpath_without_gate"}:
            resolved["variant"] = "adaptive_dualpath_mlkd"
            resolved["use_dual_path"] = True
            resolved["use_enhancement"] = True
            resolved["gate_type"] = "none"
            resolved["gate_condition_mode"] = "none"
            resolved["use_aux_snr"] = False
            resolved["use_aux_noise"] = False
            resolved["use_snr_regression"] = False
        elif ablation in {
            "dual_path_gate_oracle",
            "dual_path_gate",
            "mlkd_plus_dualpath_gate",
            "dual_path_gate_aux",
            "mlkd_plus_dualpath_gate_aux",
        }:
            resolved["variant"] = "adaptive_dualpath_mlkd"
            resolved["use_dual_path"] = True
            resolved["use_enhancement"] = True
            resolved["gate_type"] = "estimated"
            resolved["gate_condition_mode"] = "oracle"
            resolved["use_oracle_snr"] = True
            resolved["use_oracle_noise"] = True
            resolved["use_aux_snr"] = True
            resolved["use_aux_noise"] = True
            resolved["use_snr_regression"] = False
        elif ablation == "dual_path_gate_pred":
            resolved["variant"] = "adaptive_dualpath_mlkd"
            resolved["use_dual_path"] = True
            resolved["use_enhancement"] = True
            resolved["gate_type"] = "estimated"
            resolved["gate_condition_mode"] = "predicted"
            resolved["use_oracle_snr"] = True
            resolved["use_oracle_noise"] = True
            resolved["use_aux_snr"] = True
            resolved["use_aux_noise"] = True
            resolved["use_snr_regression"] = False
        elif ablation == "dual_path_gate_pred_reg":
            resolved["variant"] = "adaptive_dualpath_mlkd"
            resolved["use_dual_path"] = True
            resolved["use_enhancement"] = True
            resolved["gate_type"] = "estimated"
            resolved["gate_condition_mode"] = "predicted"
            resolved["use_oracle_snr"] = True
            resolved["use_oracle_noise"] = True
            resolved["use_aux_snr"] = True
            resolved["use_aux_noise"] = True
            resolved["use_snr_regression"] = True
        elif ablation == "dual_path_gate_hybrid":
            resolved["variant"] = "adaptive_dualpath_mlkd"
            resolved["use_dual_path"] = True
            resolved["use_enhancement"] = True
            resolved["gate_type"] = "estimated"
            resolved["gate_condition_mode"] = "hybrid"
            resolved["use_oracle_snr"] = True
            resolved["use_oracle_noise"] = True
            resolved["use_aux_snr"] = True
            resolved["use_aux_noise"] = True
            resolved["use_snr_regression"] = False

    if noise_cfg is not None:
        resolved["num_snr_classes"] = len(noise_cfg.get("snrs", []))
        resolved["num_noise_classes"] = len(noise_cfg.get("types", []))
    else:
        resolved["num_snr_classes"] = int(model_cfg.get("num_snr_classes", 0))
        resolved["num_noise_classes"] = int(model_cfg.get("num_noise_classes", 0))

    if not resolved["use_dual_path"] or str(resolved["gate_type"]).strip().lower() == "none":
        resolved["gate_condition_mode"] = "none"

    return resolved


def _prepare_aux_labels(
    batch: Dict[str, Any], device: torch.device
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    snr_ids = batch.get("snr_ids")
    noise_ids = batch.get("noise_type_ids")
    snr_values = batch.get("snr_values")
    if snr_ids is not None:
        snr_ids = snr_ids.to(device)
    if noise_ids is not None:
        noise_ids = noise_ids.to(device)
    if snr_values is not None:
        snr_values = snr_values.to(device)
    return snr_ids, noise_ids, snr_values


def _resolve_train_gate_runtime(
    model_cfg: Dict[str, Any], train_cfg: Dict[str, Any], epoch: int
) -> Tuple[str, Optional[float]]:
    gate_source = str(model_cfg.get("gate_condition_mode", "none")).strip().lower()
    if gate_source not in {"none", "oracle", "predicted", "hybrid"}:
        gate_source = "none"
    if gate_source != "hybrid":
        return gate_source, None

    warmup_epochs = max(0, int(train_cfg.get("hybrid_warmup_epochs", 0)))
    transition_epochs = max(1, int(train_cfg.get("hybrid_transition_epochs", 1)))
    if epoch <= warmup_epochs:
        alpha = 0.0
    else:
        alpha = min(1.0, float(epoch - warmup_epochs) / float(transition_epochs))
    return "hybrid", alpha


def _resolve_eval_gate_source(model_cfg: Dict[str, Any], eval_condition_mode: str) -> str:
    if not bool(model_cfg.get("use_dual_path", False)):
        return "none"
    if str(model_cfg.get("gate_type", "none")).strip().lower() == "none":
        return "none"
    mode = str(eval_condition_mode or "deploy").strip().lower()
    if mode not in {"deploy", "oracle"}:
        mode = "deploy"
    return "oracle" if mode == "oracle" else "predicted"


def _build_student_forward_kwargs(
    *,
    gate_source: str,
    hybrid_alpha: Optional[float],
    snr_values: Optional[torch.Tensor],
    snr_ids: Optional[torch.Tensor],
    noise_ids: Optional[torch.Tensor],
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"condition_source": str(gate_source)}
    if gate_source == "hybrid" and hybrid_alpha is not None:
        kwargs["hybrid_alpha"] = float(hybrid_alpha)

    if gate_source in {"oracle", "hybrid"}:
        if snr_values is not None:
            kwargs["snr_values"] = snr_values
        if snr_ids is not None:
            kwargs["snr_ids"] = snr_ids
        if noise_ids is not None:
            kwargs["noise_ids"] = noise_ids
    return kwargs


def _update_gate_stats(stats: Dict[str, float], gate_tensor: Optional[torch.Tensor]) -> None:
    if gate_tensor is None:
        return
    gate_vals = gate_tensor.detach().float().reshape(-1)
    if gate_vals.numel() == 0:
        return
    stats["sum"] += float(gate_vals.sum().item())
    stats["sq_sum"] += float((gate_vals * gate_vals).sum().item())
    stats["count"] += int(gate_vals.numel())


def _finalize_gate_stats(stats: Dict[str, float]) -> Tuple[float, float]:
    count = max(1, int(stats.get("count", 0)))
    mean = float(stats.get("sum", 0.0)) / float(count)
    second_moment = float(stats.get("sq_sum", 0.0)) / float(count)
    var = max(0.0, second_moment - mean * mean)
    return float(mean), float(math.sqrt(var))


def _build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn
        if hasattr(dataset, "collate_fn")
        else None,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=True,
    )


def _append_log(csv_path: str, header: List[str], row: Dict) -> None:
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_resume(
    ckpt_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[Any] = None,
) -> int:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"resume_ckpt not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    load_result = model.load_state_dict(ckpt["model_state"], strict=False)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if missing or unexpected:
        print(
            f"Resume load_state_dict(non-strict) | missing={len(missing)} unexpected={len(unexpected)}"
        )
    if "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scaler_state" in ckpt and scaler is not None:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    last_epoch = int(ckpt.get("epoch", 0))
    return last_epoch + 1


def _resolve_eval_noise_type(noise_type: str, available_types: List[str]) -> str:
    if noise_type in available_types:
        return noise_type

    alias_map = {
        "hfchannel": "hf",
    }
    resolved = alias_map.get(noise_type)
    if resolved in available_types:
        return resolved
    raise ValueError(
        f"Unsupported validation noise_type={noise_type}; available types: {available_types}"
    )


def _display_eval_noise_type(noise_type: str) -> str:
    display_map = {
        "hf": "hfchannel",
    }
    return display_map.get(noise_type, noise_type)


def _build_student_val_loaders(
    clean_val_ds,
    noise_adder: NoiseAdder,
    noise_cfg: Dict,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    requested_types = list(noise_cfg.get("val_types") or noise_cfg["types"])
    requested_snrs = list(noise_cfg.get("val_snrs") or noise_cfg["snrs"])

    val_conditions = []
    for noise_type in requested_types:
        resolved_noise_type = _resolve_eval_noise_type(noise_type, noise_adder.noise_types)
        display_noise_type = _display_eval_noise_type(noise_type)
        for snr in requested_snrs:
            val_ds = NoisyPairDataset(
                clean_val_ds,
                noise_adder,
                is_train=False,
                fixed_noise_type=resolved_noise_type,
                fixed_snr=snr,
                rng=random.Random(seed),
                deterministic_eval=True,
                eval_seed=seed,
            )
            loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=noisy_collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed),
                pin_memory=True,
            )
            val_conditions.append(
                {
                    "noise_type": resolved_noise_type,
                    "display_name": display_noise_type,
                    "snr_db": float(snr),
                    "label": f"{display_noise_type}@{int(snr)}",
                    "loader": loader,
                }
            )
    return val_conditions


def _evaluate_student_with_teacher_conditions(
    student,
    teacher,
    val_conditions,
    device,
    loss_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    eval_condition_mode: str,
):
    aggregate = {
        "total": 0.0,
        "loss_ce": 0.0,
        "loss_kl": 0.0,
        "loss_mse": 0.0,
        "loss_enh": 0.0,
        "loss_snr": 0.0,
        "loss_noise": 0.0,
        "loss_snr_reg": 0.0,
        "gate_mean": 0.0,
        "gate_std": 0.0,
    }
    condition_metrics = []
    eval_gate_source = _resolve_eval_gate_source(model_cfg, eval_condition_mode)

    for condition in val_conditions:
        metrics = _evaluate_student_with_teacher(
            student,
            teacher,
            condition["loader"],
            device,
            loss_cfg,
            model_cfg,
            gate_source=eval_gate_source,
            hybrid_alpha=None,
        )
        condition_metrics.append(
            {
                "noise_type": condition["noise_type"],
                "display_name": condition["display_name"],
                "snr_db": condition["snr_db"],
                "label": condition["label"],
                **metrics,
            }
        )
        for key in aggregate:
            aggregate[key] += metrics[key]

    num_conditions = max(1, len(condition_metrics))
    condition_uas = {item["label"]: item["ua"] for item in condition_metrics}
    mean_ua = sum(condition_uas.values()) / num_conditions

    return {
        "total": aggregate["total"] / num_conditions,
        "LC": aggregate["loss_ce"] / num_conditions,
        "LKL": aggregate["loss_kl"] / num_conditions,
        "LMSE": aggregate["loss_mse"] / num_conditions,
        "loss_ce": aggregate["loss_ce"] / num_conditions,
        "loss_kl": aggregate["loss_kl"] / num_conditions,
        "loss_mse": aggregate["loss_mse"] / num_conditions,
        "loss_enh": aggregate["loss_enh"] / num_conditions,
        "loss_snr": aggregate["loss_snr"] / num_conditions,
        "loss_noise": aggregate["loss_noise"] / num_conditions,
        "loss_snr_reg": aggregate["loss_snr_reg"] / num_conditions,
        "gate_mean": aggregate["gate_mean"] / num_conditions,
        "gate_std": aggregate["gate_std"] / num_conditions,
        "ua": mean_ua,
        "mean_ua": mean_ua,
        "condition_uas": condition_uas,
        "condition_metrics": condition_metrics,
        "eval_gate_source": eval_gate_source,
    }


def _print_condition_ua_summary(condition_metrics) -> None:
    grouped = {}
    for item in condition_metrics:
        grouped.setdefault(item["display_name"], []).append(item)

    for display_name, items in grouped.items():
        items = sorted(items, key=lambda x: x["snr_db"])
        summary = " ".join(f"{item['label']}={item['ua']:.4f}" for item in items)
        print(f"Val UA | {summary}")


def _format_snr_label(snr: float) -> str:
    return str(int(snr)) if float(snr).is_integer() else f"{float(snr):.2f}"


def _resolve_student_val_cfg(config: Dict, output_dir: str) -> Dict[str, Any]:
    val_cfg = dict(config.get("student_val", {}))
    proxy_manifest_path = val_cfg.get("proxy_manifest_path") or os.path.join(
        output_dir, "proxy_val_manifest.json"
    )
    full_manifest_path = val_cfg.get("full_manifest_path") or os.path.join(
        output_dir, "full_val_manifest.json"
    )
    return {
        "proxy_eval_enabled": bool(val_cfg.get("proxy_eval_enabled", True)),
        "proxy_ratio": float(val_cfg.get("proxy_ratio", 0.15)),
        "min_samples_per_condition": int(val_cfg.get("min_samples_per_condition", 4)),
        "full_eval_every": int(val_cfg.get("full_eval_every", val_cfg.get("full_val_every", 5))),
        "top_k_checkpoints": int(val_cfg.get("top_k_checkpoints", 5)),
        "trigger_full_eval_on_proxy_best": bool(
            val_cfg.get("trigger_full_eval_on_proxy_best", False)
        ),
        "final_rerank_with_full_eval": bool(val_cfg.get("final_rerank_with_full_eval", True)),
        "val_manifest_seed": int(val_cfg.get("val_manifest_seed", 42)),
        "proxy_manifest_path": str(proxy_manifest_path),
        "full_manifest_path": str(full_manifest_path),
    }


def _collect_clean_eval_records(clean_ds: IemocapDataset) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(clean_ds.records):
        wav, _, label, speaker_id, utt_id = clean_ds[idx]
        records.append(
            {
                "clean_index": idx,
                "target_length": int(wav.shape[0]),
                "label_id": int(label),
                "speaker_id": str(speaker_id),
                "utt_id": str(utt_id),
                "utterance_path": str(row.get("path", "")),
            }
        )
    return records


def _generate_manifest_entries(
    *,
    manifest_type: str,
    clean_records: List[Dict[str, Any]],
    noise_adder: NoiseAdder,
    noise_types: List[str],
    snr_levels: List[float],
    seed: int,
    proxy_ratio: float,
    min_samples_per_condition: int,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    total_clean = len(clean_records)
    all_indices = list(range(total_clean))

    for noise_type in noise_types:
        resolved_noise_type = _resolve_eval_noise_type(noise_type, noise_adder.noise_types)
        for snr_db in snr_levels:
            if manifest_type == "proxy":
                keep_n = max(min_samples_per_condition, int(math.ceil(total_clean * proxy_ratio)))
                keep_n = min(total_clean, keep_n)
                cond_rng = random.Random(f"{seed}|proxy|{resolved_noise_type}|{float(snr_db):.4f}")
                selected = all_indices.copy()
                cond_rng.shuffle(selected)
                selected = sorted(selected[:keep_n])
            else:
                selected = all_indices

            for clean_idx in selected:
                rec = clean_records[clean_idx]
                target_length = int(rec["target_length"])
                sample_rng = random.Random(
                    f"{seed}|{manifest_type}|{clean_idx}|{resolved_noise_type}|{float(snr_db):.4f}"
                )
                noise_paths = noise_adder.noise_files[resolved_noise_type]
                noise_source_path = str(sample_rng.choice(noise_paths))
                noise_signal = noise_adder._load_noise(noise_source_path)
                if noise_signal.numel() > target_length:
                    noise_offset = int(sample_rng.randint(0, int(noise_signal.numel() - target_length)))
                else:
                    noise_offset = int(sample_rng.randint(0, max(0, int(noise_signal.numel()) - 1)))

                entries.append(
                    {
                        "sample_id": f"{rec['utt_id']}|{resolved_noise_type}|{_format_snr_label(float(snr_db))}",
                        "clean_index": int(rec["clean_index"]),
                        "utt_id": rec["utt_id"],
                        "utterance_path": rec["utterance_path"],
                        "label_id": int(rec["label_id"]),
                        "speaker_id": rec["speaker_id"],
                        "noise_type": resolved_noise_type,
                        "snr_db": float(snr_db),
                        "noise_source_path": noise_source_path,
                        "noise_source_id": os.path.basename(noise_source_path),
                        "noise_offset": int(noise_offset),
                        "target_length": target_length,
                        "mix_seed": int(seed),
                    }
                )
    return entries


def _build_or_load_val_manifest(
    *,
    manifest_path: str,
    manifest_type: str,
    clean_records: List[Dict[str, Any]],
    noise_adder: NoiseAdder,
    noise_cfg: Dict[str, Any],
    seed: int,
    proxy_ratio: float,
    min_samples_per_condition: int,
) -> Dict[str, Any]:
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    requested_types = list(noise_cfg.get("val_types") or noise_cfg["types"])
    requested_snrs = [float(s) for s in (noise_cfg.get("val_snrs") or noise_cfg["snrs"])]
    entries = _generate_manifest_entries(
        manifest_type=manifest_type,
        clean_records=clean_records,
        noise_adder=noise_adder,
        noise_types=requested_types,
        snr_levels=requested_snrs,
        seed=seed,
        proxy_ratio=proxy_ratio,
        min_samples_per_condition=min_samples_per_condition,
    )
    manifest = {
        "manifest_type": manifest_type,
        "seed": int(seed),
        "noise_types": [_resolve_eval_noise_type(nt, noise_adder.noise_types) for nt in requested_types],
        "snr_levels": requested_snrs,
        "num_clean_samples": len(clean_records),
        "proxy_ratio": float(proxy_ratio),
        "min_samples_per_condition": int(min_samples_per_condition),
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "entries": entries,
    }
    ensure_dir(os.path.dirname(manifest_path) or ".")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def _build_manifest_val_loaders(
    clean_val_ds: IemocapDataset,
    noise_adder: NoiseAdder,
    manifest: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> List[Dict[str, Any]]:
    by_condition: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for entry in manifest["entries"]:
        key = (str(entry["noise_type"]), float(entry["snr_db"]))
        by_condition.setdefault(key, []).append(entry)

    val_conditions: List[Dict[str, Any]] = []
    for noise_type in manifest["noise_types"]:
        display_name = _display_eval_noise_type(str(noise_type))
        for snr in manifest["snr_levels"]:
            key = (str(noise_type), float(snr))
            cond_entries = by_condition.get(key, [])
            if not cond_entries:
                continue
            val_ds = ManifestNoisyPairDataset(clean_val_ds, noise_adder, cond_entries)
            loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=noisy_collate_fn,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(seed),
                pin_memory=True,
            )
            label = f"{display_name}@{_format_snr_label(float(snr))}"
            val_conditions.append(
                {
                    "noise_type": str(noise_type),
                    "display_name": display_name,
                    "snr_db": float(snr),
                    "label": label,
                    "loader": loader,
                }
            )
    return val_conditions


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _save_student_checkpoint(
    path: str,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    scheduler: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    model_meta = {}
    if hasattr(student, "get_model_meta"):
        model_meta = dict(student.get_model_meta())
    payload = {
        "model_state": student.state_dict(),
        "model_config": student.backbone.config.to_dict(),
        "model_meta": model_meta,
        "model_variant": model_meta.get("variant", "mlkd_baseline"),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
    }
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _sort_proxy_topk(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: (-float(x["proxy_mean_ua"]), int(x["epoch"])))


def _update_proxy_topk(
    topk_items: List[Dict[str, Any]],
    candidate: Dict[str, Any],
    max_k: int,
) -> Tuple[List[Dict[str, Any]], bool, Optional[Dict[str, Any]]]:
    previous_paths = {str(item["checkpoint_path"]) for item in topk_items}
    merged = [item for item in topk_items if int(item["epoch"]) != int(candidate["epoch"])]
    merged.append(candidate)
    merged = _sort_proxy_topk(merged)
    trimmed = merged[: max(1, max_k)]
    new_paths = {str(item["checkpoint_path"]) for item in trimmed}
    entered = str(candidate["checkpoint_path"]) in new_paths

    removed = None
    removed_paths = previous_paths - new_paths
    if removed_paths:
        removed_path = sorted(removed_paths)[0]
        removed = next((item for item in merged if str(item["checkpoint_path"]) == removed_path), None)
    return trimmed, entered, removed


def _format_topk_summary(topk_items: List[Dict[str, Any]]) -> str:
    return "; ".join(
        f"e{int(item['epoch'])}:{float(item['proxy_mean_ua']):.4f}" for item in topk_items
    )


def _read_checkpoint_metric(ckpt_path: str, keys: List[str], default: float = -1.0) -> float:
    if not os.path.isfile(ckpt_path):
        return float(default)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for key in keys:
        if key in ckpt and ckpt[key] is not None:
            return float(ckpt[key])
    return float(default)


def train_teacher(
    metadata_csv: str,
    output_dir: str,
    fold_speaker: Optional[str],
    config: Dict,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    resume_ckpt: Optional[str] = None,
) -> str:
    train_cfg = _resolve_runtime_train_cfg(config)
    split_cfg = _resolve_data_split_cfg(config, train_cfg)
    model_cfg = _resolve_model_cfg(config, config.get("noise", {}))
    audio_cfg = config["audio"]

    epochs = int(epochs or train_cfg["epochs"])
    batch_size = int(batch_size or train_cfg["batch_size"])
    lr = float(lr or train_cfg["lr"])

    set_seed(train_cfg["seed"])
    ensure_dir(output_dir)

    df = load_metadata_df(metadata_csv)
    train_df, val_df, _ = split_train_val_test(
        df,
        split_mode=split_cfg["split_mode"],
        seed=train_cfg["seed"],
        fold_speaker=fold_speaker,
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    train_ds = IemocapDataset(
        train_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=True,
    )
    val_ds = IemocapDataset(
        val_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=False,
        eval_clip=audio_cfg.get("eval_clip", True),
    )

    train_loader = _build_loader(
        train_ds, batch_size, train_cfg["num_workers"], True, train_cfg["seed"]
    )
    val_loader = _build_loader(
        val_ds, batch_size, train_cfg["num_workers"], False, train_cfg["seed"]
    )

    device = get_device()
    model = Wav2Vec2SERTeacher(
        num_classes=4,
        backbone_name=model_cfg.get("teacher_backbone_name"),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    best_ua = -1.0
    best_path = ""
    save_every = int(train_cfg.get("save_every", 20))
    log_path = os.path.join(output_dir, "train_log.csv")
    log_header = [
        "timestamp",
        "epoch",
        "split",
        "lr",
        "train_time_s",
        "val_time_s",
        "epoch_time_s",
        "LC",
        "LKL",
        "LMSE",
        "total",
        "UA",
        "best_UA",
        "is_best",
    ]
    start_epoch = 1
    if resume_ckpt:
        start_epoch = _load_resume(resume_ckpt, model, optimizer, scaler)
        print(f"Resuming teacher from {resume_ckpt} at epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_count = 0

        train_start = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Teacher Epoch {epoch}", leave=False):
            inputs = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                outputs = model(inputs, attention_mask=attention_mask)
                logits = outputs["logits"]
                loss = torch.nn.functional.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)

        train_time_s = time.perf_counter() - train_start
        train_lc = total_loss / max(1, total_count)
        val_start = time.perf_counter()
        val_metrics = evaluate_classifier(
            model, val_loader, device, num_classes=4, input_key="input_values"
        )
        val_time_s = time.perf_counter() - val_start
        epoch_time_s = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]
        is_best = val_metrics["ua"] > best_ua

        log = (
            f"Epoch {epoch}/{epochs} | "
            f"lr={lr:.2e} "
            f"train LC={train_lc:.4f} LKL=0.0000 LMSE=0.0000 total={train_lc:.4f} | "
            f"val LC={val_metrics['loss']:.4f} LKL=0.0000 LMSE=0.0000 total={val_metrics['loss']:.4f} "
            f"UA={val_metrics['ua']:.4f} | "
            f"time train={train_time_s:.1f}s val={val_time_s:.1f}s epoch={epoch_time_s:.1f}s"
        )
        print(log)
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _append_log(
            log_path,
            log_header,
            {
                "timestamp": now,
                "epoch": epoch,
                "split": "train",
                "lr": lr,
                "train_time_s": train_time_s,
                "val_time_s": val_time_s,
                "epoch_time_s": epoch_time_s,
                "LC": train_lc,
                "LKL": 0.0,
                "LMSE": 0.0,
                "total": train_lc,
                "UA": "",
                "best_UA": max(best_ua, val_metrics["ua"]),
                "is_best": "",
            },
        )
        _append_log(
            log_path,
            log_header,
            {
                "timestamp": now,
                "epoch": epoch,
                "split": "val",
                "lr": lr,
                "train_time_s": train_time_s,
                "val_time_s": val_time_s,
                "epoch_time_s": epoch_time_s,
                "LC": val_metrics["loss"],
                "LKL": 0.0,
                "LMSE": 0.0,
                "total": val_metrics["loss"],
                "UA": val_metrics["ua"],
                "best_UA": max(best_ua, val_metrics["ua"]),
                "is_best": int(is_best),
            },
        )

        if is_best:
            best_ua = val_metrics["ua"]
            best_path = os.path.join(output_dir, "best_teacher.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.backbone.config.to_dict(),
                    "model_meta": {
                        "variant": "teacher",
                        "teacher_backbone_name": model_cfg.get("teacher_backbone_name"),
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                },
                best_path,
            )
        if save_every > 0 and (epoch % save_every == 0):
            ckpt_path = os.path.join(output_dir, f"teacher_epoch_{epoch}.pt")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.backbone.config.to_dict(),
                    "model_meta": {
                        "variant": "teacher",
                        "teacher_backbone_name": model_cfg.get("teacher_backbone_name"),
                    },
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

    return best_path


@torch.no_grad()
def _evaluate_student_with_teacher(
    student,
    teacher,
    dataloader,
    device,
    loss_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    gate_source: str,
    hybrid_alpha: Optional[float] = None,
):
    student.eval()
    teacher.eval()

    all_labels = []
    all_preds = []
    total_loss = 0.0
    total_lc = 0.0
    total_lkl = 0.0
    total_lmse = 0.0
    total_lenh = 0.0
    total_lsnr = 0.0
    total_lnoise = 0.0
    total_lsnr_reg = 0.0
    total_count = 0
    gate_stats = {"sum": 0.0, "sq_sum": 0.0, "count": 0}

    for batch in dataloader:
        clean = batch["clean_input_values"].to(device)
        noisy = batch["noisy_input_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        snr_ids, noise_ids, snr_values = _prepare_aux_labels(batch, device)

        t_out = teacher(clean, attention_mask=attention_mask)
        forward_kwargs = _build_student_forward_kwargs(
            gate_source=gate_source,
            hybrid_alpha=hybrid_alpha,
            snr_values=snr_values,
            snr_ids=snr_ids,
            noise_ids=noise_ids,
        )
        s_out = student(
            noisy,
            attention_mask=attention_mask,
            **forward_kwargs,
        )

        snr_logits_for_loss = s_out.get("snr_logits_pred")
        if snr_logits_for_loss is None:
            snr_logits_for_loss = s_out.get("snr_logits")
        noise_logits_for_loss = s_out.get("noise_logits_pred")
        if noise_logits_for_loss is None:
            noise_logits_for_loss = s_out.get("noise_logits")

        loss_dict = compute_adaptive_mlkd_loss(
            logits_s=s_out["logits"],
            logits_t=t_out["logits"],
            student_states=s_out["hidden_states"],
            teacher_states=t_out["hidden_states"],
            labels=labels,
            mask=s_out["feature_mask"],
            temperature=loss_cfg["temperature"],
            lambda_ce=loss_cfg["lambda_ce"],
            lambda_kl=loss_cfg["lambda_kl"],
            lambda_mse=loss_cfg["lambda_mse"],
            lambda_enh=loss_cfg["lambda_enh"],
            lambda_snr=loss_cfg["lambda_snr"],
            lambda_noise=loss_cfg["lambda_noise"],
            lambda_snr_reg=loss_cfg.get("lambda_snr_reg", 0.0),
            distill_layers=loss_cfg["distill_layers"],
            enhanced_wav=s_out.get("enhanced_wav"),
            clean_wav=clean,
            w_wav=loss_cfg["w_wav"],
            w_stft=loss_cfg["w_stft"],
            snr_logits=snr_logits_for_loss,
            snr_labels=snr_ids if model_cfg.get("use_aux_snr", False) else None,
            noise_logits=noise_logits_for_loss,
            noise_labels=noise_ids if model_cfg.get("use_aux_noise", False) else None,
            snr_value_pred=s_out.get("snr_value_pred"),
            snr_values=snr_values if model_cfg.get("use_snr_regression", False) else None,
            snr_reg_loss_type=loss_cfg.get("snr_reg_loss_type", "smooth_l1"),
        )

        total_loss += loss_dict["total"].item() * labels.size(0)
        total_lc += loss_dict["loss_ce"].item() * labels.size(0)
        total_lkl += loss_dict["loss_kl"].item() * labels.size(0)
        total_lmse += loss_dict["loss_mse"].item() * labels.size(0)
        total_lenh += loss_dict["loss_enh"].item() * labels.size(0)
        total_lsnr += loss_dict["loss_snr"].item() * labels.size(0)
        total_lnoise += loss_dict["loss_noise"].item() * labels.size(0)
        total_lsnr_reg += loss_dict["loss_snr_reg"].item() * labels.size(0)
        total_count += labels.size(0)
        _update_gate_stats(gate_stats, s_out.get("gate"))

        preds = torch.argmax(s_out["logits"], dim=-1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    from src.training.metrics import confusion_matrix, unweighted_accuracy

    cm = confusion_matrix(all_labels, all_preds, num_classes=4)
    ua = unweighted_accuracy(cm)
    gate_mean, gate_std = _finalize_gate_stats(gate_stats)
    return {
        "total": total_loss / max(1, total_count),
        "LC": total_lc / max(1, total_count),
        "LKL": total_lkl / max(1, total_count),
        "LMSE": total_lmse / max(1, total_count),
        "loss_ce": total_lc / max(1, total_count),
        "loss_kl": total_lkl / max(1, total_count),
        "loss_mse": total_lmse / max(1, total_count),
        "loss_enh": total_lenh / max(1, total_count),
        "loss_snr": total_lsnr / max(1, total_count),
        "loss_noise": total_lnoise / max(1, total_count),
        "loss_snr_reg": total_lsnr_reg / max(1, total_count),
        "gate_mean": gate_mean,
        "gate_std": gate_std,
        "ua": ua,
    }


def train_student_mlkd(
    metadata_csv: str,
    teacher_ckpt: str,
    output_dir: str,
    fold_speaker: Optional[str],
    config: Dict,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    resume_ckpt: Optional[str] = None,
) -> str:
    train_cfg = _resolve_runtime_train_cfg(config)
    split_cfg = _resolve_data_split_cfg(config, train_cfg)
    audio_cfg = config["audio"]
    noise_cfg = config["noise"]
    model_cfg = _resolve_model_cfg(config, noise_cfg)
    loss_schedule_cfg = _resolve_loss_schedule_cfg(config)
    val_cfg = _resolve_student_val_cfg(config, output_dir)

    epochs = int(train_cfg["epochs"] if epochs is None else epochs)
    batch_size = int(train_cfg["batch_size"] if batch_size is None else batch_size)
    target_lr = float(train_cfg["lr"] if lr is None else lr)
    scheduler_cfg = _resolve_scheduler_cfg({**train_cfg, "lr": target_lr})
    scheduler_mode = scheduler_cfg["mode"]

    set_seed(train_cfg["seed"])
    ensure_dir(output_dir)

    df = load_metadata_df(metadata_csv)
    train_df, val_df, _ = split_train_val_test(
        df,
        split_mode=split_cfg["split_mode"],
        seed=train_cfg["seed"],
        fold_speaker=fold_speaker,
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )

    clean_train_ds = IemocapDataset(
        train_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=True,
    )
    clean_val_ds = IemocapDataset(
        val_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=False,
        eval_clip=audio_cfg.get("eval_clip", True),
    )
    clean_eval_records = _collect_clean_eval_records(clean_val_ds)

    noise_adder = NoiseAdder(
        noise_root=config["paths"]["noise_root"],
        noise_types=noise_cfg["types"],
        snr_levels=noise_cfg["snrs"],
        sample_rate=audio_cfg["sample_rate"],
    )

    proxy_manifest = _build_or_load_val_manifest(
        manifest_path=val_cfg["proxy_manifest_path"],
        manifest_type="proxy",
        clean_records=clean_eval_records,
        noise_adder=noise_adder,
        noise_cfg=noise_cfg,
        seed=val_cfg["val_manifest_seed"],
        proxy_ratio=val_cfg["proxy_ratio"],
        min_samples_per_condition=val_cfg["min_samples_per_condition"],
    )
    full_manifest = _build_or_load_val_manifest(
        manifest_path=val_cfg["full_manifest_path"],
        manifest_type="full",
        clean_records=clean_eval_records,
        noise_adder=noise_adder,
        noise_cfg=noise_cfg,
        seed=val_cfg["val_manifest_seed"],
        proxy_ratio=val_cfg["proxy_ratio"],
        min_samples_per_condition=val_cfg["min_samples_per_condition"],
    )

    train_ds = NoisyPairDataset(
        clean_train_ds,
        noise_adder,
        is_train=True,
        rng=random.Random(train_cfg["seed"]),
    )

    def _build_noisy_loader(dataset, shuffle: bool):
        generator = torch.Generator()
        generator.manual_seed(train_cfg["seed"])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=train_cfg["num_workers"],
            collate_fn=noisy_collate_fn,
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=True,
        )

    train_loader = _build_noisy_loader(train_ds, True)
    proxy_conditions = _build_manifest_val_loaders(
        clean_val_ds,
        noise_adder,
        proxy_manifest,
        batch_size=batch_size,
        num_workers=train_cfg["num_workers"],
        seed=val_cfg["val_manifest_seed"],
    )
    full_conditions = _build_manifest_val_loaders(
        clean_val_ds,
        noise_adder,
        full_manifest,
        batch_size=batch_size,
        num_workers=train_cfg["num_workers"],
        seed=val_cfg["val_manifest_seed"],
    )

    device = get_device()
    teacher = Wav2Vec2SERTeacher(
        num_classes=4,
        backbone_name=model_cfg.get("teacher_backbone_name"),
    ).to(device)
    teacher_state = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_state["model_state"])
    teacher.eval()
    if bool(train_cfg.get("freeze_teacher", True)):
        for p in teacher.parameters():
            p.requires_grad = False

    student_cfg_overrides: Dict[str, Any] = {}
    student_backbone_name = model_cfg.get("student_backbone_name")
    if student_backbone_name and str(student_backbone_name).lower() not in {"teacher", "none"}:
        local_only = os.path.isdir(str(student_backbone_name))
        student_backbone_cfg = Wav2Vec2Config.from_pretrained(
            str(student_backbone_name),
            local_files_only=local_only,
        )
        student_cfg_overrides.update(student_backbone_cfg.to_dict())
    if model_cfg.get("student_num_hidden_layers") is not None:
        student_cfg_overrides["num_hidden_layers"] = int(model_cfg["student_num_hidden_layers"])
    student = Wav2Vec2SERStudent.from_teacher(
        teacher,
        num_classes=4,
        student_cfg_overrides=student_cfg_overrides,
        model_args=model_cfg,
    ).to(device)
    optimizer = AdamW(student.parameters(), lr=target_lr)
    if scheduler_mode == "cosine":
        _set_optimizer_lr(optimizer, scheduler_cfg["warmup_start_lr"])

    plateau_scheduler = None
    if scheduler_mode == "plateau":
        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg["plateau_factor"],
            patience=scheduler_cfg["plateau_patience"],
            threshold=scheduler_cfg["plateau_threshold"],
        )
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
    monitor_name = str(train_cfg.get("monitor", "full_mean_UA"))
    eval_condition_mode = str(train_cfg.get("eval_condition_mode", "deploy")).strip().lower()
    if eval_condition_mode not in {"deploy", "oracle"}:
        eval_condition_mode = "deploy"
    early_stopping_patience = int(
        train_cfg.get("early_stop_patience", train_cfg.get("early_stopping", 0)) or 0
    )
    save_best_full = bool(train_cfg.get("save_best_full", True))
    save_best_proxy = bool(train_cfg.get("save_best_proxy", True))

    save_every = int(train_cfg.get("save_every", 20))
    log_path = os.path.join(output_dir, "train_log.csv")
    topk_path = os.path.join(output_dir, "proxy_topk.json")
    full_history_path = os.path.join(output_dir, "full_eval_history.json")
    last_ckpt_path = os.path.join(output_dir, "last.pt")
    best_proxy_ckpt_path = os.path.join(output_dir, "best_proxy.pt")
    best_full_ckpt_path = os.path.join(output_dir, "best_full.pt")
    condition_headers = [condition["label"] for condition in full_conditions]
    log_header = [
        "timestamp",
        "epoch",
        "split",
        "eval_stage",
        "lr",
        "train_time_s",
        "proxy_val_time_s",
        "full_val_time_s",
        "epoch_time_s",
        "LC",
        "LKL",
        "LMSE",
        "loss_ce",
        "loss_kl",
        "loss_mse",
        "loss_enh",
        "loss_snr",
        "loss_noise",
        "loss_snr_reg",
        "gate_mean",
        "gate_std",
        "total",
        "proxy_mean_UA",
        "full_mean_UA",
        "best_proxy_mean_UA",
        "best_full_mean_UA",
        "is_proxy_best",
        "is_full_best",
        "monitor_metric",
        "monitor_value",
        "best_monitor_value",
        "early_stop_counter",
        "current_full_mean_UA",
        "loss_stage",
        "lambda_ce",
        "lambda_kl",
        "lambda_mse",
        "lambda_enh",
        "lambda_snr",
        "lambda_noise",
        "lambda_snr_reg",
        "gate_condition_mode",
        "eval_condition_mode",
        "train_gate_source",
        "eval_gate_source",
        "hybrid_alpha",
        "grad_clip",
        "proxy_topk",
        "noise_types",
        "snr_db",
    ] + condition_headers

    _save_json(
        os.path.join(output_dir, "val_strategy.json"),
        {
            "proxy_ratio": val_cfg["proxy_ratio"],
            "min_samples_per_condition": val_cfg["min_samples_per_condition"],
            "proxy_eval_enabled": val_cfg["proxy_eval_enabled"],
            "full_eval_every": val_cfg["full_eval_every"],
            "top_k_checkpoints": val_cfg["top_k_checkpoints"],
            "trigger_full_eval_on_proxy_best": val_cfg["trigger_full_eval_on_proxy_best"],
            "final_rerank_with_full_eval": val_cfg["final_rerank_with_full_eval"],
            "val_manifest_seed": val_cfg["val_manifest_seed"],
            "proxy_manifest_path": val_cfg["proxy_manifest_path"],
            "full_manifest_path": val_cfg["full_manifest_path"],
            "proxy_manifest_entries": len(proxy_manifest["entries"]),
            "full_manifest_entries": len(full_manifest["entries"]),
            "model": model_cfg,
            "loss_schedule": loss_schedule_cfg,
            "optimizer": {
                "target_lr": target_lr,
                "grad_clip": grad_clip,
            },
            "scheduler": scheduler_cfg,
            "monitor": monitor_name,
            "eval_condition_mode": eval_condition_mode,
            "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
            "hybrid_warmup_epochs": int(train_cfg.get("hybrid_warmup_epochs", 0)),
            "hybrid_transition_epochs": int(train_cfg.get("hybrid_transition_epochs", 10)),
            "early_stop_patience": early_stopping_patience,
            "save_best_full": save_best_full,
            "save_best_proxy": save_best_proxy,
        },
    )
    print(
        "Val strategy | "
        f"proxy_eval_enabled={val_cfg['proxy_eval_enabled']} "
        f"proxy_ratio={val_cfg['proxy_ratio']} "
        f"min_samples_per_condition={val_cfg['min_samples_per_condition']} "
        f"full_eval_every={val_cfg['full_eval_every']} "
        f"top_k_checkpoints={val_cfg['top_k_checkpoints']} "
        f"trigger_full_eval_on_proxy_best={val_cfg['trigger_full_eval_on_proxy_best']} "
        f"val_manifest_seed={val_cfg['val_manifest_seed']} "
        f"proxy_manifest_path={val_cfg['proxy_manifest_path']} "
        f"full_manifest_path={val_cfg['full_manifest_path']}"
    )
    print(
        "Train controls | "
        f"scheduler={scheduler_mode} target_lr={target_lr:.2e} "
        f"warmup_epochs={scheduler_cfg['warmup_epochs']} "
        f"warmup_start_lr={scheduler_cfg['warmup_start_lr']:.2e} "
        f"min_lr={scheduler_cfg['min_lr']:.2e} "
        f"monitor={monitor_name} early_stop_patience={early_stopping_patience} "
        f"gate_condition_mode={model_cfg.get('gate_condition_mode', 'none')} "
        f"eval_condition_mode={eval_condition_mode} "
        f"hybrid_warmup_epochs={int(train_cfg.get('hybrid_warmup_epochs', 0))} "
        f"hybrid_transition_epochs={int(train_cfg.get('hybrid_transition_epochs', 10))} "
        f"grad_clip={'off' if grad_clip <= 0 else grad_clip}"
    )

    start_epoch = 1
    resume_path = _resolve_resume_checkpoint_path(resume_ckpt, output_dir)
    resume_payload: Dict[str, Any] = {}
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"resume_ckpt not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location="cpu")
        start_epoch = _load_resume(
            resume_path,
            student,
            optimizer,
            scaler,
            scheduler=plateau_scheduler,
        )
        print(f"Resuming student from {resume_path} at epoch {start_epoch}")

    proxy_topk: List[Dict[str, Any]] = []
    if os.path.isfile(topk_path):
        with open(topk_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            proxy_topk = list(loaded.get("items", []))
    proxy_topk = [x for x in proxy_topk if os.path.isfile(str(x.get("checkpoint_path", "")))]
    proxy_topk = _sort_proxy_topk(proxy_topk)[: max(1, val_cfg["top_k_checkpoints"])]

    full_eval_history: List[Dict[str, Any]] = []
    if os.path.isfile(full_history_path):
        with open(full_history_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            full_eval_history = list(loaded.get("items", []))

    best_proxy_mean_ua = max([float(x["proxy_mean_ua"]) for x in proxy_topk], default=-1.0)
    best_proxy_mean_ua = max(
        best_proxy_mean_ua,
        _read_checkpoint_metric(
            best_proxy_ckpt_path, ["proxy_mean_ua", "best_metric_value"], default=-1.0
        ),
    )
    best_full_mean_ua = max([float(x["full_mean_ua"]) for x in full_eval_history], default=-1.0)
    best_full_mean_ua = max(
        best_full_mean_ua,
        _read_checkpoint_metric(best_full_ckpt_path, ["full_mean_ua", "best_metric_value"], -1.0),
    )
    if resume_payload:
        best_proxy_mean_ua = max(
            best_proxy_mean_ua, float(resume_payload.get("best_proxy_mean_ua", -1.0))
        )
        best_full_mean_ua = max(
            best_full_mean_ua, float(resume_payload.get("best_full_mean_ua", -1.0))
        )
    best_path = best_full_ckpt_path if os.path.isfile(best_full_ckpt_path) else ""
    early_stop_counter = int(resume_payload.get("early_stop_counter", 0) or 0)
    best_monitor_value = float(
        resume_payload.get(
            "best_monitor_value",
            best_full_mean_ua if "proxy" not in monitor_name.lower() else best_proxy_mean_ua,
        )
    )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()

        if scheduler_mode == "cosine":
            scheduled_lr = _compute_warmup_cosine_lr(
                epoch=epoch,
                total_epochs=epochs,
                target_lr=target_lr,
                warmup_epochs=scheduler_cfg["warmup_epochs"],
                warmup_start_lr=scheduler_cfg["warmup_start_lr"],
                min_lr=scheduler_cfg["min_lr"],
            )
            _set_optimizer_lr(optimizer, scheduled_lr)
        elif scheduler_mode == "fixed":
            _set_optimizer_lr(optimizer, target_lr)

        epoch_loss_cfg = get_loss_weights(epoch, config)
        if not model_cfg.get("use_enhancement", False):
            epoch_loss_cfg["lambda_enh"] = 0.0
        if not model_cfg.get("use_aux_snr", False):
            epoch_loss_cfg["lambda_snr"] = 0.0
        if not model_cfg.get("use_aux_noise", False):
            epoch_loss_cfg["lambda_noise"] = 0.0
        if not model_cfg.get("use_snr_regression", False):
            epoch_loss_cfg["lambda_snr_reg"] = 0.0

        train_gate_source, train_hybrid_alpha = _resolve_train_gate_runtime(
            model_cfg, train_cfg, epoch
        )
        eval_gate_source = _resolve_eval_gate_source(model_cfg, eval_condition_mode)

        freeze_epochs = int(train_cfg.get("freeze_feature_encoder_epochs", 0))
        if hasattr(student, "freeze_feature_extractor"):
            student.freeze_feature_extractor(freeze=(epoch <= freeze_epochs))
        student.train()
        total_loss = 0.0
        total_lc = 0.0
        total_lkl = 0.0
        total_lmse = 0.0
        total_lenh = 0.0
        total_lsnr = 0.0
        total_lnoise = 0.0
        total_lsnr_reg = 0.0
        total_count = 0
        gate_stats = {"sum": 0.0, "sq_sum": 0.0, "count": 0}

        noise_counter = Counter()
        snr_counter = Counter()
        train_start = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Student Epoch {epoch}", leave=False):
            clean = batch["clean_input_values"].to(device)
            noisy = batch["noisy_input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            noise_counter.update(batch["noise_types"])
            snr_counter.update(batch["snr_db"])
            snr_ids, noise_ids, snr_values = _prepare_aux_labels(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                t_out = teacher(clean, attention_mask=attention_mask)

            with autocast(enabled=use_amp):
                forward_kwargs = _build_student_forward_kwargs(
                    gate_source=train_gate_source,
                    hybrid_alpha=train_hybrid_alpha,
                    snr_values=snr_values,
                    snr_ids=snr_ids,
                    noise_ids=noise_ids,
                )
                s_out = student(
                    noisy,
                    attention_mask=attention_mask,
                    **forward_kwargs,
                )

                snr_logits_for_loss = s_out.get("snr_logits_pred")
                if snr_logits_for_loss is None:
                    snr_logits_for_loss = s_out.get("snr_logits")
                noise_logits_for_loss = s_out.get("noise_logits_pred")
                if noise_logits_for_loss is None:
                    noise_logits_for_loss = s_out.get("noise_logits")

                loss_dict = compute_adaptive_mlkd_loss(
                    logits_s=s_out["logits"],
                    logits_t=t_out["logits"],
                    student_states=s_out["hidden_states"],
                    teacher_states=t_out["hidden_states"],
                    labels=labels,
                    mask=s_out["feature_mask"],
                    temperature=epoch_loss_cfg["temperature"],
                    lambda_ce=epoch_loss_cfg["lambda_ce"],
                    lambda_kl=epoch_loss_cfg["lambda_kl"],
                    lambda_mse=epoch_loss_cfg["lambda_mse"],
                    lambda_enh=epoch_loss_cfg["lambda_enh"],
                    lambda_snr=epoch_loss_cfg["lambda_snr"],
                    lambda_noise=epoch_loss_cfg["lambda_noise"],
                    lambda_snr_reg=epoch_loss_cfg.get("lambda_snr_reg", 0.0),
                    distill_layers=epoch_loss_cfg["distill_layers"],
                    enhanced_wav=s_out.get("enhanced_wav"),
                    clean_wav=clean,
                    w_wav=epoch_loss_cfg["w_wav"],
                    w_stft=epoch_loss_cfg["w_stft"],
                    snr_logits=snr_logits_for_loss,
                    snr_labels=snr_ids if model_cfg.get("use_aux_snr", False) else None,
                    noise_logits=noise_logits_for_loss,
                    noise_labels=noise_ids if model_cfg.get("use_aux_noise", False) else None,
                    snr_value_pred=s_out.get("snr_value_pred"),
                    snr_values=snr_values if model_cfg.get("use_snr_regression", False) else None,
                    snr_reg_loss_type=epoch_loss_cfg.get("snr_reg_loss_type", "smooth_l1"),
                )

            scaler.scale(loss_dict["total"]).backward()
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss_dict["total"].item() * labels.size(0)
            total_lc += loss_dict["loss_ce"].item() * labels.size(0)
            total_lkl += loss_dict["loss_kl"].item() * labels.size(0)
            total_lmse += loss_dict["loss_mse"].item() * labels.size(0)
            total_lenh += loss_dict["loss_enh"].item() * labels.size(0)
            total_lsnr += loss_dict["loss_snr"].item() * labels.size(0)
            total_lnoise += loss_dict["loss_noise"].item() * labels.size(0)
            total_lsnr_reg += loss_dict["loss_snr_reg"].item() * labels.size(0)
            total_count += labels.size(0)
            _update_gate_stats(gate_stats, s_out.get("gate"))

        train_time_s = time.perf_counter() - train_start
        gate_mean, gate_std = _finalize_gate_stats(gate_stats)
        train_metrics = {
            "total": total_loss / max(1, total_count),
            "LC": total_lc / max(1, total_count),
            "LKL": total_lkl / max(1, total_count),
            "LMSE": total_lmse / max(1, total_count),
            "loss_ce": total_lc / max(1, total_count),
            "loss_kl": total_lkl / max(1, total_count),
            "loss_mse": total_lmse / max(1, total_count),
            "loss_enh": total_lenh / max(1, total_count),
            "loss_snr": total_lsnr / max(1, total_count),
            "loss_noise": total_lnoise / max(1, total_count),
            "loss_snr_reg": total_lsnr_reg / max(1, total_count),
            "gate_mean": gate_mean,
            "gate_std": gate_std,
        }
        lr_now = optimizer.param_groups[0]["lr"]
        noise_summary = ",".join(f"{k}:{v}" for k, v in sorted(noise_counter.items()))
        snr_summary = ",".join(
            f"{k}:{v}" for k, v in sorted(snr_counter.items(), key=lambda x: x[0])
        )

        proxy_metrics = None
        proxy_val_time_s = 0.0
        proxy_is_best = False
        if val_cfg["proxy_eval_enabled"]:
            proxy_start = time.perf_counter()
            proxy_metrics = _evaluate_student_with_teacher_conditions(
                student,
                teacher,
                proxy_conditions,
                device,
                epoch_loss_cfg,
                model_cfg,
                eval_condition_mode=eval_condition_mode,
            )
            proxy_val_time_s = time.perf_counter() - proxy_start
            proxy_is_best = proxy_metrics["mean_ua"] > best_proxy_mean_ua

        if proxy_metrics is not None:
            threshold = (
                -1.0
                if len(proxy_topk) < max(1, val_cfg["top_k_checkpoints"])
                else float(proxy_topk[-1]["proxy_mean_ua"])
            )
            if proxy_metrics["mean_ua"] > threshold:
                candidate_path = os.path.join(output_dir, f"proxy_candidate_epoch_{epoch}.pt")
                _save_student_checkpoint(
                    candidate_path,
                    student,
                    optimizer,
                    scaler,
                    epoch,
                    extra={
                        "proxy_mean_ua": proxy_metrics["mean_ua"],
                        "proxy_condition_uas": proxy_metrics["condition_uas"],
                        "proxy_manifest_path": val_cfg["proxy_manifest_path"],
                        "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                        "eval_condition_mode": eval_condition_mode,
                        "eval_gate_source": eval_gate_source,
                    },
                )
                candidate = {
                    "epoch": int(epoch),
                    "proxy_mean_ua": float(proxy_metrics["mean_ua"]),
                    "checkpoint_path": candidate_path,
                    "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                proxy_topk, entered_topk, removed = _update_proxy_topk(
                    proxy_topk, candidate, val_cfg["top_k_checkpoints"]
                )
                if removed is not None:
                    removed_path = str(removed.get("checkpoint_path", ""))
                    if removed_path and os.path.isfile(removed_path):
                        os.remove(removed_path)
                if not entered_topk and os.path.isfile(candidate_path):
                    os.remove(candidate_path)
            proxy_topk = _sort_proxy_topk(proxy_topk)[: max(1, val_cfg["top_k_checkpoints"])]
            _save_json(topk_path, {"items": proxy_topk})
            best_proxy_mean_ua = max(
                [float(x["proxy_mean_ua"]) for x in proxy_topk], default=best_proxy_mean_ua
            )
            if proxy_is_best and save_best_proxy:
                _save_student_checkpoint(
                    best_proxy_ckpt_path,
                    student,
                    optimizer,
                    scaler,
                    epoch,
                    scheduler=plateau_scheduler,
                    extra={
                        "proxy_mean_ua": float(proxy_metrics["mean_ua"]),
                        "best_metric_name": "proxy_mean_UA",
                        "best_metric_value": float(proxy_metrics["mean_ua"]),
                        "proxy_condition_uas": proxy_metrics["condition_uas"],
                        "best_proxy_mean_ua": float(proxy_metrics["mean_ua"]),
                        "best_full_mean_ua": float(best_full_mean_ua),
                        "best_monitor_value": float(best_monitor_value),
                        "early_stop_counter": int(early_stop_counter),
                        "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                        "eval_condition_mode": eval_condition_mode,
                        "train_gate_source": train_gate_source,
                        "eval_gate_source": eval_gate_source,
                        "hybrid_alpha": train_hybrid_alpha,
                    },
                )

        should_run_full = False
        if val_cfg["full_eval_every"] > 0 and (epoch % val_cfg["full_eval_every"] == 0):
            should_run_full = True
        if (
            val_cfg["trigger_full_eval_on_proxy_best"]
            and proxy_metrics is not None
            and proxy_is_best
        ):
            should_run_full = True

        full_metrics = None
        full_val_time_s = 0.0
        full_is_best = False
        if should_run_full:
            full_start = time.perf_counter()
            full_metrics = _evaluate_student_with_teacher_conditions(
                student,
                teacher,
                full_conditions,
                device,
                epoch_loss_cfg,
                model_cfg,
                eval_condition_mode=eval_condition_mode,
            )
            full_val_time_s = time.perf_counter() - full_start
            full_is_best = full_metrics["mean_ua"] > best_full_mean_ua
            full_eval_history.append(
                {
                    "epoch": int(epoch),
                    "full_mean_ua": float(full_metrics["mean_ua"]),
                    "condition_uas": full_metrics["condition_uas"],
                }
            )
            _save_json(full_history_path, {"items": full_eval_history})
            if full_is_best:
                best_full_mean_ua = full_metrics["mean_ua"]
                best_path = best_full_ckpt_path
                if save_best_full:
                    _save_student_checkpoint(
                        best_full_ckpt_path,
                        student,
                        optimizer,
                        scaler,
                        epoch,
                        scheduler=plateau_scheduler,
                        extra={
                            "full_mean_ua": float(full_metrics["mean_ua"]),
                            "best_metric_name": "full_mean_UA",
                            "best_metric_value": float(full_metrics["mean_ua"]),
                            "full_condition_uas": full_metrics["condition_uas"],
                            "full_manifest_path": val_cfg["full_manifest_path"],
                            "best_proxy_mean_ua": float(best_proxy_mean_ua),
                            "best_full_mean_ua": float(full_metrics["mean_ua"]),
                            "best_monitor_value": float(best_monitor_value),
                            "early_stop_counter": int(early_stop_counter),
                            "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                            "eval_condition_mode": eval_condition_mode,
                            "train_gate_source": train_gate_source,
                            "eval_gate_source": eval_gate_source,
                            "hybrid_alpha": train_hybrid_alpha,
                        },
                    )
                    # Keep backward compatibility with existing eval scripts.
                    _save_student_checkpoint(
                        os.path.join(output_dir, "best_student.pt"),
                        student,
                        optimizer,
                        scaler,
                        epoch,
                        scheduler=plateau_scheduler,
                        extra={
                            "full_mean_ua": float(full_metrics["mean_ua"]),
                            "best_metric_name": "full_mean_UA",
                            "best_metric_value": float(full_metrics["mean_ua"]),
                            "full_condition_uas": full_metrics["condition_uas"],
                            "full_manifest_path": val_cfg["full_manifest_path"],
                            "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                            "eval_condition_mode": eval_condition_mode,
                            "eval_gate_source": eval_gate_source,
                        },
                    )

        monitor_value, monitor_source = _resolve_monitor_value(
            monitor_name,
            full_metrics=full_metrics,
            proxy_metrics=proxy_metrics,
        )
        if monitor_value is not None:
            if monitor_value > best_monitor_value:
                best_monitor_value = float(monitor_value)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

        if plateau_scheduler is not None and monitor_value is not None:
            plateau_scheduler.step(float(monitor_value))

        _save_student_checkpoint(
            last_ckpt_path,
            student,
            optimizer,
            scaler,
            epoch,
            scheduler=plateau_scheduler,
            extra={
                "proxy_mean_ua": float(proxy_metrics["mean_ua"]) if proxy_metrics is not None else None,
                "full_mean_ua": float(full_metrics["mean_ua"]) if full_metrics is not None else None,
                "best_proxy_mean_ua": float(best_proxy_mean_ua),
                "best_full_mean_ua": float(best_full_mean_ua),
                "best_monitor_value": float(best_monitor_value),
                "monitor_name": monitor_name,
                "monitor_source": monitor_source,
                "monitor_value": float(monitor_value) if monitor_value is not None else None,
                "early_stop_counter": int(early_stop_counter),
                "loss_stage": epoch_loss_cfg.get("loss_stage", "fixed"),
                "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                "eval_condition_mode": eval_condition_mode,
                "train_gate_source": train_gate_source,
                "eval_gate_source": eval_gate_source,
                "hybrid_alpha": train_hybrid_alpha,
            },
        )

        epoch_time_s = time.perf_counter() - epoch_start
        proxy_mean_txt = f"{proxy_metrics['mean_ua']:.4f}" if proxy_metrics is not None else "NA"
        full_mean_txt = f"{full_metrics['mean_ua']:.4f}" if full_metrics is not None else "NA"
        monitor_txt = f"{monitor_value:.4f}" if monitor_value is not None else "NA"
        topk_summary = _format_topk_summary(proxy_topk)
        print(
            f"Epoch {epoch}/{epochs} | lr={lr_now:.2e} | "
            f"train LC={train_metrics['LC']:.4f} LKL={train_metrics['LKL']:.4f} "
            f"LMSE={train_metrics['LMSE']:.4f} LENH={train_metrics['loss_enh']:.4f} "
            f"LSNR={train_metrics['loss_snr']:.4f} LNOISE={train_metrics['loss_noise']:.4f} "
            f"LSNRREG={train_metrics['loss_snr_reg']:.4f} "
            f"gate_mean={train_metrics['gate_mean']:.4f} gate_std={train_metrics['gate_std']:.4f} "
            f"total={train_metrics['total']:.4f} | "
            f"proxy_mean_UA={proxy_mean_txt} full_mean_UA={full_mean_txt} "
            f"is_best_proxy={int(proxy_is_best)} is_best_full={int(full_is_best)} "
            f"full_ran={int(should_run_full)} | "
            f"train_gate_source={train_gate_source} eval_gate_source={eval_gate_source} "
            f"eval_condition_mode={eval_condition_mode} "
            f"monitor({monitor_source})={monitor_txt} best_monitor={best_monitor_value:.4f} "
            f"early_stop_counter={early_stop_counter} | "
            f"time train={train_time_s:.1f}s proxy={proxy_val_time_s:.1f}s "
            f"full={full_val_time_s:.1f}s epoch={epoch_time_s:.1f}s"
        )
        print(
            "Loss weights | "
            f"stage={epoch_loss_cfg.get('loss_stage', 'fixed')} "
            f"ce={epoch_loss_cfg['lambda_ce']:.3f} "
            f"kl={epoch_loss_cfg['lambda_kl']:.3f} "
            f"mse={epoch_loss_cfg['lambda_mse']:.3f} "
            f"enh={epoch_loss_cfg['lambda_enh']:.3f} "
            f"snr={epoch_loss_cfg['lambda_snr']:.3f} "
            f"noise={epoch_loss_cfg['lambda_noise']:.3f} "
            f"snr_reg={epoch_loss_cfg.get('lambda_snr_reg', 0.0):.3f}"
        )
        print(
            "Early stop state | "
            f"current_full_mean_UA={full_mean_txt} "
            f"best_full_mean_UA={best_full_mean_ua:.4f} "
            f"early_stop_counter={early_stop_counter}"
        )
        print(f"Proxy Top-K | {topk_summary if topk_summary else 'empty'}")
        if full_metrics is not None:
            _print_condition_ua_summary(full_metrics["condition_metrics"])
            print(f"Full Val UA | mean_ua={full_metrics['mean_ua']:.4f} full_best={int(full_is_best)}")
        if noise_counter:
            print(f"Noise stats | types: {noise_summary} | snr_db: {snr_summary}")

        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _append_log(
            log_path,
            log_header,
            {
                "timestamp": now,
                "epoch": epoch,
                "split": "train",
                "eval_stage": "train",
                "lr": lr_now,
                "train_time_s": train_time_s,
                "proxy_val_time_s": proxy_val_time_s,
                "full_val_time_s": full_val_time_s,
                "epoch_time_s": epoch_time_s,
                "LC": train_metrics["LC"],
                "LKL": train_metrics["LKL"],
                "LMSE": train_metrics["LMSE"],
                "loss_ce": train_metrics["loss_ce"],
                "loss_kl": train_metrics["loss_kl"],
                "loss_mse": train_metrics["loss_mse"],
                "loss_enh": train_metrics["loss_enh"],
                "loss_snr": train_metrics["loss_snr"],
                "loss_noise": train_metrics["loss_noise"],
                "loss_snr_reg": train_metrics["loss_snr_reg"],
                "gate_mean": train_metrics["gate_mean"],
                "gate_std": train_metrics["gate_std"],
                "total": train_metrics["total"],
                "proxy_mean_UA": proxy_metrics["mean_ua"] if proxy_metrics is not None else "",
                "full_mean_UA": full_metrics["mean_ua"] if full_metrics is not None else "",
                "best_proxy_mean_UA": best_proxy_mean_ua,
                "best_full_mean_UA": best_full_mean_ua,
                "is_proxy_best": int(proxy_is_best),
                "is_full_best": int(full_is_best),
                "monitor_metric": monitor_source,
                "monitor_value": monitor_value if monitor_value is not None else "",
                "best_monitor_value": best_monitor_value,
                "early_stop_counter": early_stop_counter,
                "current_full_mean_UA": full_metrics["mean_ua"] if full_metrics is not None else "",
                "loss_stage": epoch_loss_cfg.get("loss_stage", "fixed"),
                "lambda_ce": epoch_loss_cfg["lambda_ce"],
                "lambda_kl": epoch_loss_cfg["lambda_kl"],
                "lambda_mse": epoch_loss_cfg["lambda_mse"],
                "lambda_enh": epoch_loss_cfg["lambda_enh"],
                "lambda_snr": epoch_loss_cfg["lambda_snr"],
                "lambda_noise": epoch_loss_cfg["lambda_noise"],
                "lambda_snr_reg": epoch_loss_cfg.get("lambda_snr_reg", 0.0),
                "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                "eval_condition_mode": eval_condition_mode,
                "train_gate_source": train_gate_source,
                "eval_gate_source": eval_gate_source,
                "hybrid_alpha": train_hybrid_alpha if train_hybrid_alpha is not None else "",
                "grad_clip": grad_clip if grad_clip > 0 else 0.0,
                "proxy_topk": topk_summary,
                "noise_types": noise_summary if noise_counter else "",
                "snr_db": snr_summary if noise_counter else "",
                **{header: "" for header in condition_headers},
            },
        )

        if proxy_metrics is not None:
            _append_log(
                log_path,
                log_header,
                {
                    "timestamp": now,
                    "epoch": epoch,
                    "split": "proxy_val",
                    "eval_stage": "proxy",
                    "lr": lr_now,
                    "train_time_s": train_time_s,
                    "proxy_val_time_s": proxy_val_time_s,
                    "full_val_time_s": "",
                    "epoch_time_s": epoch_time_s,
                    "LC": proxy_metrics["LC"],
                    "LKL": proxy_metrics["LKL"],
                    "LMSE": proxy_metrics["LMSE"],
                    "loss_ce": proxy_metrics["loss_ce"],
                    "loss_kl": proxy_metrics["loss_kl"],
                    "loss_mse": proxy_metrics["loss_mse"],
                    "loss_enh": proxy_metrics["loss_enh"],
                    "loss_snr": proxy_metrics["loss_snr"],
                    "loss_noise": proxy_metrics["loss_noise"],
                    "loss_snr_reg": proxy_metrics["loss_snr_reg"],
                    "gate_mean": proxy_metrics["gate_mean"],
                    "gate_std": proxy_metrics["gate_std"],
                    "total": proxy_metrics["total"],
                    "proxy_mean_UA": proxy_metrics["mean_ua"],
                    "full_mean_UA": "",
                    "best_proxy_mean_UA": best_proxy_mean_ua,
                    "best_full_mean_UA": best_full_mean_ua,
                    "is_proxy_best": int(proxy_is_best),
                    "is_full_best": "",
                    "monitor_metric": monitor_source,
                    "monitor_value": monitor_value if monitor_value is not None else "",
                    "best_monitor_value": best_monitor_value,
                    "early_stop_counter": early_stop_counter,
                    "current_full_mean_UA": "",
                    "loss_stage": epoch_loss_cfg.get("loss_stage", "fixed"),
                    "lambda_ce": epoch_loss_cfg["lambda_ce"],
                    "lambda_kl": epoch_loss_cfg["lambda_kl"],
                    "lambda_mse": epoch_loss_cfg["lambda_mse"],
                    "lambda_enh": epoch_loss_cfg["lambda_enh"],
                    "lambda_snr": epoch_loss_cfg["lambda_snr"],
                    "lambda_noise": epoch_loss_cfg["lambda_noise"],
                    "lambda_snr_reg": epoch_loss_cfg.get("lambda_snr_reg", 0.0),
                    "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                    "eval_condition_mode": eval_condition_mode,
                    "train_gate_source": train_gate_source,
                    "eval_gate_source": proxy_metrics.get("eval_gate_source", eval_gate_source),
                    "hybrid_alpha": train_hybrid_alpha if train_hybrid_alpha is not None else "",
                    "grad_clip": grad_clip if grad_clip > 0 else 0.0,
                    "proxy_topk": topk_summary,
                    "noise_types": "",
                    "snr_db": "",
                    **proxy_metrics["condition_uas"],
                },
            )

        if full_metrics is not None:
            _append_log(
                log_path,
                log_header,
                {
                    "timestamp": now,
                    "epoch": epoch,
                    "split": "full_val",
                    "eval_stage": "full",
                    "lr": lr_now,
                    "train_time_s": train_time_s,
                    "proxy_val_time_s": proxy_val_time_s,
                    "full_val_time_s": full_val_time_s,
                    "epoch_time_s": epoch_time_s,
                    "LC": full_metrics["LC"],
                    "LKL": full_metrics["LKL"],
                    "LMSE": full_metrics["LMSE"],
                    "loss_ce": full_metrics["loss_ce"],
                    "loss_kl": full_metrics["loss_kl"],
                    "loss_mse": full_metrics["loss_mse"],
                    "loss_enh": full_metrics["loss_enh"],
                    "loss_snr": full_metrics["loss_snr"],
                    "loss_noise": full_metrics["loss_noise"],
                    "loss_snr_reg": full_metrics["loss_snr_reg"],
                    "gate_mean": full_metrics["gate_mean"],
                    "gate_std": full_metrics["gate_std"],
                    "total": full_metrics["total"],
                    "proxy_mean_UA": proxy_metrics["mean_ua"] if proxy_metrics is not None else "",
                    "full_mean_UA": full_metrics["mean_ua"],
                    "best_proxy_mean_UA": best_proxy_mean_ua,
                    "best_full_mean_UA": best_full_mean_ua,
                    "is_proxy_best": int(proxy_is_best),
                    "is_full_best": int(full_is_best),
                    "monitor_metric": monitor_source,
                    "monitor_value": monitor_value if monitor_value is not None else "",
                    "best_monitor_value": best_monitor_value,
                    "early_stop_counter": early_stop_counter,
                    "current_full_mean_UA": full_metrics["mean_ua"],
                    "loss_stage": epoch_loss_cfg.get("loss_stage", "fixed"),
                    "lambda_ce": epoch_loss_cfg["lambda_ce"],
                    "lambda_kl": epoch_loss_cfg["lambda_kl"],
                    "lambda_mse": epoch_loss_cfg["lambda_mse"],
                    "lambda_enh": epoch_loss_cfg["lambda_enh"],
                    "lambda_snr": epoch_loss_cfg["lambda_snr"],
                    "lambda_noise": epoch_loss_cfg["lambda_noise"],
                    "lambda_snr_reg": epoch_loss_cfg.get("lambda_snr_reg", 0.0),
                    "gate_condition_mode": model_cfg.get("gate_condition_mode", "none"),
                    "eval_condition_mode": eval_condition_mode,
                    "train_gate_source": train_gate_source,
                    "eval_gate_source": full_metrics.get("eval_gate_source", eval_gate_source),
                    "hybrid_alpha": train_hybrid_alpha if train_hybrid_alpha is not None else "",
                    "grad_clip": grad_clip if grad_clip > 0 else 0.0,
                    "proxy_topk": topk_summary,
                    "noise_types": "",
                    "snr_db": "",
                    **full_metrics["condition_uas"],
                },
            )

        if save_every > 0 and (epoch % save_every == 0):
            periodic_ckpt = os.path.join(output_dir, f"student_epoch_{epoch}.pt")
            _save_student_checkpoint(
                periodic_ckpt,
                student,
                optimizer,
                scaler,
                epoch,
                scheduler=plateau_scheduler,
            )

        if early_stopping_patience > 0 and monitor_value is not None and early_stop_counter >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch} "
                f"(monitor={monitor_name}, no improvement for {early_stop_counter} epochs)."
            )
            break

    if not best_path and os.path.isfile(best_full_ckpt_path):
        best_path = best_full_ckpt_path
    if not best_path and os.path.isfile(best_proxy_ckpt_path):
        best_path = best_proxy_ckpt_path
    if not best_path and os.path.isfile(last_ckpt_path):
        best_path = last_ckpt_path

    if val_cfg["final_rerank_with_full_eval"] and proxy_topk:
        best_path = rerank_student_topk_checkpoints(
            metadata_csv=metadata_csv,
            teacher_ckpt=teacher_ckpt,
            output_dir=output_dir,
            fold_speaker=fold_speaker,
            config=config,
            batch_size=batch_size,
        )
    elif best_path == "" and proxy_topk:
        best_path = str(proxy_topk[0]["checkpoint_path"])

    if not best_path and proxy_topk:
        best_path = str(proxy_topk[0]["checkpoint_path"])
    return best_path


def rerank_student_topk_checkpoints(
    metadata_csv: str,
    teacher_ckpt: str,
    output_dir: str,
    fold_speaker: Optional[str],
    config: Dict,
    batch_size: Optional[int] = None,
) -> str:
    train_cfg = _resolve_runtime_train_cfg(config)
    split_cfg = _resolve_data_split_cfg(config, train_cfg)
    audio_cfg = config["audio"]
    noise_cfg = config["noise"]
    model_cfg = _resolve_model_cfg(config, noise_cfg)
    loss_cfg = _resolve_loss_cfg(config)
    effective_loss_cfg = dict(loss_cfg)
    if not model_cfg.get("use_enhancement", False):
        effective_loss_cfg["lambda_enh"] = 0.0
    if not model_cfg.get("use_aux_snr", False):
        effective_loss_cfg["lambda_snr"] = 0.0
    if not model_cfg.get("use_aux_noise", False):
        effective_loss_cfg["lambda_noise"] = 0.0
    if not model_cfg.get("use_snr_regression", False):
        effective_loss_cfg["lambda_snr_reg"] = 0.0
    eval_condition_mode = str(train_cfg.get("eval_condition_mode", "deploy")).strip().lower()
    if eval_condition_mode not in {"deploy", "oracle"}:
        eval_condition_mode = "deploy"
    val_cfg = _resolve_student_val_cfg(config, output_dir)

    topk_path = os.path.join(output_dir, "proxy_topk.json")
    if not os.path.isfile(topk_path):
        raise FileNotFoundError(f"proxy_topk.json not found: {topk_path}")
    with open(topk_path, "r", encoding="utf-8") as f:
        topk_payload = json.load(f)
    topk_items = list(topk_payload.get("items", [])) if isinstance(topk_payload, dict) else []
    topk_items = [x for x in topk_items if os.path.isfile(str(x.get("checkpoint_path", "")))]
    if not topk_items:
        raise RuntimeError("No valid proxy top-k checkpoints to rerank.")
    topk_items = _sort_proxy_topk(topk_items)[: max(1, val_cfg["top_k_checkpoints"])]

    eval_batch_size = int(batch_size or train_cfg["batch_size"])
    df = load_metadata_df(metadata_csv)
    _, val_df, _ = split_train_val_test(
        df,
        split_mode=split_cfg["split_mode"],
        seed=train_cfg["seed"],
        fold_speaker=fold_speaker,
        val_ratio=split_cfg["val_ratio"],
        test_ratio=split_cfg["test_ratio"],
    )
    clean_val_ds = IemocapDataset(
        val_df,
        sample_rate=audio_cfg["sample_rate"],
        max_seconds=audio_cfg["max_seconds"],
        is_train=False,
        eval_clip=audio_cfg.get("eval_clip", True),
    )
    clean_eval_records = _collect_clean_eval_records(clean_val_ds)
    noise_adder = NoiseAdder(
        noise_root=config["paths"]["noise_root"],
        noise_types=noise_cfg["types"],
        snr_levels=noise_cfg["snrs"],
        sample_rate=audio_cfg["sample_rate"],
    )
    full_manifest = _build_or_load_val_manifest(
        manifest_path=val_cfg["full_manifest_path"],
        manifest_type="full",
        clean_records=clean_eval_records,
        noise_adder=noise_adder,
        noise_cfg=noise_cfg,
        seed=val_cfg["val_manifest_seed"],
        proxy_ratio=val_cfg["proxy_ratio"],
        min_samples_per_condition=val_cfg["min_samples_per_condition"],
    )
    full_conditions = _build_manifest_val_loaders(
        clean_val_ds,
        noise_adder,
        full_manifest,
        batch_size=eval_batch_size,
        num_workers=train_cfg["num_workers"],
        seed=val_cfg["val_manifest_seed"],
    )

    device = get_device()
    teacher = Wav2Vec2SERTeacher(
        num_classes=4,
        backbone_name=model_cfg.get("teacher_backbone_name"),
    ).to(device)
    teacher_state = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_state["model_state"])
    teacher.eval()
    if bool(train_cfg.get("freeze_teacher", True)):
        for p in teacher.parameters():
            p.requires_grad = False

    student_cfg_overrides: Dict[str, Any] = {}
    student_backbone_name = model_cfg.get("student_backbone_name")
    if student_backbone_name and str(student_backbone_name).lower() not in {"teacher", "none"}:
        local_only = os.path.isdir(str(student_backbone_name))
        student_backbone_cfg = Wav2Vec2Config.from_pretrained(
            str(student_backbone_name),
            local_files_only=local_only,
        )
        student_cfg_overrides.update(student_backbone_cfg.to_dict())
    if model_cfg.get("student_num_hidden_layers") is not None:
        student_cfg_overrides["num_hidden_layers"] = int(model_cfg["student_num_hidden_layers"])
    student = Wav2Vec2SERStudent.from_teacher(
        teacher,
        num_classes=4,
        student_cfg_overrides=student_cfg_overrides,
        model_args=model_cfg,
    ).to(device)

    rerank_items: List[Dict[str, Any]] = []
    for rank_idx, item in enumerate(topk_items, start=1):
        ckpt_path = str(item["checkpoint_path"])
        ckpt = torch.load(ckpt_path, map_location=device)
        load_result = student.load_state_dict(ckpt["model_state"], strict=False)
        missing = list(getattr(load_result, "missing_keys", []))
        unexpected = list(getattr(load_result, "unexpected_keys", []))
        if missing or unexpected:
            print(
                f"Rerank load_state_dict(non-strict) | missing={len(missing)} unexpected={len(unexpected)}"
            )
        ckpt_model_meta = ckpt.get("model_meta")
        if isinstance(ckpt_model_meta, dict) and hasattr(student, "model_args"):
            student.model_args.update(ckpt_model_meta)
        metrics = _evaluate_student_with_teacher_conditions(
            student,
            teacher,
            full_conditions,
            device,
            effective_loss_cfg,
            model_cfg,
            eval_condition_mode=eval_condition_mode,
        )
        rerank_items.append(
            {
                "rank_input": rank_idx,
                "epoch": int(item["epoch"]),
                "proxy_mean_ua": float(item["proxy_mean_ua"]),
                "checkpoint_path": ckpt_path,
                "full_mean_ua": float(metrics["mean_ua"]),
                "full_condition_uas": metrics["condition_uas"],
            }
        )

    rerank_items = sorted(rerank_items, key=lambda x: (-x["full_mean_ua"], int(x["epoch"])))
    best_item = rerank_items[0]
    best_ckpt = torch.load(best_item["checkpoint_path"], map_location="cpu")
    best_ckpt.update(
        {
            "best_metric_name": "full_mean_ua_25_conditions",
            "best_metric_value": float(best_item["full_mean_ua"]),
            "full_condition_uas": best_item["full_condition_uas"],
            "selection_source": "proxy_topk_full_rerank",
            "source_checkpoint_path": best_item["checkpoint_path"],
            "proxy_mean_ua": float(best_item["proxy_mean_ua"]),
        }
    )

    best_path = os.path.join(output_dir, "best_student.pt")
    torch.save(best_ckpt, best_path)
    _save_json(
        os.path.join(output_dir, "topk_full_rerank.json"),
        {
            "best_checkpoint": best_path,
            "full_manifest_path": val_cfg["full_manifest_path"],
            "items": rerank_items,
        },
    )
    print(
        f"Full rerank done | best epoch={best_item['epoch']} "
        f"proxy={best_item['proxy_mean_ua']:.4f} full={best_item['full_mean_ua']:.4f}"
    )
    return best_path
