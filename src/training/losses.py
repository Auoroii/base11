from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def kd_kl_loss(
    logits_s: torch.Tensor, logits_t: torch.Tensor, temperature: float
) -> torch.Tensor:
    t = float(max(1e-6, temperature))
    log_ps = F.log_softmax(logits_s / t, dim=-1)
    pt = F.softmax(logits_t / t, dim=-1)
    return F.kl_div(log_ps, pt, reduction="batchmean")


def _ensure_mask(
    mask: Optional[torch.Tensor], ref: torch.Tensor
) -> torch.Tensor:
    if mask is not None:
        return mask
    return torch.ones(ref.shape[0], ref.shape[1], device=ref.device, dtype=torch.long)


def masked_mse(
    student: torch.Tensor, teacher: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    mask = _ensure_mask(mask, student).float()
    mse = (student - teacher).pow(2).mean(dim=-1)
    denom = mask.sum().clamp_min(1.0)
    return (mse * mask).sum() / denom


def _layer_mapping(
    student_states: Tuple[torch.Tensor, ...],
    teacher_states: Tuple[torch.Tensor, ...],
    distill_layers: str = "original_even_mapping",
) -> List[Tuple[int, int]]:
    s_last = len(student_states) - 1
    t_last = len(teacher_states) - 1
    if s_last <= 0 or t_last <= 0:
        return []

    if distill_layers == "original_even_mapping" and s_last >= 6 and t_last >= 12:
        return list(zip([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]))

    n = min(s_last, t_last)
    s_indices = list(range(1, n + 1))
    if n == 1:
        t_indices = [t_last]
    else:
        t_indices = torch.linspace(1, t_last, steps=n).round().long().tolist()
    return list(zip(s_indices, t_indices))


def multi_level_mse(
    student_states: Tuple[torch.Tensor, ...],
    teacher_states: Tuple[torch.Tensor, ...],
    mask: Optional[torch.Tensor],
    distill_layers: str = "original_even_mapping",
) -> torch.Tensor:
    mapping = _layer_mapping(student_states, teacher_states, distill_layers=distill_layers)
    if not mapping:
        return torch.tensor(0.0, device=student_states[0].device)

    losses: List[torch.Tensor] = []
    for s_idx, t_idx in mapping:
        if s_idx >= len(student_states) or t_idx >= len(teacher_states):
            raise ValueError("Hidden states are missing required layers.")
        losses.append(masked_mse(student_states[s_idx], teacher_states[t_idx], mask))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=student_states[0].device)


def stft_reconstruction_loss(
    pred_wav: torch.Tensor,
    target_wav: torch.Tensor,
    fft_sizes: Sequence[int] = (256, 512, 1024),
    hop_sizes: Sequence[int] = (64, 128, 256),
    win_sizes: Sequence[int] = (256, 512, 1024),
) -> torch.Tensor:
    if pred_wav.shape != target_wav.shape:
        min_len = min(pred_wav.shape[-1], target_wav.shape[-1])
        pred_wav = pred_wav[..., :min_len]
        target_wav = target_wav[..., :min_len]

    time_len = int(pred_wav.shape[-1])
    losses: List[torch.Tensor] = []
    for n_fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes):
        cur_fft = int(min(int(n_fft), time_len))
        if cur_fft < 16:
            continue
        cur_win = int(min(int(win), cur_fft))
        cur_hop = int(min(int(hop), max(1, cur_win // 4)))
        window = torch.hann_window(cur_win, dtype=pred_wav.dtype, device=pred_wav.device)
        pred_spec = torch.stft(
            pred_wav,
            n_fft=cur_fft,
            hop_length=cur_hop,
            win_length=cur_win,
            window=window,
            return_complex=True,
            center=False,
        )
        target_spec = torch.stft(
            target_wav,
            n_fft=cur_fft,
            hop_length=cur_hop,
            win_length=cur_win,
            window=window,
            return_complex=True,
            center=False,
        )
        losses.append(torch.mean(torch.abs(pred_spec.abs() - target_spec.abs())))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred_wav.device)


def enhancement_reconstruction_loss(
    enhanced_wav: Optional[torch.Tensor],
    clean_wav: Optional[torch.Tensor],
    w_wav: float,
    w_stft: float,
) -> Dict[str, torch.Tensor]:
    if enhanced_wav is None or clean_wav is None:
        device = (
            enhanced_wav.device
            if enhanced_wav is not None
            else clean_wav.device
            if clean_wav is not None
            else torch.device("cpu")
        )
        zero = torch.tensor(
            0.0,
            device=device,
        )
        return {"loss_enh": zero, "loss_enh_wav": zero, "loss_enh_stft": zero}

    min_len = min(enhanced_wav.shape[-1], clean_wav.shape[-1])
    enhanced = enhanced_wav[..., :min_len]
    clean = clean_wav[..., :min_len]

    loss_wav = F.l1_loss(enhanced, clean)
    loss_stft = stft_reconstruction_loss(enhanced, clean)
    loss_enh = float(w_wav) * loss_wav + float(w_stft) * loss_stft
    return {"loss_enh": loss_enh, "loss_enh_wav": loss_wav, "loss_enh_stft": loss_stft}


def compute_adaptive_mlkd_loss(
    *,
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    student_states: Tuple[torch.Tensor, ...],
    teacher_states: Tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    mask: Optional[torch.Tensor],
    temperature: float,
    lambda_ce: float,
    lambda_kl: float,
    lambda_mse: float,
    lambda_enh: float,
    lambda_snr: float = 0.0,
    lambda_noise: float = 0.0,
    lambda_snr_reg: float = 0.0,
    distill_layers: str = "original_even_mapping",
    enhanced_wav: Optional[torch.Tensor] = None,
    clean_wav: Optional[torch.Tensor] = None,
    w_wav: float = 1.0,
    w_stft: float = 1.0,
    snr_logits: Optional[torch.Tensor] = None,
    snr_labels: Optional[torch.Tensor] = None,
    noise_logits: Optional[torch.Tensor] = None,
    noise_labels: Optional[torch.Tensor] = None,
    snr_value_pred: Optional[torch.Tensor] = None,
    snr_values: Optional[torch.Tensor] = None,
    snr_reg_loss_type: str = "smooth_l1",
) -> Dict[str, torch.Tensor]:
    loss_ce = F.cross_entropy(logits_s, labels)
    loss_kl = kd_kl_loss(logits_s, logits_t, temperature=temperature)
    loss_mse = multi_level_mse(
        student_states, teacher_states, mask, distill_layers=distill_layers
    )

    enh_losses = enhancement_reconstruction_loss(
        enhanced_wav=enhanced_wav,
        clean_wav=clean_wav,
        w_wav=w_wav,
        w_stft=w_stft,
    )
    loss_enh = enh_losses["loss_enh"]

    loss_snr = torch.tensor(0.0, device=logits_s.device)
    if snr_logits is not None and snr_labels is not None:
        loss_snr = F.cross_entropy(snr_logits, snr_labels)

    loss_noise = torch.tensor(0.0, device=logits_s.device)
    if noise_logits is not None and noise_labels is not None:
        loss_noise = F.cross_entropy(noise_logits, noise_labels)

    loss_snr_reg = torch.tensor(0.0, device=logits_s.device)
    if snr_value_pred is not None and snr_values is not None:
        pred = snr_value_pred.squeeze(-1).float()
        target = snr_values.float()
        if str(snr_reg_loss_type).strip().lower() == "mse":
            loss_snr_reg = F.mse_loss(pred, target)
        else:
            loss_snr_reg = F.smooth_l1_loss(pred, target)

    total = (
        float(lambda_ce) * loss_ce
        + float(lambda_kl) * loss_kl
        + float(lambda_mse) * loss_mse
        + float(lambda_enh) * loss_enh
        + float(lambda_snr) * loss_snr
        + float(lambda_noise) * loss_noise
        + float(lambda_snr_reg) * loss_snr_reg
    )

    return {
        "loss_ce": loss_ce,
        "loss_kl": loss_kl,
        "loss_mse": loss_mse,
        "loss_enh": loss_enh,
        "loss_enh_wav": enh_losses["loss_enh_wav"],
        "loss_enh_stft": enh_losses["loss_enh_stft"],
        "loss_snr": loss_snr,
        "loss_noise": loss_noise,
        "loss_snr_reg": loss_snr_reg,
        "total": total,
        # Backward-compatible names for existing training logs/callers.
        "LC": loss_ce,
        "LKL": loss_kl,
        "LMSE": loss_mse,
    }


def compute_mlkd_loss(
    logits_s: torch.Tensor,
    logits_t: torch.Tensor,
    student_states: Tuple[torch.Tensor, ...],
    teacher_states: Tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    mask: Optional[torch.Tensor],
    alpha: float,
    temperature: float,
) -> Dict[str, torch.Tensor]:
    # Compatibility wrapper used by baseline paths.
    return compute_adaptive_mlkd_loss(
        logits_s=logits_s,
        logits_t=logits_t,
        student_states=student_states,
        teacher_states=teacher_states,
        labels=labels,
        mask=mask,
        temperature=temperature,
        lambda_ce=(1.0 - float(alpha)),
        lambda_kl=float(alpha),
        lambda_mse=1.0,
        lambda_enh=0.0,
        lambda_snr=0.0,
        lambda_noise=0.0,
        lambda_snr_reg=0.0,
        distill_layers="original_even_mapping",
    )
