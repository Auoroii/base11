from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model


def _compute_feature_mask(
    backbone: Wav2Vec2Model, attention_mask: Optional[torch.Tensor], feature_len: int
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(
            (feature_len,), dtype=torch.long, device=next(backbone.parameters()).device
        ).unsqueeze(0)
    input_lengths = attention_mask.long().sum(dim=-1)
    feat_lengths = backbone._get_feat_extract_output_lengths(input_lengths)
    max_len = feature_len
    mask = torch.arange(max_len, device=attention_mask.device)[None, :] < feat_lengths[:, None]
    return mask


def _masked_mean(
    hidden_states: torch.Tensor, feature_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if feature_mask is None:
        return hidden_states.mean(dim=1)
    mask = feature_mask.float().unsqueeze(-1)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _resolve_wav2vec2_path() -> str:
    # Prefer explicit env var, then common project-local cache locations.
    env_path = os.environ.get("WAV2VEC2_BASE_PATH")
    if env_path:
        env_candidate = Path(env_path).expanduser()
        looks_like_local_path = (
            env_candidate.is_absolute()
            or env_path.startswith(".")
            or "/" in env_path
            or "\\" in env_path
        )
        if not looks_like_local_path or env_candidate.is_dir():
            return str(env_candidate) if env_candidate.is_dir() else env_path

    project_root = Path(__file__).resolve().parents[2]
    local_candidates = [
        project_root / "wav2vec2-base",
        Path.cwd() / "wav2vec2-base",
    ]
    for candidate in local_candidates:
        if candidate.is_dir():
            return str(candidate)

    return "facebook/wav2vec2-base"


def _disable_feature_input_requires_grad(backbone: Wav2Vec2Model) -> None:
    # HF Wav2Vec2 may force `hidden_states.requires_grad = True` in feature extractor.
    # That breaks when upstream input is a non-leaf tensor (e.g., enhancement output).
    feature_extractor = getattr(backbone, "feature_extractor", None)
    if feature_extractor is not None and hasattr(feature_extractor, "_requires_grad"):
        feature_extractor._requires_grad = False


class ResidualConvBlock1D(nn.Module):
    """Small residual block used by the enhancement front-end."""

    def __init__(self, channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.act(x + residual)


def _align_time_axis(src: torch.Tensor, target_len: int) -> torch.Tensor:
    if src.shape[-1] == target_len:
        return src
    if src.shape[-1] > target_len:
        return src[..., :target_len]
    pad = target_len - src.shape[-1]
    return F.pad(src, (0, pad))


class ResUNetEnhancer1D(nn.Module):
    """ResUNet-style 1D speech enhancement module for waveform denoising."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.in_proj = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=9, padding=4),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            ResidualConvBlock1D(c1),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            ResidualConvBlock1D(c2),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c3),
            nn.GELU(),
            ResidualConvBlock1D(c3),
        )
        self.bottleneck = nn.Sequential(
            ResidualConvBlock1D(c3),
            ResidualConvBlock1D(c3),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c2),
            nn.GELU(),
            ResidualConvBlock1D(c2),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            ResidualConvBlock1D(c1),
        )
        self.out_proj = nn.Conv1d(c1, 1, kernel_size=9, padding=4)

    def forward(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        # Keep module independent and easy to replace with future enhancement models.
        x = noisy_wav.unsqueeze(1)
        e1 = self.in_proj(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        b = self.bottleneck(e3)

        d2 = self.up2(b)
        d2 = _align_time_axis(d2, e2.shape[-1]) + e2
        d1 = self.up1(d2)
        d1 = _align_time_axis(d1, e1.shape[-1]) + e1

        residual = self.out_proj(d1)
        enhanced = torch.tanh(residual + x)
        enhanced = _align_time_axis(enhanced, noisy_wav.shape[-1])
        return enhanced.squeeze(1)


class IdentityEnhancer(nn.Module):
    def forward(self, noisy_wav: torch.Tensor) -> torch.Tensor:
        return noisy_wav


class ConditionEstimator(nn.Module):
    """Predict condition signals from dual-path pooled representations."""

    def __init__(
        self,
        hidden_size: int,
        condition_feat_dim: int = 64,
        num_snr_classes: int = 0,
        num_noise_classes: int = 0,
        predict_snr_value: bool = True,
    ) -> None:
        super().__init__()
        condition_feat_dim = int(max(1, condition_feat_dim))
        mid_dim = max(hidden_size, condition_feat_dim)
        self.backbone = nn.Sequential(
            nn.Linear(hidden_size * 2, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
        )
        self.cond_proj = nn.Linear(mid_dim, condition_feat_dim)
        self.snr_head = (
            nn.Linear(mid_dim, int(num_snr_classes))
            if int(num_snr_classes) > 0
            else None
        )
        self.noise_head = (
            nn.Linear(mid_dim, int(num_noise_classes))
            if int(num_noise_classes) > 0
            else None
        )
        self.snr_reg_head = nn.Linear(mid_dim, 1) if predict_snr_value else None

    def forward(
        self, pooled_raw: torch.Tensor, pooled_enh: torch.Tensor
    ) -> Dict[str, Optional[torch.Tensor]]:
        fused = torch.cat([pooled_raw, pooled_enh], dim=-1)
        hidden = self.backbone(fused)
        cond_feat = self.cond_proj(hidden)
        return {
            "cond_feat": cond_feat,
            "snr_logits_pred": self.snr_head(hidden) if self.snr_head is not None else None,
            "noise_logits_pred": self.noise_head(hidden) if self.noise_head is not None else None,
            "snr_value_pred": self.snr_reg_head(hidden) if self.snr_reg_head is not None else None,
        }


class AdaptiveFusionGate(nn.Module):
    """Predicts branch fusion weight in [0, 1] with selectable condition source."""

    def __init__(
        self,
        hidden_size: int,
        condition_feat_dim: int = 0,
        use_oracle_snr: bool = False,
        use_oracle_noise: bool = False,
        num_snr_classes: int = 0,
        num_noise_classes: int = 0,
        snr_embed_dim: int = 16,
        noise_embed_dim: int = 16,
        snr_temperature: float = 10.0,
        snr_bias_strength: float = 0.75,
    ) -> None:
        super().__init__()
        self.condition_feat_dim = int(max(0, condition_feat_dim))
        self.snr_temperature = float(max(1.0, snr_temperature))
        self.snr_bias_strength = float(max(0.0, snr_bias_strength))

        input_dim = hidden_size * 2
        self.snr_embedding = None
        self.noise_embedding = None
        if int(num_snr_classes) > 0:
            self.snr_embedding = nn.Embedding(int(num_snr_classes), snr_embed_dim)
            input_dim += snr_embed_dim
        if int(num_noise_classes) > 0:
            self.noise_embedding = nn.Embedding(int(num_noise_classes), noise_embed_dim)
            input_dim += noise_embed_dim
        self.use_oracle_snr = bool(use_oracle_snr and self.snr_embedding is not None)
        self.use_oracle_noise = bool(use_oracle_noise and self.noise_embedding is not None)
        if self.condition_feat_dim > 0:
            input_dim += self.condition_feat_dim

        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _zeros_like_ref(
        batch_size: int,
        dim: int,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, dim, dtype=ref.dtype, device=ref.device)

    def _embed_soft_labels(
        self, logits: Optional[torch.Tensor], embedding: Optional[nn.Embedding], ref: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if logits is None or embedding is None:
            return None
        probs = torch.softmax(logits.float(), dim=-1)
        soft_embed = probs @ embedding.weight
        return soft_embed.to(dtype=ref.dtype, device=ref.device)

    def _apply_snr_bias(
        self, gate: torch.Tensor, snr_value: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if snr_value is None or self.snr_bias_strength <= 0.0:
            return gate
        snr_norm = torch.clamp(snr_value.float() / self.snr_temperature, -4.0, 4.0)
        logits = torch.logit(gate.clamp(min=1e-4, max=1.0 - 1e-4))
        logits = logits - self.snr_bias_strength * snr_norm.unsqueeze(-1)
        return torch.sigmoid(logits)

    def _forward_single(
        self,
        *,
        pooled_raw: torch.Tensor,
        pooled_enh: torch.Tensor,
        condition_source: str,
        snr_values: Optional[torch.Tensor],
        snr_ids: Optional[torch.Tensor],
        noise_ids: Optional[torch.Tensor],
        cond_feat: Optional[torch.Tensor],
        snr_logits_pred: Optional[torch.Tensor],
        noise_logits_pred: Optional[torch.Tensor],
        snr_value_pred: Optional[torch.Tensor],
    ) -> torch.Tensor:
        source = str(condition_source).strip().lower()
        batch_size = pooled_raw.shape[0]
        feats = [pooled_raw, pooled_enh]

        if self.condition_feat_dim > 0:
            if source == "predicted" and cond_feat is not None:
                feats.append(cond_feat)
            else:
                feats.append(self._zeros_like_ref(batch_size, self.condition_feat_dim, pooled_raw))

        if self.snr_embedding is not None:
            snr_feat = self._zeros_like_ref(
                batch_size, self.snr_embedding.embedding_dim, pooled_raw
            )
            if source == "oracle" and self.use_oracle_snr and snr_ids is not None:
                snr_feat = self.snr_embedding(snr_ids)
            elif source == "predicted":
                soft_embed = self._embed_soft_labels(
                    snr_logits_pred, self.snr_embedding, pooled_raw
                )
                if soft_embed is not None:
                    snr_feat = soft_embed
            feats.append(snr_feat)

        if self.noise_embedding is not None:
            noise_feat = self._zeros_like_ref(
                batch_size, self.noise_embedding.embedding_dim, pooled_raw
            )
            if source == "oracle" and self.use_oracle_noise and noise_ids is not None:
                noise_feat = self.noise_embedding(noise_ids)
            elif source == "predicted":
                soft_embed = self._embed_soft_labels(
                    noise_logits_pred, self.noise_embedding, pooled_raw
                )
                if soft_embed is not None:
                    noise_feat = soft_embed
            feats.append(noise_feat)

        gate = torch.sigmoid(self.predictor(torch.cat(feats, dim=-1)))
        prior_snr = snr_values if source == "oracle" else None
        if source == "predicted" and snr_value_pred is not None:
            prior_snr = snr_value_pred.squeeze(-1)
        return self._apply_snr_bias(gate, prior_snr)

    def forward(
        self,
        pooled_raw: torch.Tensor,
        pooled_enh: torch.Tensor,
        condition_source: str = "none",
        hybrid_alpha: Optional[float] = None,
        snr_values: Optional[torch.Tensor] = None,
        snr_ids: Optional[torch.Tensor] = None,
        noise_ids: Optional[torch.Tensor] = None,
        cond_feat: Optional[torch.Tensor] = None,
        snr_logits_pred: Optional[torch.Tensor] = None,
        noise_logits_pred: Optional[torch.Tensor] = None,
        snr_value_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source = str(condition_source or "none").strip().lower()
        if source not in {"none", "oracle", "predicted", "hybrid"}:
            source = "none"

        if source == "hybrid":
            alpha = float(0.5 if hybrid_alpha is None else hybrid_alpha)
            alpha = max(0.0, min(1.0, alpha))
            gate_oracle = self._forward_single(
                pooled_raw=pooled_raw,
                pooled_enh=pooled_enh,
                condition_source="oracle",
                snr_values=snr_values,
                snr_ids=snr_ids,
                noise_ids=noise_ids,
                cond_feat=cond_feat,
                snr_logits_pred=snr_logits_pred,
                noise_logits_pred=noise_logits_pred,
                snr_value_pred=snr_value_pred,
            )
            gate_predicted = self._forward_single(
                pooled_raw=pooled_raw,
                pooled_enh=pooled_enh,
                condition_source="predicted",
                snr_values=snr_values,
                snr_ids=snr_ids,
                noise_ids=noise_ids,
                cond_feat=cond_feat,
                snr_logits_pred=snr_logits_pred,
                noise_logits_pred=noise_logits_pred,
                snr_value_pred=snr_value_pred,
            )
            return (1.0 - alpha) * gate_oracle + alpha * gate_predicted

        return self._forward_single(
            pooled_raw=pooled_raw,
            pooled_enh=pooled_enh,
            condition_source=source,
            snr_values=snr_values,
            snr_ids=snr_ids,
            noise_ids=noise_ids,
            cond_feat=cond_feat,
            snr_logits_pred=snr_logits_pred,
            noise_logits_pred=noise_logits_pred,
            snr_value_pred=snr_value_pred,
        )


class Wav2Vec2SERTeacher(nn.Module):
    def __init__(self, num_classes: int = 4, backbone_name: Optional[str] = None) -> None:
        super().__init__()
        model_path = backbone_name or _resolve_wav2vec2_path()
        local_only = os.path.isdir(model_path)
        self.backbone = Wav2Vec2Model.from_pretrained(
            model_path, local_files_only=local_only
        )
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def forward(
        self, wav: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor | Tuple[torch.Tensor, ...]]:
        outputs = self.backbone(
            wav,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        feature_mask = _compute_feature_mask(
            self.backbone, attention_mask, outputs.last_hidden_state.shape[1]
        )
        pooled = _masked_mean(outputs.last_hidden_state, feature_mask)
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "pooled": pooled,
            "feature_mask": feature_mask,
        }


class Wav2Vec2SERStudent(nn.Module):
    def __init__(
        self,
        config: Wav2Vec2Config,
        num_classes: int = 4,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model_args = dict(model_args or {})
        self.variant = str(self.model_args.get("variant", "mlkd_baseline"))
        self.use_dual_path = bool(self.model_args.get("use_dual_path", False))
        self.use_enhancement = bool(self.model_args.get("use_enhancement", False))
        self.separate_branch_backbones = bool(
            self.model_args.get("separate_branch_backbones", False)
        )
        self.gate_type = str(self.model_args.get("gate_type", "estimated"))
        self.use_oracle_snr = bool(self.model_args.get("use_oracle_snr", False))
        self.use_oracle_noise = bool(self.model_args.get("use_oracle_noise", False))
        default_gate_condition_mode = (
            "oracle" if (self.use_oracle_snr or self.use_oracle_noise) else "none"
        )
        self.gate_condition_mode = str(
            self.model_args.get("gate_condition_mode", default_gate_condition_mode)
        ).strip().lower()
        if self.gate_condition_mode not in {"none", "oracle", "predicted", "hybrid"}:
            self.gate_condition_mode = "none"
        self.condition_feat_dim = int(max(1, self.model_args.get("condition_feat_dim", 64)))
        self.predict_snr_value = bool(
            self.model_args.get("predict_snr_value", True)
            or self.model_args.get("use_snr_regression", False)
        )
        self.num_snr_classes = int(self.model_args.get("num_snr_classes", 0) or 0)
        self.num_noise_classes = int(self.model_args.get("num_noise_classes", 0) or 0)
        self.use_aux_snr = bool(self.model_args.get("use_aux_snr", False))
        self.use_aux_noise = bool(self.model_args.get("use_aux_noise", False))

        self.backbone = Wav2Vec2Model(config)
        _disable_feature_input_requires_grad(self.backbone)
        if self.use_dual_path and self.separate_branch_backbones:
            self.enh_backbone = Wav2Vec2Model(config)
            _disable_feature_input_requires_grad(self.enh_backbone)
        else:
            self.enh_backbone = self.backbone

        if self.use_enhancement:
            enh_type = str(self.model_args.get("enhancement_type", "resunet"))
            if enh_type.lower() == "resunet":
                self.enhancement = ResUNetEnhancer1D(
                    base_channels=int(self.model_args.get("enhancement_channels", 32))
                )
            else:
                self.enhancement = IdentityEnhancer()
        else:
            self.enhancement = IdentityEnhancer()

        self.condition_estimator = None
        if self.use_dual_path and self.gate_type != "none":
            self.condition_estimator = ConditionEstimator(
                hidden_size=self.backbone.config.hidden_size,
                condition_feat_dim=self.condition_feat_dim,
                num_snr_classes=self.num_snr_classes,
                num_noise_classes=self.num_noise_classes,
                predict_snr_value=self.predict_snr_value,
            )

        if self.use_dual_path and self.gate_type != "none":
            self.gate = AdaptiveFusionGate(
                hidden_size=self.backbone.config.hidden_size,
                condition_feat_dim=self.condition_feat_dim,
                use_oracle_snr=self.use_oracle_snr,
                use_oracle_noise=self.use_oracle_noise,
                num_snr_classes=self.num_snr_classes,
                num_noise_classes=self.num_noise_classes,
                snr_temperature=float(self.model_args.get("snr_gate_temperature", 10.0)),
                snr_bias_strength=float(self.model_args.get("snr_gate_bias_strength", 0.75)),
            )
        else:
            self.gate = None

        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)
        self.snr_head = (
            nn.Linear(self.backbone.config.hidden_size, self.num_snr_classes)
            if self.use_aux_snr and self.num_snr_classes > 0
            else None
        )
        self.noise_head = (
            nn.Linear(self.backbone.config.hidden_size, self.num_noise_classes)
            if self.use_aux_noise and self.num_noise_classes > 0
            else None
        )

    @classmethod
    def from_teacher(
        cls,
        teacher: Wav2Vec2SERTeacher,
        num_classes: int = 4,
        student_cfg_overrides: Optional[Dict[str, Any]] = None,
        model_args: Optional[Dict[str, Any]] = None,
    ) -> "Wav2Vec2SERStudent":
        overrides = dict(student_cfg_overrides or {})
        teacher_cfg = teacher.backbone.config.to_dict()
        if "num_hidden_layers" in overrides and overrides["num_hidden_layers"] is not None:
            teacher_cfg["num_hidden_layers"] = int(overrides.pop("num_hidden_layers"))
        student_cfg = Wav2Vec2Config(**{**teacher_cfg, **overrides})
        student = cls(config=student_cfg, num_classes=num_classes, model_args=model_args)
        student.init_from_teacher(teacher)
        return student

    @staticmethod
    def _build_layer_mapping(student_layers: int, teacher_layers: int) -> list[tuple[int, int]]:
        if student_layers <= 0 or teacher_layers <= 0:
            return []
        if student_layers == 6 and teacher_layers >= 12:
            return list(zip(range(6), [1, 3, 5, 7, 9, 11]))
        if student_layers == teacher_layers:
            return list(zip(range(student_layers), range(teacher_layers)))
        t_idx = torch.linspace(0, teacher_layers - 1, steps=student_layers).round().long().tolist()
        return list(zip(range(student_layers), t_idx))

    @staticmethod
    def _copy_backbone_weights(dst_backbone: Wav2Vec2Model, src_backbone: Wav2Vec2Model) -> list:
        try:
            dst_backbone.feature_extractor.load_state_dict(src_backbone.feature_extractor.state_dict())
        except RuntimeError:
            pass
        try:
            dst_backbone.feature_projection.load_state_dict(src_backbone.feature_projection.state_dict())
        except RuntimeError:
            pass
        try:
            dst_backbone.encoder.pos_conv_embed.load_state_dict(src_backbone.encoder.pos_conv_embed.state_dict())
        except RuntimeError:
            pass
        if hasattr(dst_backbone.encoder, "layer_norm") and hasattr(src_backbone.encoder, "layer_norm"):
            try:
                dst_backbone.encoder.layer_norm.load_state_dict(src_backbone.encoder.layer_norm.state_dict())
            except RuntimeError:
                pass

        mapping = Wav2Vec2SERStudent._build_layer_mapping(
            len(dst_backbone.encoder.layers), len(src_backbone.encoder.layers)
        )
        for s_idx, t_idx in mapping:
            try:
                dst_backbone.encoder.layers[s_idx].load_state_dict(
                    src_backbone.encoder.layers[t_idx].state_dict()
                )
            except RuntimeError:
                continue
        return mapping

    def init_from_teacher(self, teacher: Wav2Vec2SERTeacher) -> None:
        mapping = self._copy_backbone_weights(self.backbone, teacher.backbone)
        if self.enh_backbone is not self.backbone:
            self._copy_backbone_weights(self.enh_backbone, teacher.backbone)
        if self.classifier.weight.shape == teacher.classifier.weight.shape:
            self.classifier.load_state_dict(teacher.classifier.state_dict())
        print("Initialized student from teacher layers:", mapping)

    def get_model_meta(self) -> Dict[str, Any]:
        return dict(self.model_args)

    def freeze_feature_extractor(self, freeze: bool = True) -> None:
        for p in self.backbone.feature_extractor.parameters():
            p.requires_grad = not freeze
        if self.enh_backbone is not self.backbone:
            for p in self.enh_backbone.feature_extractor.parameters():
                p.requires_grad = not freeze

    @staticmethod
    def _run_wav2vec(
        backbone: Wav2Vec2Model, wav: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ):
        return backbone(
            wav,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    @staticmethod
    def _fuse_hidden_states(
        raw_states: Tuple[torch.Tensor, ...],
        enh_states: Tuple[torch.Tensor, ...],
        gate: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        n = min(len(raw_states), len(enh_states))
        gate_t = gate.unsqueeze(-1)
        fused = []
        for i in range(n):
            fused.append(gate_t * enh_states[i] + (1.0 - gate_t) * raw_states[i])
        return tuple(fused)

    @staticmethod
    def _resolve_condition_source(condition_source: Optional[str], default_source: str) -> str:
        source = str(condition_source or default_source or "none").strip().lower()
        if source not in {"none", "oracle", "predicted", "hybrid"}:
            source = "none"
        return source

    def forward(
        self,
        wav: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        snr_values: Optional[torch.Tensor] = None,
        snr_ids: Optional[torch.Tensor] = None,
        noise_ids: Optional[torch.Tensor] = None,
        condition_source: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.use_dual_path:
            branch_wav = self.enhancement(wav) if self.use_enhancement else wav
            outputs = self._run_wav2vec(self.backbone, branch_wav, attention_mask)
            feature_mask = _compute_feature_mask(
                self.backbone, attention_mask, outputs.last_hidden_state.shape[1]
            )
            pooled = _masked_mean(outputs.last_hidden_state, feature_mask)
            logits = self.classifier(pooled)
            out: Dict[str, Any] = {
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "pooled": pooled,
                "feature_mask": feature_mask,
                "enhanced_wav": branch_wav,
                "gate": torch.zeros(wav.shape[0], 1, device=wav.device),
                "gate_condition_source": "none",
                "cond_feat": None,
                "snr_logits_pred": None,
                "noise_logits_pred": None,
                "snr_value_pred": None,
            }
            if self.snr_head is not None:
                out["snr_logits"] = self.snr_head(pooled)
            if self.noise_head is not None:
                out["noise_logits"] = self.noise_head(pooled)
            return out

        enhanced_wav = self.enhancement(wav)
        raw_out = self._run_wav2vec(self.backbone, wav, attention_mask)
        enh_out = self._run_wav2vec(self.enh_backbone, enhanced_wav, attention_mask)

        feature_mask = _compute_feature_mask(
            self.backbone, attention_mask, raw_out.last_hidden_state.shape[1]
        )
        pooled_raw = _masked_mean(raw_out.last_hidden_state, feature_mask)
        pooled_enh = _masked_mean(enh_out.last_hidden_state, feature_mask)

        cond_feat = None
        snr_logits_pred = None
        noise_logits_pred = None
        snr_value_pred = None
        if self.condition_estimator is not None:
            cond_outputs = self.condition_estimator(pooled_raw, pooled_enh)
            cond_feat = cond_outputs.get("cond_feat")
            snr_logits_pred = cond_outputs.get("snr_logits_pred")
            noise_logits_pred = cond_outputs.get("noise_logits_pred")
            snr_value_pred = cond_outputs.get("snr_value_pred")

        gate_source = self._resolve_condition_source(condition_source, self.gate_condition_mode)
        if self.gate is None:
            gate_source = "none"
            gate = torch.full((wav.shape[0], 1), 0.5, device=wav.device, dtype=pooled_raw.dtype)
        else:
            gate = self.gate(
                pooled_raw=pooled_raw,
                pooled_enh=pooled_enh,
                condition_source=gate_source,
                hybrid_alpha=hybrid_alpha,
                snr_values=snr_values,
                snr_ids=snr_ids,
                noise_ids=noise_ids,
                cond_feat=cond_feat,
                snr_logits_pred=snr_logits_pred,
                noise_logits_pred=noise_logits_pred,
                snr_value_pred=snr_value_pred,
            )

        fused_states = self._fuse_hidden_states(raw_out.hidden_states, enh_out.hidden_states, gate)
        fused_last = fused_states[-1]
        pooled = _masked_mean(fused_last, feature_mask)
        logits = self.classifier(pooled)

        out: Dict[str, Any] = {
            "logits": logits,
            "hidden_states": fused_states,
            "pooled": pooled,
            "feature_mask": feature_mask,
            "raw_pooled": pooled_raw,
            "enh_pooled": pooled_enh,
            "enhanced_wav": enhanced_wav,
            "gate": gate,
            "gate_condition_source": gate_source,
            "raw_hidden_states": raw_out.hidden_states,
            "enh_hidden_states": enh_out.hidden_states,
            "cond_feat": cond_feat,
            "snr_logits_pred": snr_logits_pred,
            "noise_logits_pred": noise_logits_pred,
            "snr_value_pred": snr_value_pred,
        }
        if self.snr_head is not None:
            out["snr_logits"] = self.snr_head(pooled)
        if self.noise_head is not None:
            out["noise_logits"] = self.noise_head(pooled)
        return out
