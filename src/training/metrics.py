from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def confusion_matrix(labels: List[int], preds: List[int], num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y_true, y_pred in zip(labels, preds):
        cm[y_true, y_pred] += 1
    return cm


def unweighted_accuracy(cm: np.ndarray) -> float:
    recalls = []
    for i in range(cm.shape[0]):
        denom = cm[i].sum()
        if denom == 0:
            recalls.append(0.0)
        else:
            recalls.append(cm[i, i] / denom)
    return float(np.mean(recalls))


@torch.no_grad()
def evaluate_classifier(
    model,
    dataloader,
    device: torch.device,
    num_classes: int,
    input_key: str = "input_values",
    pass_condition_labels: bool = True,
    condition_source_override: Optional[str] = None,
    hybrid_alpha_override: Optional[float] = None,
) -> Dict:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    total_loss = 0.0
    total_count = 0

    for batch in dataloader:
        inputs = batch[input_key].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        snr_values = batch.get("snr_values")
        snr_ids = batch.get("snr_ids")
        noise_ids = batch.get("noise_type_ids")
        forward_kwargs = {}
        if hasattr(model, "model_args"):
            if condition_source_override is not None:
                forward_kwargs["condition_source"] = str(condition_source_override)
            if hybrid_alpha_override is not None:
                forward_kwargs["hybrid_alpha"] = float(hybrid_alpha_override)
            if pass_condition_labels:
                if snr_values is not None:
                    forward_kwargs["snr_values"] = snr_values.to(device)
                if snr_ids is not None:
                    forward_kwargs["snr_ids"] = snr_ids.to(device)
                if noise_ids is not None:
                    forward_kwargs["noise_ids"] = noise_ids.to(device)

        outputs = model(inputs, attention_mask=attention_mask, **forward_kwargs)
        logits = outputs["logits"]
        loss = F.cross_entropy(logits, labels)

        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)

        preds = torch.argmax(logits, dim=-1)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds, num_classes)
    ua = unweighted_accuracy(cm)
    avg_loss = total_loss / max(1, total_count)
    return {"ua": ua, "loss": avg_loss, "confusion_matrix": cm}
