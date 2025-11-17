"""Custom loss utilities for class-balanced focal loss with uncertainty regularization."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


def _ensure_tensor_counts(class_counts: Sequence[int]) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    if counts.ndim != 1:
        raise ValueError("class_counts must be a 1-D sequence of counts")
    return counts


def _extract_label(target, num_classes: int) -> int:
    """Safely convert dataset targets to integer class indices."""
    if isinstance(target, torch.Tensor):
        if target.ndim == 0:
            return int(target.item())
        if target.ndim == 1 and target.numel() == num_classes:
            return int(target.argmax().item())
        raise ValueError(f"Unsupported tensor label shape {tuple(target.shape)}")
    if isinstance(target, (int, np.integer)):
        return int(target)
    if isinstance(target, Sequence):
        if len(target) == num_classes:
            return int(np.argmax(np.asarray(target)))
    raise ValueError(f"Unsupported label type: {type(target)}")


def compute_class_counts(dataset: Dataset, num_classes: int) -> torch.Tensor:
    """Iterate over a dataset (or subset) to collect per-class sample counts."""
    counts = torch.zeros(num_classes, dtype=torch.long)

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        for index in dataset.indices:
            _, target = base_dataset[index]
            label = _extract_label(target, num_classes)
            counts[label] += 1
    else:
        for _, target in dataset:
            label = _extract_label(target, num_classes)
            counts[label] += 1

    return counts


class ClassBalancedFocalLoss(nn.Module):
    """Focal loss with class-balanced re-weighting for long-tailed datasets."""

    def __init__(
        self,
        class_counts: Sequence[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        counts = _ensure_tensor_counts(class_counts)
        if counts.numel() == 0:
            raise ValueError("class_counts must contain at least one class")

        effective_num = 1.0 - torch.pow(torch.clamp(counts, min=0.0), beta)
        weights = (1.0 - beta) / torch.clamp(effective_num, min=eps)
        weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
        weight_sum = weights.sum().item()
        if weight_sum > 0:
            weights = weights * (weights.numel() / weight_sum)
        else:
            weights = torch.ones_like(weights)

        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("logits tensor must be 2-D (batch_size, num_classes)")

        num_classes = logits.size(-1)

        if targets.ndim == 1:
            targets = F.one_hot(targets.to(torch.long), num_classes=num_classes).float()
        elif targets.ndim != 2 or targets.size(-1) != num_classes:
            raise ValueError("targets must be either class indices or soft labels matching logits")

        class_weights = self.class_weights.to(logits.device)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(torch.clamp(probs, min=self.eps))

        pt = (probs * targets).sum(dim=-1)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        example_weights = (class_weights.unsqueeze(0) * targets).sum(dim=-1)

        loss = -example_weights * focal_factor * (targets * log_probs).sum(dim=-1)
        return loss.mean()


class UncertaintyRegularizer(nn.Module):
    """Regularize predictive uncertainty, e.g. via entropy minimization."""

    def __init__(self, weight: float = 0.0, mode: str = "entropy", eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = weight
        self.mode = mode
        self.eps = eps

    def forward(self, logits: torch.Tensor, _: torch.Tensor | None = None) -> torch.Tensor:
        if self.weight <= 0:
            return logits.new_tensor(0.0)

        probs = torch.softmax(logits, dim=-1)

        if self.mode == "entropy":
            entropy = -(probs * torch.log(torch.clamp(probs, min=self.eps))).sum(dim=-1)
            reg_term = entropy.mean()
        elif self.mode == "confidence":
            confidence = probs.max(dim=-1).values
            reg_term = (1.0 - confidence).mean()
        else:
            raise ValueError(f"Unsupported uncertainty mode: {self.mode}")

        return reg_term * self.weight
```,