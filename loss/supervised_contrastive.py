"""Supervised contrastive loss for object-level retrieval embeddings."""

from __future__ import annotations

import torch

def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    if not positive.any(dim=1).all():
        raise ValueError("every anchor needs a distinct same-class positive")
    logits = logits.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_log_prob = torch.where(positive, log_prob, torch.zeros_like(log_prob))
    return -(positive_log_prob.sum(dim=1) / positive.sum(dim=1)).mean()

