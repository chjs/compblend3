"""HKVD (High KV Deviation) score 및 selective recompute index 선택.

Step 7 scope: 알고리즘 정확성 + numpy oracle 일치 검증.
실제 selective recompute forward 통합은 Step 8+ (또는 Phase 추후).

HKVD score 정의 (CC 자율 채택, CacheBlend paper 일반 패턴):
  score(t, L) = || K_actual[L, :, t, :] - K_reference[L, :, t, :] ||_2
  (per-token, per-layer L2 norm of K deviation between actual and reference)

  반환은 두 형식:
    - per-layer per-token score: (num_layers, T)
    - aggregated per-token (mean across layers): (T,)

선택 정책:
  recompute_ratio r ∈ [0, 1]: 상위 ceil(r * T) tokens 를 recompute.
  ratio=1.0 → 모든 token recompute (Step 6).
  ratio=0.0 → recompute ❌ (Step 4 100% reuse).

tie-break:
  같은 score 시 token index ascending (deterministic).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def hkvd_score_torch(
    k_actual: torch.Tensor,    # (num_layers, H_kv, T, D) — chunk reused 후 blended K
    k_reference: torch.Tensor, # (num_layers, H_kv, T, D) — vanilla full prefill K
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch HKVD score.

    Returns:
      per_layer_token: (num_layers, T) — per (layer, token) L2 norm.
      aggregated_token: (T,) — layer 평균.
    """
    assert k_actual.shape == k_reference.shape, (
        f"shape mismatch: actual={k_actual.shape} reference={k_reference.shape}"
    )
    assert k_actual.dim() == 4, f"expected (num_layers, H_kv, T, D), got {k_actual.shape}"
    diff = (k_actual - k_reference).to(torch.float32)
    # L2 norm per (layer, token): reduce over (H_kv, D)
    per_layer_token = torch.linalg.norm(diff.reshape(*diff.shape[:3], -1), dim=-1)
    aggregated_token = per_layer_token.mean(dim=0)
    return per_layer_token, aggregated_token


def hkvd_score_numpy_oracle(
    k_actual: np.ndarray,
    k_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """numpy oracle — PyTorch와 독립 구현. fp32로 강제."""
    assert k_actual.shape == k_reference.shape
    assert k_actual.ndim == 4
    diff = (k_actual.astype(np.float32) - k_reference.astype(np.float32))
    flat = diff.reshape(*diff.shape[:3], -1)
    per_layer_token = np.linalg.norm(flat, axis=-1)
    aggregated_token = per_layer_token.mean(axis=0)
    return per_layer_token, aggregated_token


def select_recompute_indices_torch(
    aggregated_token_score: torch.Tensor,    # (T,)
    recompute_ratio: float,
) -> torch.Tensor:
    """top ceil(r * T) tokens 선택 (highest score).

    tie-break: 같은 score 시 작은 token index 우선 (ascending).
    """
    assert 0.0 <= recompute_ratio <= 1.0
    T = aggregated_token_score.shape[0]
    k = math.ceil(recompute_ratio * T)
    if k == 0:
        return torch.empty(0, dtype=torch.long)
    if k >= T:
        return torch.arange(T)
    # tie-break: torch.sort는 stable 보장 → ascending sort 후 reverse 후 top-k
    # 또는 직접 score desc + index asc 정렬
    scores = aggregated_token_score.detach().cpu().float()
    # negative score asc + index asc → stable sort
    indices = torch.argsort(
        torch.stack([-scores, torch.arange(T, dtype=torch.float32)], dim=0),
        dim=-1,
    )[0][:k]
    # 더 간단하게:
    idx_sorted = sorted(range(T), key=lambda i: (-float(scores[i]), i))
    return torch.tensor(idx_sorted[:k], dtype=torch.long)


def select_recompute_indices_numpy_oracle(
    aggregated_token_score: np.ndarray,
    recompute_ratio: float,
) -> np.ndarray:
    """numpy oracle for top-k selection with tie-break."""
    assert 0.0 <= recompute_ratio <= 1.0
    T = aggregated_token_score.shape[0]
    k = math.ceil(recompute_ratio * T)
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k >= T:
        return np.arange(T, dtype=np.int64)
    scores = aggregated_token_score.astype(np.float32)
    idx_sorted = sorted(range(T), key=lambda i: (-float(scores[i]), i))
    return np.array(idx_sorted[:k], dtype=np.int64)
