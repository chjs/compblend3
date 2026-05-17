"""CacheBlend forward (minimal scaffold for Step 6).

Step 6 scope (recompute_ratio=1.0 only):
  - blended cache가 주어진 상태에서 모든 token·모든 layer를 재계산하는 경로
  - mechanism적으로 vanilla full prefill과 동등 (cache 입력 무시, 새 forward)
  - vanilla logits와 bitwise 일치 검증

Step 7+ scope:
  - recompute_ratio < 1.0: HKVD 기반 selective token recompute
  - partial cache reuse + layer-wise mask
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache


def cacheblend_forward_full_recompute(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    blended_cache: DynamicCache | None = None,
    recompute_ratio: float = 1.0,
) -> torch.Tensor:
    """recompute_ratio=1.0 모드의 CacheBlend forward.

    `blended_cache` (chunk별 prefill 후 concat된 cache)를 입력으로 받지만
    recompute_ratio=1.0이면 그 cache를 사용 ❌ — 모든 token을 새로 prefill
    하여 vanilla와 동등한 logits을 만든다.

    Args:
      model: MistralForCausalLM (or any HF causal LM).
      input_ids: (B, T) — full sequence.
      attention_mask: (B, T) — full mask.
      blended_cache: ChunkedKVStore.to_dynamic_cache() 결과 (recompute_ratio=1.0
                     모드에서 사용되지 ❌, 미래 selective recompute 검증 시점에
                     활성).
      recompute_ratio: 1.0만 지원 (Step 6 범위). 다른 값은 NotImplementedError.

    Returns:
      logits: (B, T, vocab).

    Step 6 의 validation rationale:
      "recompute_ratio=1.0 에서도 vanilla와 안 맞으면, partial recompute는 더
       안 맞는다" — 본 함수가 vanilla forward와 동등함을 보임으로써 추후
       selective recompute 의 fallback path를 확보.
    """
    if recompute_ratio != 1.0:
        raise NotImplementedError(
            f"Step 6 scope = recompute_ratio=1.0 only. got {recompute_ratio}. "
            "Selective recompute (HKVD) 은 Step 7."
        )
    # blended_cache는 입력으로 받았으나 100% recompute path에서는 무시.
    # vanilla forward (cache 없이) 실행 — Step 0~3에서 검증된 deterministic path.
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits
