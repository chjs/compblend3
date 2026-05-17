"""RoPE re-rotation for CacheBlend chunk reuse (Step 4.1).

배경:
  각 chunk를 독립적으로 prefill하면 chunk 내부 token이 항상 position 0부터
  적용된 RoPE를 가진다. blend 시 chunk가 sequence 안에서 실제 position p로
  재배치되려면 K에 적용된 RoPE를 R(p_new - p_old)만큼 추가 회전해야 한다.

RoPE 회전의 group property:
  R(α) · R(β) = R(α + β)
  R(p_new) · R(-p_old) = R(p_new - p_old)
  K_old = R(p_old) · K_raw  (chunk 내부 prefill 결과)
  K_new = R(p_new) · K_raw = R(p_new - p_old) · K_old

따라서 chunk K에 R(p_new - p_old) shift만 적용하면 됨. V는 RoPE 없음, 변경 ❌.

본 모듈은 fork modeling_mistral.py의 `apply_rotary_pos_emb`와 같은 RoPE 정의를
공유한다. Mistral는 GPT-NeoX 스타일 (split half + concat) 사용.

Step 4 scope:
  - re_rotate_k: 단일 chunk K에 position shift 적용 (single chunk)
  - re_rotate_chunked_store_k: ChunkedKVStore의 각 chunk K에 (new_offset -
    original_offset) shift 적용 (multi-chunk)

V는 회전 ❌ (RoPE는 Q·K에만).
"""
from __future__ import annotations

import torch


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Mistral apply_rotary_pos_emb과 동일 정의 (fork modeling_mistral.py:68-71).

    Mistral은 GPT-NeoX 스타일: last dim을 반으로 나눠 [x1, x2] → [-x2, x1].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _compute_rope_freqs(
    head_dim: int,
    positions: torch.Tensor,
    rope_theta: float = 10000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """주어진 position에 대한 (cos, sin) 계산.

    Mistral의 MistralRotaryEmbedding (fork modeling_mistral.py:296-307) 과
    동일 결과를 내야 함. attention_scaling은 default rope_type="default"에서
    1.0 (간략화 — Step 4 smoke 가정).

    Returns:
      cos, sin: shape (T, head_dim). RoPE concat 후 형태.
    """
    half = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, half, dtype=torch.float32, device=device) / half
    ))                                                      # (half,)
    pos = positions.to(device=device, dtype=torch.float32)  # (T,)
    freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)       # (T, half)
    emb = torch.cat((freqs, freqs), dim=-1)                 # (T, head_dim)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def re_rotate_k(
    k: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """RoPE-rotated K에 position shift 적용.

    Args:
      k: (H_kv, T, D) — chunk 내부 prefill로 얻은 RoPE-적용된 K (B 차원 제거,
         ChunkedKVStore 형식).
      old_positions: (T,) — chunk가 prefill될 때 사용한 RoPE position (보통
         [0, 1, ..., T-1]).
      new_positions: (T,) — chunk가 blend 후 갖게 될 실제 position
         (예: [p, p+1, ..., p+T-1]).
      rope_theta: Mistral의 rope_theta (config.rope_theta). v0.2는 10000.

    Returns:
      k_new: (H_kv, T, D) — R(new_pos - old_pos)이 추가 적용된 K.

    Mechanism:
      K_old = R(old) · K_raw
      K_new = R(new) · K_raw = R(new - old) · K_old

      구현은 `apply_rotary_pos_emb`와 동일하나 적용 angle이 (new - old).
    """
    assert k.dim() == 3, f"expected (H_kv, T, D), got {tuple(k.shape)}"
    H_kv, T, D = k.shape
    assert old_positions.shape == (T,), f"old_positions shape mismatch: {old_positions.shape}"
    assert new_positions.shape == (T,), f"new_positions shape mismatch: {new_positions.shape}"
    delta = new_positions.to(torch.float32) - old_positions.to(torch.float32)
    cos, sin = _compute_rope_freqs(D, delta, rope_theta=rope_theta,
                                     device=k.device, dtype=k.dtype)
    # cos/sin: (T, D), k: (H_kv, T, D) — unsqueeze head 차원으로 broadcast
    cos_b = cos.unsqueeze(0)   # (1, T, D)
    sin_b = sin.unsqueeze(0)   # (1, T, D)
    k_rotated = k * cos_b + _rotate_half(k) * sin_b
    return k_rotated


def re_rotate_chunked_store_k_inplace(store) -> None:
    """ChunkedKVStore의 각 chunk K에 (new_offset - original_offset) shift 적용.

    각 chunk는 original_offset에서 prefill되었다 가정 (또는 동등하게 0에서
    prefill 후 original_offset이 0인 채로 저장). new_offset이 다른 경우
    shift = new_offset - original_offset 적용.

    K만 회전. V는 무변경.

    원본 K 텐서를 modify (`.copy_`). 호출자가 store를 따로 보존하려면 사전
    deepcopy 필요.

    가정:
      - 각 chunk의 RoPE는 original_offset 기준으로 적용되어 저장됨.
      - Mistral v0.2 기본 rope_theta=10000.
    """
    from compblend.cache import ChunkedKVStore  # circular import 방지
    assert isinstance(store, ChunkedKVStore)
    for chunk_id, cm in store.chunks.items():
        if cm.new_offset == cm.original_offset:
            continue  # shift 0 — re-rotation 불필요
        T = cm.original_length
        old_pos = torch.arange(cm.original_offset, cm.original_offset + T)
        new_pos = torch.arange(cm.new_offset, cm.new_offset + T)
        for layer_idx in range(store.num_layers):
            k, v = store.kv[chunk_id][layer_idx]
            # device/dtype 일치 (CPU 또는 GPU)
            old_p = old_pos.to(k.device)
            new_p = new_pos.to(k.device)
            k_new = re_rotate_k(k, old_p, new_p)
            # in-place tuple replacement (kv list는 mutable tuple, replace)
            store.kv[chunk_id][layer_idx] = (k_new, v)
