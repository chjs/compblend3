"""ChunkedKVStore — chunk별 분리 K/V 저장 + DynamicCache 양방향 변환.

Step 3 scope (자료구조 + 메타 + Cache 인터페이스 호환성):
  - DynamicCache → ChunkedKVStore → DynamicCache' roundtrip K/V bitwise (3.1)
  - ChunkMeta 7 필드 보존 (3.2)
  - 변환된 DynamicCache 인스턴스의 Cache 인터페이스 충족 (3.3A)
  - 모델 forward 시 logits SHA-256 일치 (3.3B, vast.ai)

해석 A 채택 (Step 3 결정):
  - transformers Cache 상속 ❌ — dataclass-like container.
  - 모델 forward에는 `to_dynamic_cache()`의 반환값 (DynamicCache 인스턴스) 전달.
  - RoPE re-rotation은 Step 4.1로 이연. Step 3 ChunkedKVStore는 raw K/V만 저장·복원.

DECISIONS.md §3.8 명세 그대로:
  - Chunk K/V shape: `(H_kv, T_chunk, D)` per layer per chunk. **B 차원 제거**.
  - GQA repeat 적용 ❌ (메모리 절약).
  - Phase 1~6 DynamicCache만 지원.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class ChunkMeta:
    """Per-chunk metadata (DECISIONS §3.8 7 필드 그대로)."""
    chunk_id: str               # 예: "system", "doc_0", "doc_1"
    token_ids: list[int]        # 이 chunk의 토큰 id
    original_offset: int        # 원본 prefill 시의 시작 position
    new_offset: int             # blend 시의 시작 position (Step 4+ 시 의미 가짐)
    original_length: int        # 토큰 수 (== len(token_ids) 권장)
    is_cacheable: bool          # query는 False, system/doc은 True
    is_permanent_hit: bool      # system은 True, doc은 False


@dataclass
class ChunkedKVStore:
    """chunk별 K/V 분리 저장. dataclass container (Cache 상속 ❌).

    Storage layout:
      chunks: {chunk_id -> ChunkMeta}
      kv:     {chunk_id -> [(K_layer_i, V_layer_i)] of length num_layers}
              K, V shape: (H_kv, T_chunk, D)  — DECISIONS §3.8, B 차원 제거.
      num_layers: 모델 layer 수 (Mistral 7B = 32).
    """
    chunks: dict[str, ChunkMeta]
    kv: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]
    num_layers: int

    @classmethod
    def from_dynamic_cache(
        cls,
        dc: DynamicCache,
        chunk_spec: list[ChunkMeta],
    ) -> "ChunkedKVStore":
        """DynamicCache의 layer별 K/V를 chunk_spec에 따라 slicing해서 분리 저장.

        slicing 영역은 `ChunkMeta.original_offset` ~ `+original_length`.
        결과 K/V shape: `(H_kv, T_chunk, D)` per chunk per layer (B 차원 제거).

        가정:
          - dc.key_cache[i].shape[0] == 1 (B=1, 본 phase).
          - chunk_spec의 [original_offset, original_offset+original_length) 가
            dc seq_len 범위 안에 모두 들어감.
          - chunk_id는 고유 (중복 시 에러).
        """
        if not isinstance(dc, DynamicCache):
            raise TypeError(f"expected DynamicCache, got {type(dc).__name__}")
        num_layers = len(dc.key_cache)
        if num_layers == 0:
            raise ValueError("DynamicCache is empty (no layers populated)")
        seq_len = dc.key_cache[0].shape[-2]
        chunks_dict: dict[str, ChunkMeta] = {}
        kv_dict: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for cm in chunk_spec:
            if cm.chunk_id in chunks_dict:
                raise ValueError(f"duplicate chunk_id: {cm.chunk_id!r}")
            start = cm.original_offset
            end = start + cm.original_length
            if start < 0 or end > seq_len:
                raise ValueError(
                    f"chunk {cm.chunk_id!r} range [{start},{end}) out of cache "
                    f"seq_len={seq_len}"
                )
            chunks_dict[cm.chunk_id] = cm
            chunk_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i in range(num_layers):
                k = dc.key_cache[i]      # (B, H_kv, T, D)
                v = dc.value_cache[i]    # (B, H_kv, T, D)
                assert k.shape[0] == 1, f"expected B=1, got {k.shape[0]}"
                # B 차원 제거 + T 차원 slicing
                k_chunk = k[0, :, start:end, :].detach().clone()   # (H_kv, T_chunk, D)
                v_chunk = v[0, :, start:end, :].detach().clone()   # (H_kv, T_chunk, D)
                chunk_kv.append((k_chunk, v_chunk))
            kv_dict[cm.chunk_id] = chunk_kv
        return cls(chunks=chunks_dict, kv=kv_dict, num_layers=num_layers)

    def to_dynamic_cache(self) -> DynamicCache:
        """저장된 chunk들을 `new_offset` 순서로 concat하여 DynamicCache 생성.

        - new_offset 순 정렬 후 layer별 K/V tensor 재구성.
        - B=1 dimension 복원 (unsqueeze(0)).
        - Step 3 scope에서는 new_offset == original_offset 인 케이스만 검증
          (재배열은 Step 4+ blend logic 책임).
        """
        # new_offset 순 정렬 — 같은 값이면 chunk_id로 tie-break (결정론)
        ordered = sorted(
            self.chunks.values(),
            key=lambda cm: (cm.new_offset, cm.chunk_id),
        )
        dc = DynamicCache()
        for layer_idx in range(self.num_layers):
            k_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []
            for cm in ordered:
                k_chunk, v_chunk = self.kv[cm.chunk_id][layer_idx]
                # K/V shape: (H_kv, T_chunk, D)
                k_parts.append(k_chunk)
                v_parts.append(v_chunk)
            # concat along T dim → (H_kv, T_total, D) → unsqueeze B → (1, H_kv, T_total, D)
            k_full = torch.cat(k_parts, dim=-2).unsqueeze(0)
            v_full = torch.cat(v_parts, dim=-2).unsqueeze(0)
            dc.update(k_full, v_full, layer_idx)
        return dc
