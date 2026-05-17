# Step 4 — N chunks 따로 prefill → concat = vanilla full prefill (with RoPE re-rotation)

> 2026-05-18 (overnight round): stub → 자체완결 spec 확장.
> Phase 2 진입 (CacheBlend 핵심). DECISIONS §3.8 / §13 v13 사전 가정 기반.

---

## 1. 목표

각 chunk를 독립적으로 prefill하여 얻은 K/V를 RoPE re-rotation으로 sequence position에 정렬한 후 concat한 cache가 vanilla full prefill의 cache와 얼마나 일치하는지 검증·정량화.

**범위 명확**: Step 4는 **100% chunk reuse (recompute 없음)** 상태에서의 동등성·drift 측정. selective recompute (HKVD)는 Step 7, recompute_ratio=1.0 = vanilla는 Step 6 범위.

## 2. 사전 확인 / 외부 코드 의존 (CLAUDE.md §4.5 (d))

| 항목 | 확인 결과 |
|---|---|
| Mistral RoPE 정의 | fork `modeling_mistral.py:68-95` `apply_rotary_pos_emb` + `_rotate_half`. GPT-NeoX 스타일 (split half + concat) |
| Mistral RoPE freqs | `MistralRotaryEmbedding:296-307` — `inv_freq` matmul with `position_ids` → cos/sin |
| `rope_theta` (Mistral v0.2) | 10000 (default rope_type) |
| `H_kv` / `D` | 8 / 128 (DECISIONS §3.8) |
| ChunkedKVStore (Step 3) | `src/compblend/cache.py` — `from_dynamic_cache` / `to_dynamic_cache`. `new_offset` 순 concat 지원 |
| RoPE group property | R(α)·R(β) = R(α+β), R(p_new − p_old)·R(p_old) = R(p_new). 수학적으로 K_new = R(p_new − p_old)·K_old |
| Step 2 cross-shape GEMM | `q_proj`(4096→4096)는 shape-dependent, `k_proj`/`v_proj`(4096→1024)는 shape-invariant kernel. K/V는 same-shape에서 bitwise, cross-shape에서도 1024 출력은 bitwise (C-7 결과) |

## 3. Step 4 원칙

- `src/compblend/modeling/` fork 무수정 (Step 1·2·3 원칙).
- `src/compblend/cache.py` (Step 3 `ChunkedKVStore`) 재사용.
- 신규 모듈: `src/compblend/rope_rotation.py` — re-rotation helpers.
- chunk padding 정책 (DECISIONS §13 v13): right-padding · bucket size · PAD K/V 저장 ❌. caller (검증 스크립트) 가 정책 준수 — `ChunkMeta.original_length`는 real token 수만.
- drift_budget (Step 2 1e-4) 의 Step 4 적용 여부: **4.1·4.2는 bitwise** (model-less / same-shape). **4.3은 atol/drift measurement** (cross-attention 누락 영향 정량화, cross-shape GEMM 위험 동반). 4.3 gate는 budget pass/fail이 아니라 drift 측정값 + budget exceeded flag.

## 4. Invariants

### 4.1 RoPE re-rotation self-consistency (model-less, atol 1e-6)

`re_rotate_k`가 group property를 atol 안에서 구현하는지:

```
K_raw (random)
K_old        = R(old_pos)·K_raw                            # 우리 _compute_rope_freqs 사용
K_new_actual = re_rotate_k(K_old, old_pos, new_pos)        # = R(new − old)·K_old
K_new_target = R(new_pos)·K_raw                            # 동일 _compute_rope_freqs

|K_new_actual − K_new_target|_max ≤ 1e-6
```

**Gate**: atol 1e-6 (fp32 RoPE composition noise floor). 초기 가설 (bitwise) 은 MacBook smoke에서 max_abs ~5e-7로 정정 — fp32에서 R(a+b) 직접 계산 vs R(a)·R(b) 합성은 cos/sin trig identity의 fp32 누적 오차로 bitwise 보장 안 됨 (mechanism). atol 1e-6 은 측정된 RoPE 단계 noise floor 위.

### 4.2 ChunkedKVStore reordering storage (model-less, bitwise)

Step 3.1의 reorder 확장. `new_offset != original_offset` 인 chunk_spec 에서 `to_dynamic_cache()`가 `new_offset` 순으로 concat:

```
chunk_spec: 3 chunks, new_offset = [4, 0, 2] (즉 reorder)
expected concat order: chunk@new=0 → chunk@new=2 → chunk@new=4
torch.equal(to_dynamic_cache().key_cache[i], expected_concat[i]) == True
```

### 4.3 Multi-chunk vanilla equivalence (model-backed, vast.ai, drift measurement)

```
prompt (6 tokens) → N chunks (2 tokens each, N=3)
각 chunk i 독립 prefill at position 0 → K/V_i (T=2)
re-rotate K_i to new_offset = 2*i
ChunkedKVStore.to_dynamic_cache() → D_blended
vanilla forward (use_cache=True) → D_vanilla
decode_blended = model(next_token, past_key_values=D_blended).logits
decode_vanilla = model(next_token, past_key_values=D_vanilla).logits

drift = (decode_blended − decode_vanilla).abs()
```

**Gate**: measurement only. `drift_budget_exceeded = (max_abs > 1e-2)` (loose, CacheBlend literature 기준 — 100% reuse w/o recompute 시 drift 큼). `step_04_final_gate_passed = 4.1 AND 4.2`. 4.3는 측정만, gate ❌.

추가 측정 (per-layer): chunk[i] re-rotated K[layer 0] vs vanilla K[pos 2*i..2*i+1, layer 0] — `k_proj` out_dim 1024라 bitwise 기대. Layer 1+는 cross-attention 누락 영향으로 drift 예상.

## 5. 구현 사양

### 5.1 신규 모듈 `src/compblend/rope_rotation.py`

- `_rotate_half`, `_compute_rope_freqs` — Mistral RoPE 정의 복제 (rope_theta=10000 default).
- `re_rotate_k(K, old_pos, new_pos)` — single chunk K re-rotation.
- `re_rotate_chunked_store_k_inplace(store)` — ChunkedKVStore 전체 in-place.

### 5.2 검증 스크립트

`tasks/step_04_multi_chunk_concat/run_multi_chunk_concat_check.py`:
- 4.1·4.2 (model-less, MacBook 가능).
- 4.3 `--enable-model-check` (vast.ai 전용).
- `summary.json`: invariants 4종 + `local_smoke_gate_passed` + `step_04_final_gate_passed` + 4.3 drift measurement.

### 5.3 Tensor shape

| 변수 | shape | 비고 |
|---|---|---|
| `K_raw` | `(H_kv=8, T, D=128)` | unrotated |
| `K_old / K_new` | 동상 | rotated at old / new position |
| `store.kv[chunk_id][layer_i].K` | `(H_kv=8, T_chunk, D=128)` | re-rotated K (in-place) |
| `D_blended.key_cache[i]` | `(1, 8, 6, 128)` | concat 후 B 복원 |
| `decode.logits[:, -1, :]` | `(1, 32000)` | 4.3 비교 대상 |

## 6. 검증 계측

- forward hook ❌ (Step 2·3 원칙).
- `past_key_values.key_cache/value_cache` 직접 접근.
- 4.3 drift = logits 전체 비교 + max_abs/mean_abs + argmax + top5 overlap.

## 7. 실행 환경

| 환경 | invariants |
|---|---|
| MacBook (model-less CPU) | 4.1, 4.2 |
| vast.ai A100-SXM4-80GB | 4.1, 4.2 (re-PASS) + 4.3 (model-backed) |

## 8. 결과 저장

`results/step_04/{macbook,vastai}/summary.json` — Step 3 hygiene 패턴 적용.

## 9. 통과 기준 / gate

| gate | 조건 |
|---|---|
| `local_smoke_gate_passed` | 4.1 (atol 1e-6) AND 4.2 (bitwise) |
| **`step_04_final_gate_passed`** | 4.1 AND 4.2 (4.3는 measurement, gate ❌) |
| `all_invariants_passed` | `= step_04_final_gate_passed` |
| 4.3 `drift_budget_exceeded` | `max_abs > 1e-2` (loose). 통과 조건 ❌, regression monitoring |

## 10. 작업 순서

1. 사전 확인 — DECISIONS / GOAL / fork 동작 확인 (완료).
2. `src/compblend/rope_rotation.py` 작성.
3. 검증 스크립트 작성.
4. MacBook smoke (4.1 + 4.2).
5. step branch commit + push.
6. vast.ai 실행 (4.1 + 4.2 + 4.3).
7. 보고서 + PROGRESS 갱신.
8. final gate PASS 시 main merge + tag.

## 11. 솔직성 노트

- **4.3 drift 크기 기대치**: CacheBlend paper에서 100% reuse w/o recompute는 F1 손실 큼. 4.3 logits drift도 O(1e-1)~O(1) 수준 예상. measurement만, gate ❌.
- **4.1 bitwise 가정**: 같은 `_compute_rope_freqs` 사용 시 R(new − old)·R(old) == R(new) 가 fp32에서 bitwise 가능 — `cos(a+b)` direct vs `cos(a)cos(b) − sin(a)sin(b)`가 fp32 bitwise 보장 ❌. **그러나 K_old를 우리 함수로 만들고 K_new도 우리 함수로 만들면** internal consistency는 bitwise. vanilla Mistral의 RoPE 결과와의 bitwise는 별도 (vast.ai 4.3 부분 측정).
- **GQA repeat 저장 ❌ 유지** (DECISIONS §3.8).
- **chunk padding 정책 caller responsibility**: `ChunkedKVStore`는 `chunk_spec.original_length` 따라 slicing. PAD position 제외는 caller 책임. `ChunkMeta.original_length == len(ChunkMeta.token_ids)` 권장.
- **`re_rotate_k`는 K만 회전** (V는 RoPE 없음, 무변경).

## 12. 다음 step 예고

- Step 5: 1 chunk reuse = vanilla (Step 3.3B의 N=1 특수 case 확장).
- Step 6: N chunks reuse, recompute_ratio=1.0 = vanilla (100% recompute 시 bitwise 회복).
- Step 7: HKVD oracle (selective recompute 알고리즘 정확성).
