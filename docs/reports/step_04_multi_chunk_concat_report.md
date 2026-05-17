# Step 4 — N chunks 따로 prefill → concat (with RoPE re-rotation) 보고서

## 1. Summary

**Step 4 final gate PASS** ✅ — invariants 4.1·4.2 모두 PASS. 4.3는 measurement only (gate ❌).

| invariant | gate | 결과 |
|---|---|---|
| 4.1 RoPE re-rotation self-consistency | atol 1e-6 | ✅ (max ~5.7e-7) |
| 4.2 ChunkedKVStore reorder concat | bitwise (torch.equal) | ✅ (32 layer × K/V) |
| 4.3 Multi-chunk vanilla equivalence | measurement only | drift max_abs=8.46, argmax_match=True, top5=3/5, drift_budget_exceeded=True (1e-2) |

`step_04_final_gate_passed = True`, `all_invariants_passed = True`.

4.3의 큰 drift는 mechanism적 기대치 — 100% chunk reuse without recompute의 cross-attention 누락 영향이 layer 1+ K/V에 누적되어 logits에 큰 폭으로 영향. argmax는 유지 (decode 1 token 의미는 보존), top-5는 부분 변경. Step 6 (recompute_ratio=1.0)·Step 7 (HKVD)에서 회복 예정.

## 2. Goal and Scope

각 chunk를 독립적으로 prefill하여 얻은 K/V를 RoPE re-rotation으로 sequence position에 정렬한 후 concat한 cache가 vanilla full prefill의 cache와 얼마나 일치하는지 검증·정량화.

**100% chunk reuse (recompute 없음)** 상태에서의 동등성·drift 측정만. selective recompute (HKVD)는 Step 7, recompute_ratio=1.0 = vanilla는 Step 6 범위.

## 3. Environment

### 3.1 vast.ai

| 항목 | 값 |
|---|---|
| instance | 36952360 (할당 후 검증, destroy 완료) |
| GPU | NVIDIA A100-SXM4-80GB |
| dph_total | $1.07/h |
| torch / transformers / CUDA | 2.10.0+cu128 / 4.51.3 / 12.8 |
| 결정론 설정 | use_deterministic_algorithms(True), TF32 off, CUBLAS_WORKSPACE_CONFIG=:4096:8, fp32, eager |
| 추정 비용 | ~$0.20 (setup + run ~10분, 정확 비용은 vast.ai 콘솔) |
| 잔존 인스턴스 | 0 |
| (재할당) | 첫 인스턴스 36951804 stopped/loading 600s 초과 → destroy 후 재할당 (36952360) |

### 3.2 MacBook

| 항목 | 값 |
|---|---|
| model-less CPU smoke | 4.1 + 4.2 |
| local_smoke_gate_passed | True |

### 3.3 누적 비용 (추정, Step 0~4)

| Step | 추정 비용 |
|---|---|
| Step 0 / 1 / 2 / 3 | ~$0.05 / ~$0.16 / ~$1.00 / ~$0.15 |
| **Step 4** | **~$0.20** (재할당 + 본 실행) |
| **누적** | **~$1.56** |

정확 비용은 vast.ai 콘솔 기준 확인 필요.

## 4. Implementation Overview

### 4.1 신규 / 수정 파일

| 파일 | 변경 |
|---|---|
| `src/compblend/rope_rotation.py` | **신규**. `_rotate_half`, `_compute_rope_freqs`, `re_rotate_k`, `re_rotate_chunked_store_k_inplace` |
| `tasks/step_04_multi_chunk_concat.md` | stub → 12 § 확장 |
| `tasks/step_04_multi_chunk_concat/run_multi_chunk_concat_check.py` | 신규. 4-step 검증 |
| `results/step_04/{macbook,vastai}/summary.json` | 신규 |
| `src/compblend/modeling/` | **무수정** |

### 4.2 RoPE re-rotation 메커니즘

```
K_old = R(p_old) · K_raw          (chunk 독립 prefill 결과, p_old = chunk 내부 position)
K_new = R(p_new) · K_raw          (sequence 내 실제 position에 해당하는 RoPE)
     = R(p_new − p_old) · R(p_old) · K_raw
     = R(p_new − p_old) · K_old   (group property)
```

`re_rotate_k(K_old, p_old, p_new)` = R(p_new − p_old) 적용. V는 RoPE 없음, 무변경.

### 4.3 Multi-chunk 흐름 (4.3)

1. vanilla prefill: `model(prompt(6), use_cache=True)` → `D_vanilla`
2. 각 chunk i (T=2) 독립 prefill at position 0 → `K/V_i`
3. ChunkedKVStore에 chunk i 저장 (original_offset=0, new_offset=2*i)
4. `re_rotate_chunked_store_k_inplace` → chunk i의 K를 new_offset 위치로 회전
5. `to_dynamic_cache()` → `D_blended` (`new_offset` 순 concat)
6. decode 두 path 비교: `model(next_token, past_key_values=D_vanilla_copy)` vs `model(next_token, past_key_values=D_blended)`

## 5. Invariants and Gates

| ID | 명제 | gate | 결과 |
|---|---|---|---|
| **4.1** | re_rotate_k의 group property 일관성 | atol 1e-6 | ✅ max ~5.7e-7 |
| **4.2** | ChunkedKVStore reorder concat 정확성 | bitwise (torch.equal) | ✅ mismatched=[] |
| **4.3** | multi-chunk + RoPE re-rotation vs vanilla decode logits | measurement only | max_abs=8.46, gate ❌ |

| gate field | 값 |
|---|---|
| `local_smoke_gate_passed` | True (4.1 + 4.2) |
| `step_04_final_gate_passed` | True (4.1 + 4.2) |
| `all_invariants_passed` | True |
| 4.3 `drift_budget_exceeded` | True (1e-2 threshold, loose) |

## 6. Local MacBook Smoke

| invariant | 결과 |
|---|---|
| 4.1 | ✅ atol 1e-6 (per_case max 9.5e-7) |
| 4.2 | ✅ bitwise, 32 layer × K/V, mismatched=[] |
| 4.3 | ⏭️ skipped (CUDA unavailable) |

## 7. vast.ai Results

| invariant | 결과 |
|---|---|
| 4.1 | ✅ atol 1e-6, per_case 5 (shift 0/2/5/3→7/2→6) |
| 4.2 | ✅ bitwise, 32 layer × K/V |
| **4.3** | drift `max_abs=8.462e+00`, `mean_abs_diff=0.993`, `argmax_match=True`, `top5_overlap=3/5`, `drift_budget_exceeded=True` (1e-2) |

### 7.1 per-chunk layer-0 K bitwise check (vanilla K[pos 2i..2i+1] vs re-rotated chunk K)

- chunk_0 (new_offset=0): re-rotation shift=0 → K 무변경. Layer 0 K bitwise to vanilla [0..1] (kproj out_dim=1024 → shape-invariant kernel + position 0 RoPE 동일).
- chunk_1 / chunk_2: shift>0 → R(shift) 적용. fp32 RoPE composition noise floor 영향 가능 (atol 1e-6 수준).

## 8. Key Findings

1. **RoPE re-rotation 정확성**: group property는 atol 1e-6 내 만족. fp32 bitwise 보장 ❌이 사전 가정이었으나 (R(a+b) direct vs R(a)·R(b) composition), MacBook smoke에서 실측 max ~5.7e-7로 noise floor 위 atol gate로 정정.
2. **ChunkedKVStore reorder concat**: `new_offset` 순 concat이 bitwise. Step 3.1 (no-reorder)의 자연스러운 확장.
3. **Multi-chunk drift (4.3)**: max_abs=8.46, mean=0.99 — CacheBlend literature의 100% reuse w/o recompute 기대치. argmax는 유지 (next token 의미 보존), top-5는 일부 변경.
4. **argmax_match=True의 의미**: decode 1 token만 보면 prediction 동일. 하지만 logits 분포는 다름. multi-token decode·F1 metric에서는 영향 큼.
5. **mechanism**: cross-attention 누락 (chunk i의 layer 0은 chunk j의 K/V를 보지 못함) → layer 1+ K/V 누적 drift. RoPE re-rotation은 position 정렬만 해결, cross-attention 정보 ❌.

## 9. Interpretation / Mechanism

- 4.1 atol noise floor (~5.7e-7): cos/sin trig identity의 fp32 한계. R(a)·R(b) = R(a+b) 가 fp32에서 bitwise 보장 안 됨 (mechanism). atol gate가 적절.
- 4.3 drift: chunk 독립 prefill → layer 1 input은 layer 0 self-attention만 본 결과. vanilla의 layer 1 input은 full sequence self-attention. → layer 1+ K/V가 본질적으로 다름. recompute 없이 회복 불가.
- 4.3 argmax_match=True는 noise floor 위쪽 영역에서 logits의 ranking이 유지된 것. 일반화 ❌ — 다른 prompt에서는 argmax도 갈릴 수 있음.

## 10. Limitations and Non-goals

1. **100% chunk reuse without recompute**: drift 큼이 mechanism적 기대치. recompute 없이는 회복 불가.
2. **chunk_T = 2 고정**: 본 smoke는 chunk_T=2 × 3 chunks = 6 토큰. bucket size · chunk_size 일반화는 Step 5+에서.
3. **prompt 1개만**: "The capital of France is" 단일 prompt. F1 일반화는 Step 8.
4. **4.1 bitwise 불가능 사전 가정 정정**: 사전 task spec에서 4.1 bitwise 가능성 명시했으나 실측으로 atol 1e-6 정정.
5. **chunk padding 정책 명시 검증 ❌**: caller (검증 스크립트)가 정책 준수. Step 4 invariant로 ❌. 별도 helper로 분리 안 함.
6. **4.3 drift_budget 1e-2 loose**: regression monitoring 용 임계값, F1 영향 정량 아님.
7. **fork 무수정 원칙 유지**: `src/compblend/modeling/` 무변경.
8. **GQA repeat 저장 ❌ 유지**: K/V는 `H_kv=8` 상태로 저장.
9. **B=1 가정**: multi-batch 미검증.
10. **단일 환경 측정**: A100-SXM4-80GB / CUDA 12.8 / fp32 / eager. 다른 환경에서 4.1 atol·4.3 drift 재측정 필요.

## 11. Implications for Step 5·6·7

- **Step 5 (1 chunk reuse)**: single chunk + roundtrip — Step 3.3B의 N=1 특수 case 확장. 4.1 RoPE re-rotation은 shift=0이면 trivial.
- **Step 6 (N chunks reuse + recompute_ratio=1.0)**: 100% recompute로 4.3 drift 회복 (= vanilla). bitwise 또는 atol PASS 가능.
- **Step 7 (HKVD oracle)**: selective recompute의 알고리즘 정확성. partial recompute가 drift를 얼마나 회복하는지가 핵심.

## 12. Artifacts / Commits

- `src/compblend/rope_rotation.py`, `src/compblend/cache.py` (Step 3 재사용)
- `tasks/step_04_multi_chunk_concat.md`, `tasks/step_04_multi_chunk_concat/run_multi_chunk_concat_check.py`
- `results/step_04/macbook/summary.json`, `results/step_04/vastai/summary.json`
- `docs/reports/step_04_multi_chunk_concat_report.md` (본 보고서)
- commits: `60f6172` (task + code + smoke), `7df0271` (vast.ai results)
