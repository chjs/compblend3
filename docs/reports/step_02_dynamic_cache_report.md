# Step 2 — DynamicCache 신뢰성 검증 보고서 (옵션 B: padded cache K/V equivalence)

## 1. 요약

HF transformers 4.51.3의 `DynamicCache`가 same-shape 조건에서 logits·layer hidden state·KV cache를 **bitwise 보존**함을 검증했다. **invariant 2.1·2.2·2.3A 모두 PASS** ✅, `all_invariants_passed: true`.

초기 명세(`split prefill + decode = single full forward, bitwise`)는 C-3~C-7 누적 진단으로 **cuBLAS GEMM의 shape-dependent kernel dispatch가 mechanism적으로 cross-shape bitwise를 막는다는 사실**이 확정됐다 (atol 완화 ❌, mechanism 미해소). 이에 사용자 결정으로 옵션 B 채택 — 명세를 둘로 분리:

- **2.3A** = padded single-call prefill cache K/V[:6] vs single prefill cache K/V[:6] (`torch.equal` bitwise gate). same-shape (M=7 양쪽, DynamicCache empty 시작) 구성으로 bitwise 가능.
- **2.3B** = 운영 split forward (prefill 6 + M=1 decode)와 single full forward(M=7) logits 비교의 drift measurement만. gate ❌.

vast.ai A100-SXM4-80GB(instance `36876915`)에서 2.3A는 32 layer × K·V 모두 `torch.equal` 통과. 2.3B drift는 `max_abs_diff = 6.20e-06`, `argmax_match=True`, `top5_overlap=5/5`, `drift_budget_exceeded=False` (1e-4 threshold) — 운영 경로의 numerical noise floor 정량화이며 token 예측에 영향 ❌.

## 2. Background / Motivation

Step 2 첫 실행에서 초기 2.3A 명세(split prefill+decode 의 last-token logits SHA-256 == single full forward last-token logits SHA-256)가 **bitwise FAIL** (`max_abs_diff = 6.199e-06`). atol 1e-6 fallback도 미달. 5 라운드 진단으로 원인 좁히기:

| 라운드 | 가설 | 결과 |
|---|---|---|
| 초기 옵션 E (`attention_mask`·`cache_position` 명시 전달) | HF auto-inference 의존 제거 | drift 동일 — 원인 ❌ |
| C-3 (prefill vs cached decode position 비교) | tokenization·position 오류 | first prefix KV drift = **layer 1**, case C 확정. token/position 정상 |
| C-4 (Layer 0 intra-op divergence localization) | layer 0 어느 단계? | 첫 발산 = `02_q_proj` (out_dim=4096, `max_abs=2.384e-06`). `k_proj`/`v_proj` (out_dim=1024) bitwise |
| C-6 (4 조건 명시 검증) | 입력·eager·정밀도·deterministic 누락? | A·B·C·D 모두 PASS + q_proj 발산 재현 (2.384e-06) |
| C-7 (padded shape + position) | "동일 순서" 미보장 + position 정확성 | E1·E2 (position·RoPE) bitwise 통과 + **F2 `split_padded` (M=7) vs `single` (M=7) q_proj first 6 `max_abs=0` bitwise** 확정. F1 (M=6 vs M=7) `max_abs=2.384e-06`. **GEMM input shape (M)이 동일 순서를 결정** |

→ 원인은 우리 코드 버그 ❌, fork 무수정 위배 ❌, position info 오류 ❌, fp32/TF32/deterministic 미설정 ❌. 모두 정상. **확정 mechanism**: cuBLAS GEMM의 `q_proj` (Linear(4096→4096))에서 input shape `(1, M, 4096)`이 M=6 vs M=7로 다르면 다른 내부 kernel/reduction 순서를 선택해 같은 input row에 대해서도 출력의 bit-level이 달라진다. `k_proj`/`v_proj` (out_dim=1024)는 M=6/7 모두 같은 kernel 선택 → bitwise.

이 사실은 NVIDIA cuBLAS 공식 문서가 보장하는 "same problem configuration only" 결정성 범위와 일치 (C-5 작업 0 외부 문서 조사). atol 완화는 사용자 전제("연산 순서·정밀도 통제 시 A=B")와 충돌. 사용자 결정 (2026-05-16)으로 명세를 둘로 분리하여 bitwise gate 보존 + drift 측정 분리 → **옵션 B 채택**. 본 step 의 진짜 목적이 *DynamicCache 자체의 신뢰성 검증*이라는 점으로 재정의.

## 3. Environment

| 항목 | 값 |
|---|---|
| 실험 환경 | vast.ai A100-SXM4-80GB (instance `36876915`, 검증 후 destroy) |
| GPU / driver | NVIDIA A100-SXM4-80GB / 570.211.01 |
| CUDA | 12.8 |
| Python / PyTorch / transformers | 3.10 / 2.10.0+cu128 / 4.51.3 |
| 모델 | mistralai/Mistral-7B-Instruct-v0.2 |
| dtype / attention | float32 / `eager` |
| 결정론 설정 | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `benchmark=False`, `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `float32_matmul_precision="highest"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| dph_total | 할당 시 `$1.073611/h` (offer 36289857), `show instance` `$1.205555/h` (저장소 등 추가 비용 추정) |
| running 시간 | 228초 (≈ 3.8분, `start_date`→destroy) |
| Step 2 본 실행 추정 비용 | ~$0.08~0.20 (셋업 포함) |
| **Step 2 누적 vast.ai 비용 (추정)** | **~$1.0** (옵션 A·E 초기 실행 + C-3 + C-4 + C-6 + C-7 + 본 실행). 추정치, 정확 청구액은 vast.ai 콘솔 기준 확인 필요 |

## 4. Method

### 4.1 수정 / 신규 파일

| 파일 | 변경 | 사유 |
|---|---|---|
| `tasks/step_02_dynamic_cache/run_dynamic_cache_check.py` | 신규 → 옵션 B 재작성 | 5 forward 구조, `torch.equal` 게이트, `past_key_values.key_cache/value_cache` 직접 접근 |
| `tasks/step_02_dynamic_cache.md` | 옵션 B 정의 + mechanism justification + masked dummy slot 정정 | 2.3A 재정의 + 2.3B 신설 + 직전 작업 0의 K_total 분석 정정 명시 |
| `scripts/diagnose_prefill_vs_cached_decode_position.py` | 신규 | C-3 진단 |
| `scripts/diagnose_layer0_intra_op_divergence.py` | 신규 | C-4 진단 |
| `scripts/diagnose_input_eager_precision_deterministic.py` | 신규 | C-6 진단 |
| `scripts/diagnose_padded_shape_position_info.py` | 신규 | C-7 진단 |
| `artifacts/c3_diagnosis/`, `artifacts/c4_layer0_intra_op/`, `artifacts/c6_input_eager_precision_deterministic/`, `artifacts/c7_padded_shape_position_info/` | 신규 | 진단 결과 데이터 |
| `results/step_02/vastai/summary.json` | 신규 | 본 실행 결과 |
| `DECISIONS.md` §13 v13 | 신규 entry | 옵션 B 결정 + CacheBlend chunk padding 사전 가정 |
| `PROGRESS.md` | 갱신 | Step 2 옵션 B 진입·완료 반영 |
| `src/compblend/modeling/` | **무수정** | Step 1 원칙 그대로 유지 |

### 4.2 5 forward 구조

| 라벨 | 입력 | use_cache | 용도 |
|---|---|---|---|
| (a) no-cache | `input_ids=(1,6)`, `attention_mask=[1]*6` | False, `output_hidden_states=True` | 2.1·2.2 reference |
| (b) cache | `input_ids=(1,6)`, `attention_mask=[1]*6` | True, `output_hidden_states=True` | 2.1·2.2 variant + `next_token_id` 계산 |
| (c) **padded** | `input_ids=(1,7)=[prompt + next_token_id]`, `attention_mask=[1,1,1,1,1,1,0]` | True | **2.3A padded path** |
| (d) **single** | `input_ids=(1,7)=[prompt + next_token_id]`, `attention_mask=[1]*7` | True | **2.3A single + 2.3B reference** |
| (e) operational split | (prefill: `(1,6)`, mask=`[1]*6`, cache=True) + (decode: `next_token_id (1,1)`, `past_key_values`, `attention_mask=[1]*7`, `cache_position=[6]`, cache=True) | True | 2.3B operational |

### 4.3 2.3A 비교 방식 (forward hook ❌, `output_attentions` ❌)

```python
for i in range(num_hidden_layers):           # 32 layers
    pk = padded.past_key_values.key_cache[i]      # (1, 8, 7, 128)
    sk = single.past_key_values.key_cache[i]      # (1, 8, 7, 128)
    k_match_i = torch.equal(pk[:, :, :6, :], sk[:, :, :6, :])
    # V도 동일 방식
```

- `DynamicCache.key_cache` / `.value_cache`는 Python `list[Tensor]`. `[i]` indexing 즉시 layer i tensor 반환 (4.51.3 transformers 4.51.3 검증 완료).
- 비교 대상은 **K/V[:6]** (real 위치). position 6의 masked dummy slot K/V는 cache에 들어가나 비교 대상 ❌.
- position 6은 `tokenizer.pad_token_id` (Mistral은 `None`)가 아니라 **masked dummy slot** — `input_ids[6] = next_token_id` (padded·single 동일) + `attention_mask[6] = 0`(padded만)으로 격리. 정당화: input row 6을 양쪽 동일하게 유지하면 GEMM input row 6 동일 → cuBLAS row-independence 가정 추가 불필요. 변수 isolation은 `attention_mask`만.

### 4.4 2.3B 비교 방식

```python
operational_logits = out_decode.logits[:, -1, :]          # M=1 decode last-token
single_reference   = out_single.logits[:, 6, :]           # 2.3A의 single 재사용
drift = (operational_logits - single_reference).abs()
```

측정값: `max_abs_diff`, `mean_abs_diff`, `argmax_match`, `topk_overlap_k5`, `drift_budget_exceeded (threshold=1e-4)`. **gate ❌**.

## 5. Results

### 5.1 Invariant 결과 (게이트)

| Invariant | 비교 | gate | 결과 |
|---|---|---|---|
| **2.1** | `sha256(cache.logits) == sha256(no-cache.logits)`, both M=6 | `==` | ✅ PASS |
| **2.2** | 33 layer hidden states SHA-256 일치 (cache vs no-cache, M=6) | mismatched_layers = `[]` | ✅ PASS |
| **2.3A** | 32 layers × (K, V) `torch.equal(padded[:,:,:6,:], single[:,:,:6,:])` | mismatched_layers = `[]` | ✅ **PASS** |

**`all_invariants_passed: true`** ✅

### 5.2 2.3B 측정값 (gate ❌)

| 측정 | 값 |
|---|---|
| `max_abs_diff` | **6.198883056640625e-06** |
| `mean_abs_diff` | 1.0394246601208579e-06 |
| `argmax_match` | True (양쪽 vocab id `28725`) |
| `operational_argmax` | 28725 |
| `single_argmax` | 28725 |
| `topk_overlap_k5` | **5/5** |
| `drift_budget_exceeded` | False (threshold 1e-4) |

### 5.3 디코드 토큰

- `decode_token_id`: **5465** → `"Paris"` (Step 0/1과 일치)
- next-after-Paris (2.3B의 양쪽 argmax): vocab id **28725**

### 5.4 Tensor shape 명세

| 변수 | shape | 비고 |
|---|---|---|
| `ids` | `(1, 6)` | prompt input_ids (BOS 포함) |
| `next_token_id` | `(1, 1)` | greedy argmax of (b) prompt last-token logits |
| `ids_full` | `(1, 7)` | padded·single 공통 input |
| `attn_padded` | `(1, 7)` | `[1, 1, 1, 1, 1, 1, 0]` |
| `attn_single` | `(1, 7)` | `[1]*7` |
| `out_padded.past_key_values.key_cache[i]` | `(1, 8, 7, 128)` | GQA: H_kv=8, D=128 |
| `out_single.past_key_values.key_cache[i]` | `(1, 8, 7, 128)` | 동일 |
| `out_single.logits` | `(1, 7, 32000)` | 2.3B reference `[:, 6, :]` |
| `out_prefill.past_key_values.key_cache[i]` | `(1, 8, 6, 128)` | 2.3B operational prefill cache |
| `out_decode.logits` | `(1, 1, 32000)` | 2.3B operational decode |

## 6. Mechanism Interpretation

옵션 B의 2.3A 가 bitwise PASS한 mechanism을 vast.ai 실측으로 확인했다. task §2.3A의 mechanism justification 단락이 그대로 성립:

- padded와 single 둘 다 **단일 forward call** + `use_cache=True`. prefill cache + padded decode 의 2단계 구조가 아님.
- DynamicCache가 양쪽 모두 **empty 상태에서 시작**, 한 번의 forward로 **7개 K/V를 append**.
- attention problem shape는 양쪽 모두 `(Q_len=7, K_len=7)`. (이는 prefill cache 후 padded decode를 수행했을 때 발생하는 `(Q_len=7, K_len=13)` 구조와 다름.)
- q/k/v/o/MLP의 GEMM shape도 양쪽 모두 M=7로 동일. C-7 F2에서 확인한 shape-dependent dispatch 차이를 제거한 same-shape 비교.
- `attention_mask = [1,1,1,1,1,1,0]`에서 position 6은 masked dummy slot. 비교 대상은 positions `[:6]`만.
- positions 0~5는 **causal mask** 때문에 position 6을 볼 수 없음. 따라서 position 6의 token이 masked dummy든 real이든 positions 0~5의 attention output에 영향 ❌.
- Layer 0에서 embedding·input_layernorm·q_proj·k_proj·v_proj·RoPE의 positions `[:6]`이 bitwise → cache K/V[:6] bitwise.
- positions `[:6]`의 attention output·MLP output이 bitwise → layer 1 input`[:6]` bitwise → 귀납적으로 모든 layer cache K/V[:6] bitwise.

직전 작업 0(2026-05-16 초기)의 "옵션 B는 attention K_total=13 vs 7 → bitwise 불가능" 분석은 **다른 구조** (`prefill(6) cache + padded decode(7)` 2단계)에 적용되는 것이었고, 옵션 B의 단일 forward 구조에는 적용 ❌. 본 결과로 그 정정이 실증됨.

## 7. Operational Drift (2.3B의 해석)

2.3B는 **실패가 아니라 운영 경로의 expected numerical noise floor의 정량화**다.

| path | Q_len | K_len | attention 내부 GEMM shape |
|---|---|---|---|
| 운영 split decode (cached 6 + M=1 new) | 1 | 7 | `(B, H_q, 1, 7)` |
| single full forward (M=7) | 7 | 7 | `(B, H_q, 7, 7)` |

두 path의 **q_proj 호출 shape도 다름** (split decode는 `(1, 1, 4096)`, single은 `(1, 7, 4096)`). cuBLAS shape-dependent dispatch가 적용되는 영역. 따라서 bitwise 요구 ❌ → drift measurement only.

| 단계 | 측정값 | 출처 |
|---|---|---|
| q_proj 첫 row 발산 (M=6 vs M=7, same input row) | 2.384e-06 | C-4 + C-7 F1·F3 |
| logits 누적 발산 (32 layer + lm_head) | **6.199e-06** | 본 step 2.3B |
| token 예측 일치 (argmax) | True | 본 step 2.3B |
| top-5 일치 | 5/5 | 본 step 2.3B |

mechanism 영향은 token 예측에 영향 ❌ (argmax/top-5 모두 단일 일치). drift_budget 1e-4 미달성 → regression monitoring 정상 범위.

## 8. Decision (옵션 B 채택 타당성)

| 항목 | 결과 |
|---|---|
| 옵션 B (2.3A = padded cache K/V[:6] bitwise) | ✅ vast.ai PASS 실측 확인 |
| 기존 명세 (`split-vs-single bitwise logits invariant`) | 폐기 — mechanism적으로 cross-shape bitwise 불가능 |
| 2.3A | **DynamicCache K/V equivalence bitwise gate** (게이트 유지) |
| 2.3B | drift measurement only, gate ❌ |
| atol 1e-6 fallback / atol 완화 | 채택 ❌ (사용자 전제 위배) |

옵션 B는 Step 2의 본 의도(DynamicCache의 신뢰성 검증)에 가장 직접 부합한다. cache 객체 자체의 동작이 정확하고, same-shape 조건 하에서 logits/hidden/KV가 bitwise 보존됨이 확인됐다.

## 9. CacheBlend Implication (Step 4+ 사전 메모)

DECISIONS.md §13 v13에 사전 가정으로 기록된 CacheBlend chunk padding 정책이 옵션 B의 mechanism과 정합:

| 정책 (Step 4 작업 0에서 검증 후 확정) | 옵션 B와의 정합 |
|---|---|
| (a) **right-padding** (left-padding ❌) | position/RoPE 어긋남 방지. 2.3A에서도 padded path는 right-padding 구조 |
| (b) **`padded_len = chunk_size` 또는 bucket size** | full prompt length padding은 성능 부적절 (CacheBlend 이점 상실). chunk_size/bucket 단위가 현실적 |
| (c) **저장 시 real token K/V만** (PAD K/V 버림, `attention_mask=0`) | 2.3A에서도 position 6 masked dummy slot K/V는 비교·저장 ❌ |
| (d) **chunk 간 GEMM path 동등성 검증을 Step 4 작업 0에서** | C-4/C-7이 확정한 GEMM shape-dependent dispatch의 직접적 귀결 — chunk 단위 정책을 별도 검증 round로 |

**용어 구분**: 2.3A의 "masked dummy slot"은 `tokenizer.pad_token_id`가 아니라 `input_ids` 동일 + `mask=0`으로 구성된 격리 위치. Step 4+ chunk padding의 "PAD/right-padding"은 chunk-level 의미로, 서로 다른 맥락이라 표현 분리.

## 10. Limitations / Honesty

- **2.3A의 검증 범위**: cache-empty single-call padded prefill 구조에서의 K/V[:6] bitwise. operational split decode (M=1)와 single full forward 의 bitwise equality를 증명한 것 ❌. operational drift는 2.3B 측정값으로만 정량화.
- **2.3B는 gate ❌**: drift measurement only. `drift_budget_exceeded` 는 optional regression warning이며 통과 조건 아님.
- **결과는 본 환경 단일 측정**: A100-SXM4-80GB / driver 570.211.01 / CUDA 12.8 / PyTorch 2.10.0+cu128 / transformers 4.51.3 / eager attention / fp32 / deterministic. 다른 GPU·CUDA·cuBLAS·PyTorch·attention backend에서는 drift budget 및 mechanism 발현 위치가 달라질 수 있음 — 재측정 필요.
- **누적 비용은 추정치**: ~$1.0은 6 라운드(옵션 A/E 초기 + C-3 + C-4 + C-6 + C-7 + 본 step 2 실행)의 추정 합. 정확 청구액은 vast.ai 콘솔 기준 확인 필요.
- **직전 작업 0의 K_total mismatch 분석 정정**: 2026-05-16 작업 0 초기 답변에서 "옵션 B는 attention K_total=13 vs 7 → bitwise 불가능"으로 결론냈으나, 그건 `prefill(6) cache + padded decode(7)` 2단계 구조에 적용. 옵션 B는 단일 padded forward(7) + use_cache=True 구조라 cache empty 시작 → attention shape `(7, 7)` 양쪽 동일. 본 보고서·task §2.3A·DECISIONS §13 v13에 정정 명시.
- **DynamicCache 버전 의존**: `transformers 4.51.3` 기준. v4.56+ API 변경 가능 (legacy Tuple 거부 로직 제거 예정 명시). 그 시점에 재검증.
- **fork 외부 의존**: `src/compblend/modeling/`는 transformers 외부 모듈 의존 (`Cache`, `DynamicCache`, `apply_rotary_pos_emb`, `eager_attention_forward` 등). transformers 버전 fix 필수.

## 11. Next Steps

1. **Step 2 보고서 사용자 리뷰** → 본 round.
2. 승인 시 PROGRESS.md 갱신 + §7.1대로 `main --no-ff` merge + tag `step_02_done` + 브랜치 삭제.
3. Step 3 진입 전 별도 결정 사항:
   - 2.3B `drift_budget` (1e-4) 를 향후 step의 regression 기준으로 사용할지
   - CacheBlend chunk padding 정책 (DECISIONS §13 v13 사전 가정) Step 4 작업 0 검증 일정
4. Step 3 (ChunkedKVStore 자료구조 정확성) — 현재 stub. 진입 전 자체완결 task 파일 확장 필요.
