# Step 3 — ChunkedKVStore 자료구조 정확성 + HF Cache 인터페이스 호환성 보고서

## 1. Summary

`ChunkedKVStore` 자료구조와 `transformers.DynamicCache` 인스턴스 사이의 양방향 변환이 **layer별 K/V를 bitwise 보존**하고, 변환된 DynamicCache가 모델 forward에 전달되어 원본 cache와 **logits SHA-256이 동일**함을 확인했다. **invariant 3.1·3.2·3.3A·3.3B 모두 PASS** ✅, **`step_03_final_gate_passed: true`** (vast.ai), `all_invariants_passed: true`.

핵심 측정값 (vast.ai `summary.json` 출처):
- 3.3B `logits_a_sha256 == logits_b_sha256 = e581d7f715cffb6377e63d8f19d97af3ba1b1e2a9ed2817e1730539694172c18`
- 3.3B `max_abs_diff = 0.0`, `mean_abs_diff = 0.0`
- decode token `5465 = "Paris"` (Step 0/1/2 일치)
- 3.3B failure case 자동 판정: `""` (none) — Case 1·2·3 모두 배제

> *Step 2's 2.3B drift budget is not applied in Step 3. All Step 3 gates are bitwise because the operations are intended to be tensor-preserving storage/materialization operations.*

## 2. Goal and Scope

검증 대상 4가지 (DECISIONS §3.8 KV Cache Data Model + CLAUDE.md §4.5 (d) 작업 0 결과 기반):

1. **roundtrip 정확성**: `DynamicCache → ChunkedKVStore → DynamicCache'` 변환이 layer별 K/V bitwise (3.1).
2. **메타 정확성**: `ChunkMeta` 7-field dataclass equality (3.2).
3. **HF Cache 인터페이스 호환성**: 변환된 DynamicCache 인스턴스의 `update`/`get_seq_length`/`get_max_cache_shape`/`isinstance(Cache)` (3.3A).
4. **모델 forward 일치**: 동일 prefix를 두 path로 forward 시 logits SHA-256 일치 (3.3B).

> *Step 3 does not perform RoPE re-rotation, chunk reordering, CacheBlend selection, partial recomputation, or chunk padding. Chunk padding is deferred to Step 4 작업 0.*

## 3. Environment

### 3.1 vast.ai (3.1·3.2·3.3A·3.3B)

| 항목 | 값 |
|---|---|
| 실험 환경 | vast.ai A100-SXM4-80GB (instance `36936503`, 검증 후 destroy) |
| GPU / driver | NVIDIA A100-SXM4-80GB / 580.126.20 |
| CUDA (PyTorch built) | 12.8 |
| Python / PyTorch / transformers | 3.10 / 2.10.0+cu128 / 4.51.3 |
| 모델 | `mistralai/Mistral-7B-Instruct-v0.2` |
| dtype / attention | `torch.float32` / `eager` |
| 결정론 설정 | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `benchmark=False`, `cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`, `float32_matmul_precision="highest"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| running 가동 시간 | 322초 (≈ 5.4분, `start_date` 1779020726.6 → destroy 1779021048.2) |
| setup 추가 시간 | 약 2분 (allocate → SSH ready + `install_vastai.sh`) |
| dph_total (show 기준) | $1.205555/h |
| **Step 3 본 round 추정 비용** | **~$0.15** (setup 포함) |
| 잔존 인스턴스 (destroy 후) | 0 |

### 3.2 MacBook (3.1·3.2·3.3A only, model-less CPU smoke)

| 항목 | 값 |
|---|---|
| `env_tag` | `macbook` |
| `model_check_enabled` | False |
| `local_smoke_gate_passed` | True |
| `step_03_final_gate_passed` | False (3.3B skipped — model check requires GPU) |
| dummy K/V | `make_dummy_dynamic_cache` (random fp32, deterministic seed 42) |

### 3.3 누적 비용 (Step 0~3, 추정치)

| Step | 추정 비용 |
|---|---|
| Step 0 (결정론 검증) | ~$0.05 |
| Step 1 (fork 동치성) | ~$0.16 |
| Step 2 (DynamicCache + 옵션 B + 진단 C-3~C-7) | ~$1.0 |
| **Step 3 (ChunkedKVStore + 3.3B)** | **~$0.15** |
| **누적 (Step 0~3)** | **~$1.36** |

추정치, 정확 청구액은 vast.ai 콘솔 기준 확인 필요.

## 4. Implementation Overview

### 4.1 모듈 구조 (신규 / 수정)

| 파일 | 변경 | 사유 |
|---|---|---|
| `src/compblend/cache.py` | **신규** | `ChunkMeta` 7-field dataclass + `ChunkedKVStore` dataclass + `from_dynamic_cache` / `to_dynamic_cache` 클래스 메서드 |
| `tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py` | **신규** | 5-step 검증 스크립트. `--enable-model-check` flag로 3.3B 분기 (vast.ai 전용) |
| `tasks/step_03_chunked_kv_store.md` | stub → 자체완결 12 § 확장 | task 명세 + invariant 정의 + tensor shape + gate 표 |
| `results/step_03/macbook/summary.json` | 신규 | MacBook smoke 결과 |
| `results/step_03/vastai/summary.json` | 신규 | vast.ai 3.3B 결과 |
| `PROGRESS.md` | 갱신 | Step 3 진입·결과 반영 |
| `src/compblend/modeling/` | **무수정** (Step 1·2 원칙 유지) | fork byte 무수정 |

### 4.2 해석 A — Cache 상속 ❌

> *ChunkedKVStore itself is not a transformers Cache. Step 3 adopts interpretation A: model.forward receives the DynamicCache materialized by ChunkedKVStore.to_dynamic_cache(). Therefore the HF Cache interface compatibility invariant is checked on the materialized DynamicCache, not on ChunkedKVStore itself.*

근거: 작업 0에서 modeling_mistral.py가 호출하는 Cache API surface 4종 (`update`, `get_seq_length`, `get_max_cache_shape`, `isinstance(Cache)`) 을 확인. 이들은 `DynamicCache`가 모두 만족하므로 `ChunkedKVStore`에 별도 구현 부담 ❌. dataclass container로 두고 `to_dynamic_cache()`가 표준 `DynamicCache` 인스턴스를 반환하는 것이 가장 단순한 설계.

### 4.3 5-step 검증 절차

| step | 동작 | 출처 |
|---|---|---|
| (1) | `make_dummy_dynamic_cache` — random K/V로 채워진 `DynamicCache` 32 layer × `(1, 8, 6, 128)` | CPU sanity / 3.1·3.2·3.3A 입력 |
| (2) | `build_default_chunk_spec(6)` — 2-token chunks 3개 (`chunk_0/1/2`). `new_offset == original_offset` | 3.1·3.2·3.3A 입력 |
| (3) | `from_dynamic_cache → to_dynamic_cache` + `check_3_1_roundtrip / check_3_2_meta / check_3_3A_interface` | 3.1·3.2·3.3A 검증 |
| (4) | `check_3_3B_model_forward` (`--enable-model-check` 활성 시): 2 prefill + roundtrip + 2 decode. read-only K/V 비교 + decode logits 비교 | 3.3B (vast.ai 전용) |
| (5) | gate 계산 + summary.json 저장 + 콘솔 출력 |

### 4.4 Tensor shape

| 변수 | shape | 비고 |
|---|---|---|
| `dc.key_cache[i]` / `dc.value_cache[i]` | `(B=1, H_kv=8, T, D=128)` | DynamicCache 표준 |
| `store.kv[chunk_id][layer_i].K` / `.V` | `(H_kv=8, T_chunk, D=128)` | B 차원 제거 (DECISIONS §3.8) |
| `decode.logits[:, -1, :]` | `(1, 32000)` | 3.3B 비교 대상 |
| `decode_attn` | `(1, 7)` | 운영 split decode 패턴 (Step 2 option E 재사용) |
| `decode_cache_position` | `(1,)` = `[6]` | 동상 |

## 5. Invariants and Gates

### 5.1 Invariant 정의 (task §4)

| ID | 명제 | gate |
|---|---|---|
| **3.1** | `D₀ → ChunkedKVStore → D₁` roundtrip 시 layer별 `torch.equal(K)`·`torch.equal(V)` + seq_len 일치 | `torch.equal` bitwise |
| **3.2** | `asdict(store.chunks[cm.chunk_id]) == asdict(cm)` for all 7 fields | dataclass equality |
| **3.3A** | `isinstance(D₁, DynamicCache/Cache)` + `get_seq_length`·`get_max_cache_shape` 일치 + `update()` shape 확장 정상 | 5 조건 AND |
| **3.3B** | 동일 prefix 두 path forward 시 `sha256(logits_A) == sha256(logits_B)` | SHA-256 일치 |

### 5.2 Gate 구조 (hygiene round에서 명확화)

| gate field | 의미 |
|---|---|
| `local_smoke_gate_passed` | 3.1 AND 3.2 AND 3.3A 모두 PASS (model-less, CPU·GPU 무관) |
| `step_03_final_gate_passed` | local_smoke_gate AND 3.3B PASS. 3.3B skipped 시 false |
| `all_invariants_passed` | `step_03_final_gate_passed`와 동일 (final 의미로 통일) |

## 6. Local MacBook Smoke Result

| invariant | passed | 주요 측정값 |
|---|---|---|
| 3.1 roundtrip K/V bitwise | ✅ | 32 layer × K/V `torch.equal`, `mismatched_layers=[]`, `seq_len 6/6 match=True` |
| 3.2 ChunkMeta equality | ✅ | 3 chunks, 7 fields each, `mismatched_chunk_ids=[]` |
| 3.3A Cache interface compat | ✅ | `isinstance(DynamicCache)=True`, `isinstance(Cache)=True`, `seq_length 6/6 match`, `max_cache_shape None/None match`, `update_seq_grew_by_1: 6→7=True` |
| 3.3B | ⏭️ skipped | reason: `model check 미활성 (--enable-model-check 필요, vast.ai 전용)` |
| `local_smoke_gate_passed` | True | |
| `step_03_final_gate_passed` | False | 3.3B skipped |
| 콘솔 | `==> local smoke gate PASS; Step 3 final gate pending (3.3B skipped)` | |

`results/step_03/macbook/summary.json`.

## 7. vast.ai 3.3B Model-Backed Result

### 7.1 Gate 결과

| gate field | 값 |
|---|---|
| `local_smoke_gate_passed` | True |
| **`step_03_final_gate_passed`** | **True** ✅ |
| `all_invariants_passed` | True |
| 콘솔 | `==> Step 3 final gate PASS` |

### 7.2 invariant 결과

| invariant | passed | 주요 측정값 |
|---|---|---|
| 3.1 roundtrip K/V bitwise | ✅ | 32 layer × K/V `torch.equal`, `mismatched_layers=[]`, `seq_len 6/6 match=True` |
| 3.2 ChunkMeta equality | ✅ | 3 chunks × 7 fields, `mismatched_chunk_ids=[]` |
| 3.3A Cache interface compat | ✅ | 5 조건 모두 True (`update_seq 6→7`) |
| **3.3B** model forward logits | ✅ | (아래 §7.3) |

### 7.3 3.3B diagnostic fields (PASS·FAIL 모두 채움)

| field | 값 |
|---|---|
| `passed` | True |
| `logits_a_sha256` | `e581d7f715cffb6377e63d8f19d97af3ba1b1e2a9ed2817e1730539694172c18` |
| `logits_b_sha256` | `e581d7f715cffb6377e63d8f19d97af3ba1b1e2a9ed2817e1730539694172c18` |
| `logits_sha_match` | True |
| **`max_abs_diff`** | **0.0** |
| **`mean_abs_diff`** | **0.0** |
| `decode_token_id` → `decode_token_decoded` | 5465 → `"Paris"` |
| `prefix_length` / `n_chunks` | 6 / 3 |
| `prefill_a_vs_orig.bitwise` / `mismatched_layers` | True / `[]` |
| `orig_vs_round.bitwise` / `mismatched_layers` | True / `[]` |
| `seq_len_match_all` (a/orig/round) | True (6/6/6) |
| `attention_mask_shape_decode` | `[1, 7]` |
| `cache_position_decode` | `[6]` |
| **`failure_case`** | `""` (none) |
| env (`torch` / `transformers` / `cuda`) | 2.10.0+cu128 / 4.51.3 / 12.8 |
| `cuda_device_name` | NVIDIA A100-SXM4-80GB |
| `model_dtype` / `attention_implementation` | `torch.float32` / `eager` |

### 7.4 Failure case 분류 (PASS이지만 진단 추적성 기록)

`_classify_3_3b_failure(diag)` 자동 판정:

| Case | 정의 | 본 결과 |
|---|---|---|
| Case 1 | `prefill_a_vs_orig.bitwise == False` (동일 prefix 반복 prefill 결정성 실패) | **배제** (`prefill_a_vs_orig.bitwise = True`) |
| Case 2 | `orig_vs_round.bitwise == False` (ChunkedKVStore 저장/복원 손실) | **배제** (`orig_vs_round.bitwise = True`) |
| Case 3 | K/V 모두 bitwise 이나 decode logits 불일치 (forward 인자/인터페이스) | **배제** (`logits_sha_match = True`) |
| 결과 | `failure_case = ""` | PASS 경로 |

## 8. Key Findings

1. **Roundtrip 정확성 확정**: 32 layer × K/V 모두 `torch.equal`. ChunkedKVStore의 slice + concat 변환은 fp32 K/V tensor에 대해 bitwise 보존.
2. **ChunkMeta 7-field 보존**: 입력 `chunk_spec`의 모든 필드가 dataclass equality로 일치. 자동 추론 없이 호출자가 채운 메타가 그대로 유지됨.
3. **HF Cache 인터페이스 surface 충족**: `to_dynamic_cache()`의 반환값이 표준 `DynamicCache` 인스턴스라 `isinstance`, `get_seq_length`, `get_max_cache_shape`, `update` 모두 자연스럽게 동작.
4. **모델 forward에서 bitwise 동등**: vast.ai A100 환경에서 동일 prefix를 두 경로 — (a) 원본 prefill의 `past_key_values` 직접 사용, (b) roundtrip된 `DynamicCache` 사용 — 으로 decode 시 logits SHA-256 100% 일치. `max_abs_diff = 0.0`.
5. **decode 결과 일관성**: decode token `5465 = "Paris"` — Step 0·1·2의 동일 prompt 결과와 일치.
6. **3.3B failure case 자동 배제**: Case 1/2/3 모두 진단 fields로 명시적 배제. PASS이나 향후 다른 환경에서 FAIL 시 root cause 위치를 즉시 분리할 수 있는 인프라 확보.

> *Step 2's 2.3B drift budget is not applied in Step 3. All Step 3 gates are bitwise because the operations are intended to be tensor-preserving storage/materialization operations.*

## 9. Interpretation / Mechanism

### 9.1 왜 3.1이 bitwise인가

`from_dynamic_cache`는 `dc.key_cache[i][0, :, start:end, :].detach().clone()` 으로 chunk를 slicing. `to_dynamic_cache`는 `torch.cat(parts, dim=-2).unsqueeze(0)` 으로 재구성. 둘 다 **새 메모리 할당** + **요소별 복사** (tensor 값 변경 ❌). 입력 row 순서가 `new_offset == original_offset` 이면 출력 메모리 layout이 입력과 동일 → `torch.equal` 보장.

### 9.2 왜 3.3B가 bitwise인가

3.3B 절차에서 두 path의 forward 입력이 다음과 같이 동일:

| 단계 | path A | path B | 동등성 |
|---|---|---|---|
| prefill | `model(prompt, use_cache=True)` → `D_A` | `model(prompt, use_cache=True)` → `D_orig` → roundtrip → `D_round` | `prefill_a_vs_orig.bitwise=True` (결정론) + `orig_vs_round.bitwise=True` (3.1) → `D_A ≡ D_round` |
| decode 입력 | `next_token_id`, `D_A`, mask=[1]*7, cache_position=[6] | `next_token_id`, `D_round`, mask=[1]*7, cache_position=[6] | 모든 인자 동일 |
| 따라서 logits | `logits_A` | `logits_B` | SHA-256 동일 |

3.1 + 결정론적 prefill → 3.3B는 **derived guarantee**. mechanism적으로 cross-shape cuBLAS dispatch 위험 ❌ (Step 2 2.3B와 달리 동일 shape forward).

### 9.3 `_classify_3_3b_failure` 자동 진단 (PASS 시에도 안전망 작동)

PASS 경로에서도 Case 1/2/3 sanity check (read-only K/V 비교)를 수행했다. 만약 3.3B FAIL 발생 시 진단 round 없이 즉시 layer 어느 단계 — prefill 결정성 / ChunkedKVStore 저장 손실 / forward 인자 mismatch — 인지를 명시할 수 있다. Step 2 진단 round (C-3~C-7) 5번을 단축할 가능성.

## 10. Limitations and Non-goals

1. **B=1 DynamicCache 만 지원/검증**: 본 phase 가정 (DECISIONS §3.8). multi-batch K/V는 미검증.
2. **DynamicCache 전용**: `StaticCache` / `QuantizedCache` / `OffloadedCache` 미지원 (DECISIONS §3.8). v4.56+ API 변경 시 재검증.
3. **Cache 인터페이스는 materialized DynamicCache에서만 검증**: `ChunkedKVStore`를 `model.forward`에 직접 전달 ❌. `to_dynamic_cache()` 호출 후 반환된 DynamicCache 인스턴스 전달.
4. **RoPE re-rotation 미적용**: Step 4.1로 이연. Step 3 ChunkedKVStore는 raw K/V만 저장.
5. **chunk reordering semantics 미검증**: 본 smoke는 `new_offset == original_offset`. `to_dynamic_cache()`가 `new_offset` 순 정렬을 지원하나, `new_offset != original_offset` 시의 모델 forward 의미는 Step 4+ blend logic + RoPE re-rotation 결합 후 정의됨.
6. **Partial recomputation 미수행**: HKVD selective recompute는 Step 7 범위.
7. **CacheBlend selection logic 미수행**: Step 4+ 범위.
8. **Chunk padding 정책 미검증**: DECISIONS §13 v13 사전 가정 (right-padding · bucket size · PAD K/V 저장 ❌) 의 검증은 Step 4 작업 0.
9. **GQA repeat 저장 ❌ 정책 유지**: KV cache는 `H_kv=8` 상태로 저장, attention 직전에만 `repeat_kv` (DECISIONS §3.8 메모리 정책).
10. **`ChunkMeta.token_ids` equality는 metadata equality**: 입력 chunk_spec metadata 보존 검증이지, 자동 tokenizer-derived chunking correctness 검증이 아니다. 호출자가 채운 7 필드가 그대로 보존되는지만 본다.
11. **`new_offset` duplicate / overlap / gap semantics 미정의**: `to_dynamic_cache()`는 `(new_offset, chunk_id)` tie-break으로 결정론 정렬만 보장. 중복·중첩·간격이 있는 chunk_spec 의 의미와 거부 정책은 Step 4+ blend logic에서 명시 validation이 필요할 수 있음.

> *Step 3 does not perform RoPE re-rotation, chunk reordering, CacheBlend selection, partial recomputation, or chunk padding. Chunk padding is deferred to Step 4 작업 0.*

## 11. Implications for Step 4

- Step 4 (N chunks 따로 prefill → concat = vanilla) 진입 시 ChunkedKVStore가 raw K/V 저장 토대로 사용 가능. roundtrip + interface compat 검증이 완료된 자료구조.
- **Step 4 작업 0에서 chunk padding 정책 검증 (right-padding · bucket size · PAD K/V 저장 ❌)** — DECISIONS §13 v13 사전 가정 확정 round.
- Step 4.1 = RoPE re-rotation 도입. `to_dynamic_cache()` 가 `new_offset` 순 정렬 후 RoPE 재적용을 통합하는 방향 (또는 별도 함수 분리) 은 Step 4.1 task 파일에서 결정.
- 3.3B 자동 진단 (Case 1/2/3) 인프라는 Step 4+ 의 forward 검증 round에서 재사용 가능 — `_compare_dc_kv` / `_classify_3_3b_failure` 패턴.
- chunk reordering (`new_offset != original_offset`) 의 모델 forward 의미는 Step 4.1 RoPE re-rotation 통합 후 별도 invariant로 검증.

## 12. Artifacts / Commits / Next Actions

### 12.1 Artifacts

| 경로 | 내용 |
|---|---|
| `src/compblend/cache.py` | `ChunkMeta` + `ChunkedKVStore` (해석 A) |
| `tasks/step_03_chunked_kv_store.md` | task 자체완결 spec (12 §) |
| `tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py` | 5-step 검증 + hygiene field/gate 분리 + 3.3B diagnostics |
| `results/step_03/macbook/summary.json` | MacBook smoke (3.1·3.2·3.3A PASS) |
| `results/step_03/vastai/summary.json` | vast.ai (3.1·3.2·3.3A·3.3B PASS) |
| `docs/reports/step_03_chunked_kv_store_report.md` | 본 보고서 |

### 12.2 Commits (Step 3 브랜치)

| hash | 내용 |
|---|---|
| `e2f9c00` | task 확장 + 본 코드 + MacBook smoke (3.3A까지 PASS, 3.3B는 vast.ai 별도 round) |
| `72d1b5b` | hygiene — gate 분리 + 3.3B diagnostics |
| `e9449f0` | vast.ai 3.3B 결과 (max_abs=0.0, failure_case="") |
| (이번 round) | 보고서 + PROGRESS.md 갱신 |

### 12.3 Next Actions

Step 3 final gate PASS 완료. 다음 round:

1. **사용자 리뷰** — 본 보고서 검토 (`docs/reports/step_03_chunked_kv_store_report.md`).
2. 승인 시: `git checkout main && git merge --no-ff step/step_03_chunked_kv_store` → `git tag step_03_done` → `git push origin main step_03_done` → step 브랜치 삭제 (로컬+원격) → Step 4 진입.
3. **Step 4 진입 — 작업 0 (CacheBlend chunk padding 정책 검증)**: DECISIONS §13 v13 사전 가정 확정 round. right-padding · bucket size · PAD K/V 저장 ❌.

### 12.4 충돌 사실 (source of truth 재확인 결과)

직전 round의 보고 (vast.ai 실행 직후) 와 본 보고서의 모든 수치 — `logits_a/b_sha256`, `max_abs_diff`, `mean_abs_diff`, `decode_token_id`, `prefill_a_vs_orig.bitwise`, `orig_vs_round.bitwise`, `failure_case` — 가 `results/step_03/vastai/summary.json` 직접 재확인 시 **완전 일치**. 충돌 없음.
