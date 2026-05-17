# Step 3 — ChunkedKVStore 자료구조 정확성 + HF Cache 인터페이스 호환성

> Self-contained task. 이 파일만 읽고도 작업 가능해야 한다.
> 2026-05-17: stub → 자체완결 spec 확장. CacheBlend 알고리즘 (Step 4+) 진입 전,
> chunk별 K/V 저장·DynamicCache 변환 자료구조의 정확성과 인터페이스 호환성을
> 검증하는 단계.

---

## 1. 목표

`ChunkedKVStore` 자료구조가 다음을 보장하는지 검증:

1. **roundtrip 정확성**: `DynamicCache → ChunkedKVStore → DynamicCache'` 변환이 layer별 K/V를 bitwise 보존.
2. **메타 정확성**: `ChunkMeta`의 7 필드 (DECISIONS §3.8) 가 입력과 정확히 일치.
3. **HF Cache 인터페이스 호환성**: 변환된 DynamicCache 인스턴스가 `transformers.cache_utils.Cache` 인터페이스를 만족 (modeling_mistral.py가 호출하는 API surface 충족).
4. **모델 forward 일치**: 변환된 DynamicCache를 forward에 전달해도 원본 DynamicCache 사용 시와 logits SHA-256 일치.

Step 4+ (CacheBlend logic + RoPE re-rotation) 진입 전, 자료구조 토대 검증.

## 2. 배경 / 사전 확인 결과 (작업 0 — CLAUDE.md §4.5 (d))

명세 작성 전 외부 코드를 직접 확인했다.

### 2.1 HF Cache 인터페이스 (transformers 4.51.3)

`transformers.cache_utils.Cache` abstract methods (5종):
- `update(K, V, layer_idx, cache_kwargs=None) -> (K_full, V_full)`
- `get_seq_length(layer_idx=0) -> int`
- `get_usable_length(new_seq_length, layer_idx=0) -> int`
- `get_max_cache_shape() -> Optional[int]`
- `reorder_cache(beam_idx)` (beam search용 — 우리 phase 미사용)

`DynamicCache`의 추가 attribute: `key_cache: list[Tensor]`, `value_cache: list[Tensor]` (Step 2에서 검증).

### 2.2 modeling_mistral.py (fork)의 Cache API 호출 surface

| 호출 위치 | 메서드 | 필수 여부 |
|---|---|---|
| `MistralAttention.forward:175` | `past_key_value.update(K, V, layer_idx, cache_kwargs)` | 필수 |
| `MistralModel.forward:489-490` | `isinstance(past_key_values, (type(None), Cache))` | 필수 (타입 통과 위해) |
| `MistralModel.forward:495-496` | `if use_cache and past_key_values is None: DynamicCache()` | 자동 생성 (우리는 외부에서 주입) |
| `MistralModel.forward:499` | `past_key_values.get_seq_length()` | 필수 |
| `_update_causal_mask:591` | `past_key_values.get_seq_length()` | 필수 |
| `_update_causal_mask:592-593` | `isinstance(past_key_values, (StaticCache, SlidingWindowCache))` | branch (둘 다 False 이어야 우리 path) |
| `_update_causal_mask:615` | `past_key_values.get_max_cache_shape()` | 필수 |

→ ChunkedKVStore가 직접 `Cache`를 상속하지 않고 **`to_dynamic_cache()`로 DynamicCache 인스턴스를 반환**하는 방식(해석 A)을 채택하면, 위 API surface는 DynamicCache가 모두 만족하므로 별도 구현 부담 ❌.

### 2.3 DECISIONS.md §3.8 KV Cache Data Model (기존 명세)

- Chunk export shape: `(H_kv, T_chunk, D)` per layer per chunk. **B 차원 제거** (chunk별 분리).
- GQA repeat 적용 ❌ (메모리 절약, attention 직전에만).
- Phase 1~6에서는 **DynamicCache만 지원**. StaticCache/QuantizedCache/OffloadedCache 미지원.
- **ChunkMeta dataclass** (7 필드): `chunk_id`, `token_ids`, `original_offset`, `new_offset`, `original_length`, `is_cacheable`, `is_permanent_hit`.
- 변환 시점에 RoPE re-rotation 적용 → **Step 4.1**. Step 3 ChunkedKVStore는 raw K/V만 저장.

## 3. Step 3 원칙

- **`src/compblend/modeling/` fork 코드는 import 문 외 byte 무수정** (Step 1/2 원칙 그대로).
- `ChunkedKVStore`는 **신규 모듈 `src/compblend/cache.py`** 에 정의 (Step 4+ 에서도 그대로 사용).
- **해석 A 채택**: `transformers.cache_utils.Cache` 상속 ❌, `dataclass`-like container. 모델 forward에는 `to_dynamic_cache()`의 반환값 (DynamicCache 인스턴스) 전달.
- **RoPE re-rotation 적용 ❌** (Step 4.1로 이연). Step 3 ChunkedKVStore는 raw K/V만 저장·복원.
- **chunk_spec 형식**: `list[ChunkMeta]` — 호출자가 7 필드를 직접 채움. 자동 추론 ❌.
- 검증 계측은 외부 검증 스크립트 안에서만 (`tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py`). 추가 외부 hook ❌.

## 4. Invariant 정의

### 4.1 Invariant 3.1 — roundtrip K/V bitwise (model-less)

```
D₀ = (랜덤 또는 모델 prefill로 채워진) DynamicCache
D₁ = ChunkedKVStore.from_dynamic_cache(D₀, chunk_spec).to_dynamic_cache()

∀ layer i ∈ [0, num_layers):
    torch.equal(D₀.key_cache[i],   D₁.key_cache[i])   == True
    torch.equal(D₀.value_cache[i], D₁.value_cache[i]) == True
D₀.get_seq_length() == D₁.get_seq_length()
```

Gate: **`torch.equal` bitwise**. atol fallback ❌ (Step 3는 모든 비교가 same-shape, 2.3B drift_budget 미적용).

조건: `chunk_spec` 안의 모든 chunk의 `new_offset == original_offset` (재배열 ❌). 재배열 시나리오는 Step 4+ blend logic 책임.

### 4.2 Invariant 3.2 — ChunkMeta 7-field equality (model-less)

```
∀ cm ∈ chunk_spec:
    asdict(store.chunks[cm.chunk_id]) == asdict(cm)
```

Gate: dataclass equality on all 7 fields (`chunk_id`, `token_ids`, `original_offset`, `new_offset`, `original_length`, `is_cacheable`, `is_permanent_hit`).

### 4.3 Invariant 3.3A — Cache 인터페이스 호환성 (model-less)

```
D₁ = ChunkedKVStore.from_dynamic_cache(D₀, spec).to_dynamic_cache()

isinstance(D₁, DynamicCache) == True
isinstance(D₁, Cache) == True
D₁.get_seq_length()     == D₀.get_seq_length()
D₁.get_max_cache_shape() == D₀.get_max_cache_shape()

# dummy K/V update — layer 0 길이 +1 확인
seq_before = D₁.get_seq_length()
D₁.update(new_K, new_V, layer_idx=0)
D₁.get_seq_length() == seq_before + 1
```

Gate: 위 5개 조건 모두 True.

### 4.4 Invariant 3.3B — 모델 forward logits SHA-256 일치 (model-backed, vast.ai)

```
prefill_a: model(prompt, use_cache=True) → D_A
prefill_b: model(prompt, use_cache=True) → D_orig   (동일 결과 기대)
D_round = ChunkedKVStore.from_dynamic_cache(D_orig, spec).to_dynamic_cache()

logits_A = model.decode(next_token, past_key_values=D_A).logits[:, -1, :]
logits_B = model.decode(next_token, past_key_values=D_round).logits[:, -1, :]

sha256(logits_A) == sha256(logits_B)
```

Gate: `==`. 같은 prefix를 같은 모델로 forward한 두 cache가 같으면 같은 decode logits를 만들어야 함.

## 5. 구현 사양

### 5.1 신규 모듈 `src/compblend/cache.py`

```python
from dataclasses import dataclass
import torch
from transformers.cache_utils import DynamicCache

@dataclass
class ChunkMeta:
    chunk_id: str
    token_ids: list[int]
    original_offset: int
    new_offset: int
    original_length: int
    is_cacheable: bool
    is_permanent_hit: bool

@dataclass
class ChunkedKVStore:
    chunks: dict[str, ChunkMeta]
    kv: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]  # K, V shape: (H_kv, T_chunk, D)
    num_layers: int

    @classmethod
    def from_dynamic_cache(cls, dc: DynamicCache, chunk_spec: list[ChunkMeta]) -> "ChunkedKVStore": ...

    def to_dynamic_cache(self) -> DynamicCache: ...  # new_offset 순 정렬 후 concat
```

`to_dynamic_cache()`: `new_offset` 순 정렬 후 layer별 K/V concat (`torch.cat(dim=-2)`) → `unsqueeze(0)` 으로 B=1 복원 → 새 DynamicCache에 layer별 `update`.

### 5.2 검증 스크립트 `tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py`

CLI:
```
python tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py \
    --out results/step_03/<env>/ \
    [--enable-model-check] [--model <id>]
```

- `--enable-model-check` 없으면 3.1/3.2/3.3A만 (model-less, CPU). MacBook smoke 가능.
- `--enable-model-check` 있으면 3.3B 포함. CUDA + `CUBLAS_WORKSPACE_CONFIG=:4096:8` 필요. vast.ai 전용.

### 5.3 Tensor shape 명세 (CLAUDE.md §6.1)

| 변수 | shape | 비고 |
|---|---|---|
| `dc.key_cache[i]` | `(B=1, H_kv=8, T, D=128)` | DynamicCache 표준 |
| `dc.value_cache[i]` | `(B=1, H_kv=8, T, D=128)` | 동상 |
| `chunk_kv[chunk_id][layer_i].K` | `(H_kv=8, T_chunk, D=128)` | B 차원 제거 (DECISIONS §3.8) |
| `chunk_kv[chunk_id][layer_i].V` | `(H_kv=8, T_chunk, D=128)` | 동상 |
| `model.logits[:, -1, :]` | `(1, 32000)` | 3.3B 비교 대상 |

상수 (Mistral 7B v0.2): `H_kv=8, D=128, num_layers=32, B=1`.

### 5.4 chunk_spec 디폴트 (smoke 용)

`T_TOTAL=6` 토큰을 2-token chunks 3개로 분할 (`chunk_0`, `chunk_1`, `chunk_2`). `new_offset == original_offset` (재배열 ❌). `is_permanent_hit`: 첫 chunk만 True (system prompt 가정).

## 6. 검증 계측

- `forward hook ❌` — Step 2와 동일 원칙. 직접 `past_key_values.key_cache/value_cache` 접근.
- `output_attentions=True ❌`.
- 3.1: per-layer `torch.equal` + SHA-256 보조 기록.
- 3.2: `asdict(stored) == asdict(input)` dataclass equality.
- 3.3A: `isinstance` + getattr + dummy update 호출.
- 3.3B: 2개의 fresh prefill (D_A, D_orig) + roundtrip D_round + 2개의 decode forward (cache 독립). `sha256` 비교.

## 7. 실행 환경

| 환경 | invariants | 명령 |
|---|---|---|
| **MacBook** (Claude Code, CPU only) | 3.1 + 3.2 + 3.3A | `python tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py --out results/step_03/macbook/` |
| **vast.ai A100** | 3.1 + 3.2 + 3.3A + **3.3B** | `python tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py --out results/step_03/vastai/ --enable-model-check` |

vast.ai 인스턴스: step 진입 시 `scripts/vast_helper.py`로 자동 할당. 환경 변수: `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `COMPBLEND_ENV_TAG=vastai`, `HF_TOKEN`.

3.3A·3.3B는 동일 forward hook 사용 ❌. 결정론 setup (`setup_deterministic()`) 매 실행 시작 시 1회.

## 8. 결과 저장 형식

`results/step_03/<env>/summary.json` (env ∈ {macbook, vastai}):

```json
{
  "step": 3,
  "env_tag": "vastai",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "transformers_version": "4.51.3",
  "torch_version": "2.10.0+cu128",
  "model_check_enabled": true,
  "shapes": {"B": 1, "H_kv": 8, "D": 128, "num_layers": 32, "T_total": 6},
  "chunk_spec_n_chunks": 3,
  "invariants": {
    "3.1_roundtrip_bitwise": {
      "passed": true,
      "gate": "torch.equal",
      "n_layers": 32,
      "mismatched_layers": [],
      "seq_len_orig": 6, "seq_len_round": 6, "seq_len_match": true,
      "per_layer": [...]
    },
    "3.2_chunk_meta_equality": {
      "passed": true,
      "n_chunks_input": 3, "n_chunks_stored": 3,
      "mismatched_chunk_ids": [],
      "per_chunk": [...]
    },
    "3.3A_cache_interface_compat": {
      "passed": true,
      "checks": {...}
    },
    "3.3B_model_forward_logits_equiv": {
      "passed": true,
      "gate": "sha256(logits_a) == sha256(logits_b)",
      "logits_a_sha256": "...", "logits_b_sha256": "...",
      "max_abs_diff": 0.0,
      "decode_token_id": 5465,
      "decode_token_decoded": "Paris",
      "n_chunks": 3, "prefix_length": 6
    }
  },
  "all_invariants_passed": true
}
```

`all_invariants_passed = 3.1 AND 3.2 AND 3.3A AND (3.3B if model_check_enabled else True)`.

## 9. 통과 기준 / gate

| 단계 | 통과 invariant | gate |
|---|---|---|
| MacBook smoke (작업 3) | 3.1 + 3.2 + 3.3A | local PASS |
| step branch commit (작업 4) | 3.1 + 3.2 + 3.3A PASS | smoke summary.json 첨부 (`results/step_03/macbook/`) |
| vast.ai 실행 (별도 round) | 3.3B | `sha256(logits_a) == sha256(logits_b)` |
| step_03_done tag | 3.3B PASS 또는 명시적으로 deferred | `results/step_03/vastai/summary.json` `all_invariants_passed: true` |

→ **3.3B 생략한 채 `step_03_done` 부여 ❌**. 3.3B가 명시적으로 deferred 처리되거나 PASS 되어야 함.

## 10. 작업 순서

1. (완료) 작업 0 — modeling_mistral.py Cache API 호출 surface + DECISIONS §3.8 확인.
2. (완료) 작업 1 — task 파일 자체완결 확장 (본 파일).
3. (완료) 작업 2 — `src/compblend/cache.py` 신규 + `run_chunked_kv_store_check.py` 작성.
4. (완료) 작업 3 — MacBook py_compile + smoke (3.1 + 3.2 + 3.3A PASS).
5. (완료) 작업 4 — `step/step_03_chunked_kv_store` branch + commit + push.
6. **(별도 round)** 작업 5 — vast.ai 실행 (3.3B 검증). 결과 회수 → 인스턴스 destroy.
7. 보고서 작성 (`docs/reports/step_03_chunked_kv_store_report.md`) → 사용자 리뷰 → §7.1 merge.

## 11. 솔직성 노트

- **chunk_spec 의 `new_offset == original_offset` 제약 (Step 3 scope)**: 재배열 (`new_offset != original_offset`) 시나리오의 bitwise 동등성은 Step 4+ blend logic의 책임 영역. Step 3 자료구조는 재배열도 지원하지만 (`to_dynamic_cache()`가 `new_offset` 순 정렬), 그 결과의 모델 forward 의미는 RoPE re-rotation을 동반해야 정의됨.
- **3.3A의 update() mutates dc_round**: `check_3_3A_interface` 안에서 `dc_round.update(dummy_K, dummy_V, layer_idx=0)`을 호출하여 layer 0 길이가 +1 됨. 따라서 3.1/3.2 검증이 끝난 후에 호출해야 함 (스크립트가 순서 지킴). dc_round의 자료 일관성은 3.3A 이후 보장되지 않음.
- **3.3B의 cache 독립성**: D_A와 D_round는 각각 별도 prefill 결과를 사용해야 함 (DynamicCache가 forward 중 mutate 되므로 공유 ❌). 스크립트는 2번 prefill을 별도로 실행 (결정론 보장으로 양쪽 bitwise 일치 기대).
- **2.3B drift_budget (1e-4) 미적용**: Step 3 모든 invariant가 same-shape 비교 (자료구조 roundtrip · 모델 forward에 같은 모양 cache 전달). cross-shape cuBLAS dispatch 위험 ❌. drift_budget는 향후 cross-shape 비교가 등장하는 step에서 재검토.
- **HF Cache 인터페이스 surface 변경 위험 (transformers 버전 의존)**: 본 검증은 4.51.3 기준. v4.56+ 에서 legacy Tuple 거부 로직 제거 외에도 API 추가/변경 가능. 그 시점에 재검증.

## 12. 다음 step 예고

- **Step 4 — N chunks 따로 prefill → concat = vanilla** (chunk별 prefill 결과를 ChunkedKVStore로 모아 단일 forward 결과와 동등성 검증). Step 4.1 = RoPE re-rotation.
- **Step 4 작업 0에서 chunk padding 정책 검증** (right-padding · bucket size · PAD K/V 저장 ❌). DECISIONS §13 v13 사전 가정의 검증 일정.
- ChunkedKVStore에 재배열 (`new_offset != original_offset`) + RoPE re-rotation 통합 — Step 4.1 task 파일에서 구체화.
