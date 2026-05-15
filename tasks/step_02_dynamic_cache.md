# Step 2 — HF DynamicCache forward = no-cache forward (bitwise 동등)

> Self-contained task. 이 파일만 읽고도 작업 가능해야 한다.
> 2026-05-15: stub → 본격 task 파일로 확장. CacheBlend 로직 진입 전 마지막
> "fork 무수정 + 외부 객체(DynamicCache) 호출 경로 동등성" 검증 단계.

---

## 목표

`use_cache=True` (DynamicCache 자동 생성/사용) forward가 `use_cache=False` (no cache) forward와 같은 logits·layer hidden state를 만드는지 검증. 추가로 **split forward (prefill + decode)** 가 **single forward (N+1 토큰 한 번에)** 와 동등한지도 검증. Step 4+에서 우리가 만들 ChunkedKVStore가 DynamicCache 인터페이스로 변환되어 forward에 전달될 것이므로, DynamicCache 자체가 신뢰할 수 있는 기준임을 확인한다.

## 사전 확인 결과 (작업 0 — transformers 4.51.3 외부 코드 검증)

CLAUDE.md §4.5 (d) 적용. 명세 작성 전 외부 코드를 직접 확인했다.

- **DynamicCache**: `transformers.cache_utils.DynamicCache` — 공개 API. `__init__(self, _distributed_cache_data=None)` → `DynamicCache()`로 빈 인스턴스. 공개 메서드 14개 (`update`, `get_seq_length`, `to_legacy_cache` 등).
- **MistralModel.forward**: `past_key_values: Optional[Cache] = None`. **transformers 4.51.3은 `Cache` 서브클래스 또는 `None`만 허용** — legacy Tuple of Tensors는 받지 ❌ (fork line 489-490이 `isinstance(past_key_values, (type(None), Cache))` 검사, 주석 "TODO: remove in v4.56"이 남았으나 이미 거부 로직만 작동). 비교 surface는 "DynamicCache(use_cache=True) vs None(use_cache=False)"으로 축소됨.
- **use_cache=True 출력**: 인스턴스 미전달 시 자동 `DynamicCache()` 생성 (fork line 495-496), `out.past_key_values`에 그대로 반환 (`BaseModelOutputWithPast`).
- **fork의 cache 관련 import** (Step 1 import 12줄 변환에 포함됨): `Cache`, `DynamicCache`, `SlidingWindowCache`, `StaticCache`, `BaseModelOutputWithPast` 전부 사용 가능. fork 수정 없이 Step 2 진행 가능.

## Step 2 원칙

- **`src/compblend/modeling/`의 fork 코드는 import 문 외 byte 무수정 유지** (Step 1 원칙 그대로).
- DynamicCache는 외부 객체로 forward 호출 시 전달 또는 `use_cache=True`로 자동 생성. fork 코드 안에 cache 관련 수정 ❌.
- 검증 계측은 외부 forward hook·dict 캡처만 — `tasks/step_02_dynamic_cache/run_dynamic_cache_check.py` 안에서. fork 디렉토리에 추가 ❌.

## 사전 조건

- Step 0/1 통과 (tag `step_00_done`, `step_01_done` — main에 merge됨)
- Branch: `step/step_02_dynamic_cache` (main에서 분기). **첫 commit으로 보류 (a) `scripts/vast_helper.py` push `-u` 가드** 처리(별도 round 합의).
- vast.ai 인스턴스: 작업 중 `scripts/vast_helper.py`로 할당 (disk_space>=100 필터·destroy 대화형 패치 적용본)
- 모델 캐시: Mistral-7B-Instruct-v0.2
- 환경: fp32, `attn_implementation="eager"`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`

## 통과 기준 / Invariants

### Invariant 2.1 — DynamicCache forward = no-cache forward (logits)
같은 `input_ids` (prompt = `"The capital of France is"`, 6 tokens)에 대해 `use_cache=True` (DynamicCache 자동 생성·사용) forward와 `use_cache=False` forward의 logits SHA-256 동일.
```
sha256(model(ids, use_cache=True).logits) == sha256(model(ids, use_cache=False).logits)
```

### Invariant 2.2 — Layer-by-layer hidden state 동등성
두 forward의 모든 layer hidden state SHA-256 동일.
```
∀ i ∈ [0, num_hidden_layers]: sha256(h_i^cache) == sha256(h_i^nocache)
```
(`output_hidden_states=True`로 33개(embedding + 32 layer) 추출.)

### Invariant 2.3A — Split forward (prefill + decode) = single forward (bitwise 1차 → atol 1e-6 fallback)

**split path** (use_cache=True, 2 단계):
1. prompt 6 토큰 → `use_cache=True` forward → DynamicCache 채워짐 (32 layer × K/V).
2. last_token argmax(`logits[:, -1, :]`)로 next_token id 1개 추출 (greedy. Step 0에서 5465="Paris" 확인됨).
3. 그 cache + `[next_token]` 1개로 추가 forward (decode step). → split_path 최종 logits.

**single path** (use_cache=False, 1 단계):
1. `concat(prompt, [next_token])` (7 토큰)을 `use_cache=False`로 forward 1회 → single_path logits.
2. 마지막 토큰 logits만 추출 (`logits[:, -1, :]`).

**비교 (3-tier)**:
- **Tier 1 (bitwise)**: `sha256(split_path_logits) == sha256(single_path_logits)` → PASS.
- **Tier 2 (atol fallback)**: bitwise 실패 시 `torch.allclose(..., atol=1e-6)` → PASS (fallback 발동 시 보고서·summary.json에 명시).
- **Tier 3 (FAIL)**: atol도 실패 → 진단 후 사용자 보고.

bitwise 보장 단정 ❌ — split path의 K/V concat 경로 차이로 fp32/eager/deterministic에서도 미세 차이 가능성 (§솔직성 노트 참조).

## 구현 사양

### 입력
- `prompt = "The capital of France is"` → tokenize → `ids = (1, 6)` (BOS 포함)
- 모델: `mistralai/Mistral-7B-Instruct-v0.2`, fp32, `attn_implementation="eager"`
- seed=42; `set_all_seeds(42)` 호출은 **각 forward 직전** (Step 0/1과 동일 패턴)
- `setup_deterministic()`은 main 시작 시 1회

### 호출 방식 (의사 코드)

```python
from compblend.modeling import MistralForCausalLM
model = MistralForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32,
                                            attn_implementation="eager").to("cuda").eval()

# (a) no-cache forward — 2.1·2.2의 reference
set_all_seeds(42)
with torch.no_grad():
    out_nocache = model(input_ids=ids, use_cache=False, output_hidden_states=True)

# (b) cache forward — 2.1·2.2의 변형 path (DynamicCache 자동 생성)
set_all_seeds(42)
with torch.no_grad():
    out_cache = model(input_ids=ids, use_cache=True, output_hidden_states=True)
# out_cache.past_key_values: DynamicCache 인스턴스

# (c) split path (2.3A): prefill 후 decode
set_all_seeds(42)
with torch.no_grad():
    out_prefill = model(input_ids=ids, use_cache=True)
    next_token_id = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
    out_decode = model(input_ids=next_token_id,
                       past_key_values=out_prefill.past_key_values,
                       use_cache=True)
# split_logits = out_decode.logits[:, -1, :]  shape (1, vocab)

# (d) single path (2.3A): N+1 토큰 한 번에
ids_full = torch.cat([ids, next_token_id], dim=1)  # (1, 7)
set_all_seeds(42)
with torch.no_grad():
    out_single = model(input_ids=ids_full, use_cache=False)
# single_logits = out_single.logits[:, -1, :]  shape (1, vocab)
```

### Tensor shape 명세 (CLAUDE.md §6.1)

| 변수 | shape | 비고 |
|---|---|---|
| `ids` | `(1, 6)` | prompt input_ids (BOS 포함) |
| `next_token_id` | `(1, 1)` | greedy argmax of out_prefill last-token logits |
| `ids_full` | `(1, 7)` | concat(ids, next_token_id) |
| `out_*.logits` | `(1, T, 32000)` | T=6 (prefill), T=1 (decode), T=7 (single) |
| `out_*.hidden_states[i]` | `(1, T, 4096)` | 33개 |
| `out_*.past_key_values` | `DynamicCache` | layer당 K/V shape `(1, 8, T, 128)` (GQA: H_kv=8, D=128) |

### 결정론 / 새 파일

- `setup_deterministic()`, `set_all_seeds(42)` 매 forward 직전 — Step 0/1 패턴.
- `tasks/step_02_dynamic_cache/run_dynamic_cache_check.py` (신규)
- `src/compblend/modeling/`은 Step 1 fork 그대로 — 수정 ❌.

## 실행 방법

```bash
# MacBook에서 코드 작성 → commit → push (step 브랜치)
# vast.ai에서:
cd compblend3 && git fetch origin && git checkout step/step_02_dynamic_cache && git pull
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tasks/step_02_dynamic_cache/run_dynamic_cache_check.py --out results/step_02/vastai/
# 결과 git push → MacBook git pull로 회수
```

local_a100 교차검증(invariant 0.3 계열)은 미수행 — vast.ai 단독. 사용자 결정 시 별도 round.

## 결과 저장 형식

`results/step_02/vastai/summary.json`:

```json
{
  "step": 2,
  "env_tag": "vastai",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "torch_dtype": "float32",
  "attention_implementation": "eager",
  "transformers_version": "4.51.3",
  "prompt": "The capital of France is",
  "decode_token_id": 5465,
  "decode_token_decoded": "Paris",
  "invariants": {
    "2.1_dynamic_cache_logits_equiv": {
      "passed": true,
      "cache_logits_sha256": "...",
      "nocache_logits_sha256": "..."
    },
    "2.2_per_layer_hidden_equiv": {
      "passed": true,
      "n_hidden_states": 33,
      "mismatched_layers": []
    },
    "2.3A_split_vs_single_forward": {
      "passed": true,
      "comparison_tier": "bitwise",
      "split_logits_sha256": "...",
      "single_logits_sha256": "...",
      "max_abs_diff": 0.0,
      "atol_threshold": 1e-6,
      "fallback_used": false
    }
  },
  "all_invariants_passed": true
}
```

`comparison_tier`: `"bitwise"` | `"atol_1e-6_fallback"` | `"FAIL"`. `fallback_used: true`면 보고서 §10에 명시.

## 보고서 작성 가이드

`docs/reports/step_02_dynamic_cache_report.md` 작성 (Markdown, `docs/design/report_style.md`). 섹션:
1. **요약** — invariant 3종 PASS/FAIL, 2.3A의 `comparison_tier` 명시
2. **목표·통과 기준**
3. **환경** — vast.ai instance id, dph, driver, transformers 4.51.3
4. **수정 / 신규 파일** — `run_dynamic_cache_check.py` 신규, fork 무수정 유지
5. **Tensor shape 명세** — 표 위와 동일
6. **구현 핵심** — fork 무수정, DynamicCache 외부 호출, 4 forward(no-cache, cache, prefill, decode, single), forward hook ❌ (이번 step은 logits/hidden_states/cache만)
7. **Invariant 결과 표** (2.3A의 tier·max_abs_diff 명시)
8. **결과 데이터** — summary.json 발췌
9. **환경 간 비교** — Step 1과 동일 정책 (vast.ai 단독)
10. **알려진 한계** — DynamicCache 버전 의존, 2.3A bitwise/fallback 정책, fork 외부 모듈 의존
11. **다음 step** — Step 3 (ChunkedKVStore 자료구조 정확성, 현재 stub — 진입 전 확장 필요)

## 다음 step 게이트

- [ ] `results/step_02/vastai/summary.json` `all_invariants_passed: true` (2.3A가 bitwise 또는 atol fallback이면 PASS)
- [ ] `docs/reports/step_02_dynamic_cache_report.md` 작성
- [ ] 사용자 리뷰 승인
- 통과 시 §7.1대로 `main` `--no-ff` merge + tag `step_02_done` + 브랜치 삭제 → Step 3 진입

## 작업 순서

1. 이 확장본을 `main`에 `[meta]` commit (§7.1 예외 (a), §7.2 `[meta]` prefix). push.
2. `git checkout -b step/step_02_dynamic_cache` (main에서 분기).
3. **첫 commit**: 보류 (a) `scripts/vast_helper.py` push `-u` 가드 (별도 round 합의 — DECISIONS.md §13 v11).
4. `tasks/step_02_dynamic_cache/run_dynamic_cache_check.py` 작성. py_compile + smoke test (MacBook).
5. commit + push → vast.ai 인스턴스 할당·셋업 → 실행 → 결과 회수 → 인스턴스 destroy.
6. 보고서·PROGRESS 갱신 → commit → 사용자 리뷰 대기.

## 솔직성 노트

- **trivially 통과의 의미**: 2.1·2.2가 통과해도 "구현이 맞다"가 아니라 **"DynamicCache 호출 경로가 no-cache 경로와 동일한 logits/hidden을 만든다"**만 의미. 단일 forward에서는 `cache.update`가 STORE만 하고 attention 계산엔 끼지 않으므로 trivially 통과 예상.
- **DynamicCache 버전 의존**: DynamicCache는 transformers 내부 객체라 버전 의존 강함 — 이번 검증은 4.51.3 기준. v4.56 이후 API 변경 가능 (legacy tuple 거부 로직도 v4.56에서 제거 예정 — 주석 명시). 그 시점에 재검증.
- **2.3A bitwise vs atol 정책**: split path의 K/V concatenation 순서가 single path 내부 처리와 미세하게 다를 가능성 있음 (transformers 내부 padding·concat 경로 차이). 1차 기준 bitwise → 실패 시 atol 1e-6 fallback → 모두 실패면 사용자 보고. **fp32/eager/deterministic에서도 bitwise 보장 단정 ❌** — 정직한 명세 선호.
- **실패 시 진단 순서**:
  1. `src/compblend/modeling/FORK_HASH.txt`로 fork byte hash 확인 (Step 1처럼) — fork 무변조 검증.
  2. split path와 single path의 layer별 K/V 텐서를 단계별 비교 — 어디 layer·어디 위치에서 갈라지는지 식별.
  3. DynamicCache 내부 상태 (`cache.update` 후 stored K/V)를 직접 추출해 single path의 해당 위치 K/V와 element-wise 비교.
  4. **fork 코드 수정으로 통과시키기 ❌** — 왜 다른지 먼저 규명.
