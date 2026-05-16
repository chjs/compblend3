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

### Invariant 2.3A — Padded forward vs single forward의 DynamicCache K/V[:6] bitwise (옵션 B, 2026-05-16 결정)

**채택 경위**: C-3·C-4·C-6·C-7 진단 round 누적 결과(cuBLAS shape-dependent kernel dispatch가 `q_proj` (out_dim=4096) GEMM의 동일 input row에 대한 reduction 순서를 M에 따라 다르게 만든다)를 spec에 반영. 이전 정의(split prefill+decode = single full forward)는 cross-shape mechanism으로 bitwise 불가능(C-7 F1 max_abs=2.384e-06 재현) → 옵션 B로 교체.

**padded path** (단일 forward call, use_cache=True, M_full=7):
- `input_ids = concat(prompt(6 tokens), [next_token_id])` shape `(1, 7)`. 위치 6은 masked dummy slot (tokenizer-level PAD token이 아니다 — 정당화 단락 참조).
- `attention_mask = [1, 1, 1, 1, 1, 1, 0]` — 위치 6은 mask=0으로 격리.
- `use_cache=True` → DynamicCache 자동 생성 후 채워짐 (32 layer × K/V shape `(1, 8, 7, 128)`).

**single path** (단일 forward call, use_cache=True, M_full=7):
- `input_ids = concat(prompt(6 tokens), [next_token_id])` shape `(1, 7)` — padded path와 동일 input.
- `attention_mask = [1]*7` — 모두 visible.
- `use_cache=True` → DynamicCache 자동 생성 (32 layer × K/V shape `(1, 8, 7, 128)`).

**비교 (게이트: bitwise via `torch.equal`, atol fallback ❌)**:
```python
for layer_idx in range(num_hidden_layers):
    padded_k = padded.past_key_values.key_cache[layer_idx]   # (1, 8, 7, 128)
    single_k = single.past_key_values.key_cache[layer_idx]   # (1, 8, 7, 128)
    assert torch.equal(padded_k[:, :, :6, :], single_k[:, :, :6, :])
    # V도 동일하게
```

**Gate**: `torch.equal` (bitwise). atol fallback ❌. **forward hook ❌, `output_attentions=True` ❌** — DynamicCache의 `past_key_values[i].key_cache/value_cache`를 직접 접근.

**Diagnostics (참고용, gate ❌)**: layer별 SHA-256, max_abs_diff, mean_abs_diff, mismatched_layers.

**주의**:
- padded path의 DynamicCache 길이는 7. 검증 대상은 real position `[:6]`만.
- position 6의 masked dummy slot K/V는 cache에 들어가나 비교 대상 ❌.
- 향후 CacheBlend chunk cache artifact 저장 시 masked dummy slot K/V는 반드시 버릴 것.

**Mechanism justification (옵션 B가 bitwise 가능한 이유)**:
- padded와 single 둘 다 단일 forward이다. 즉 prefill cache + padded decode의 2단계 구조가 아니다.
- 양쪽 모두 DynamicCache가 empty 상태에서 시작하고, 한 번의 forward로 7개 K/V를 append한다.
- 따라서 attention problem shape는 양쪽 모두 `(Q_len=7, K_len=7)`이다. 이는 prefill cache 이후 padded decode를 수행했을 때 발생하는 `(Q_len=7, K_len=13)` 구조와 다르다.
- q/k/v/o/MLP의 GEMM shape도 양쪽 모두 M=7로 동일하므로, 이전 C-7/F2에서 확인한 shape-dependent dispatch 차이를 제거한 same-shape 비교가 된다.
- padded path의 `attention_mask=[1,1,1,1,1,1,0]`에서 position 6은 masked dummy slot이다. 그러나 비교 대상은 positions `[:6]`이다.
- positions 0~5는 causal mask 때문에 position 6을 볼 수 없다. 따라서 position 6이 masked dummy인지 real token인지의 차이는 positions 0~5의 attention output에 영향을 주지 않는다.
- layer 0에서는 embedding/input_layernorm/q_proj/k_proj/v_proj/RoPE의 positions `[:6]`이 padded와 single에서 bitwise 동일하므로 cache K/V[:6]도 bitwise 동일하다.
- 또한 positions `[:6]`의 attention output과 MLP output이 bitwise 동일하므로 layer 1 input`[:6]`도 bitwise 동일하다.
- 같은 논리가 layer-by-layer로 귀납적으로 적용되어, 모든 layer의 DynamicCache K/V[:6]가 bitwise 동일해야 한다.
- 단, padded path의 position 6 masked dummy slot K/V는 DynamicCache에 들어갈 수 있지만 비교 대상이 아니며, 향후 CacheBlend chunk artifact로 저장해서도 안 된다.

**Padded position 6에 `next_token_id`를 사용하는 이유 (정당화 단락, 2026-05-16)**:
2.3A에서는 padded path와 single path 모두 `input_ids=[prefix 6 tokens, next_token_id]`를 사용한다. padded path의 차이는 `attention_mask=[1,1,1,1,1,1,0]` 뿐이다. 따라서 position 6은 tokenizer-level PAD token이 아니라 masked dummy slot이다. 이는 의도된 설계다. `input_ids`, `position_ids`, RoPE 입력, GEMM shape를 양쪽에서 동일하게 유지하고 `attention_mask` 차이만 분리하기 위함이다. position 6의 K/V는 invariant 비교 대상이 아니며, 향후 CacheBlend chunk artifact로 저장해서도 안 된다.

대안(`tokenizer.pad_token_id` 또는 fallback으로 `eos_token_id`/`unk_token_id` 사용)을 채택하지 않은 근거:
- Mistral은 `pad_token_id=None`이라 fallback 체인 도입 시 `eos_token_id=2`를 쓰게 됨.
- padded.input_ids[6]=2 vs single.input_ids[6]=5465 ("Paris") → GEMM input row 6이 양쪽에서 달라짐.
- 그러면 cuBLAS GEMM의 **row-independence at bit level** (output row i가 input row i에만 의존) 가정이 추가됨. 이 가정은 C-7에서 직접 검증 ❌.
- 현재 설계(`next_token_id` 양쪽 동일)는 C-7 F2가 검증한 정확한 조건의 자연스러운 확장이라 가정 추가 없음. 비교 실패 시 원인 진단이 단순 (변수가 `attention_mask` 하나로 isolated).

**직전 작업 0의 K_total mismatch 분석 정정**: 직전 작업 0의 답변(§2)에서 "padded split decode + cache → attention K_total=13"으로 분석하여 옵션 B가 bitwise 불가능이라 결론냈으나, 그건 `prefill(6) + padded decode(7)` **2단계 구조**에 대한 것이었다. 옵션 B는 **단일 padded forward(7) + use_cache=True** 구조로 DynamicCache가 empty에서 시작하여 7개 K/V만 append하는 same-shape 구조이다. 이전 분석은 옵션 B에 적용 ❌.

### Invariant 2.3B — Operational split forward drift (measurement only, gate ❌)

**operational split path** (운영 그대로, M=1 decode):
1. prefill: `model(prompt_6, use_cache=True)` → 6 K/V 캐시
2. decode: `model([next_token_id], past_key_values=prefill_cache, use_cache=True)` → 새 토큰 logits (M=1 decode input + cached 6 → attention K_total=7)
3. `operational_logits = decode.logits[:, -1, :]` shape `(1, vocab)`

**reference**: 2.3A의 single path (length 7, use_cache=True)의 `logits[:, 6, :]`.

**비교 (게이트 ❌, 측정만)**:
- `max_abs_diff(operational_logits, single.logits[:, 6, :])`
- `mean_abs_diff`
- `argmax_match`: `argmax(operational) == argmax(single.logits[6])`
- top-k overlap (k=5): top-5 토큰 ID 교집합 크기
- `drift_budget_exceeded`: `max_abs_diff > 1e-4` 시 true (optional regression warning, gate ❌)

**의미**: cuBLAS shape-dependent dispatch (C-3·C-4·C-6·C-7 결과)는 운영 split decode (M=1 input + cached 6, attention shape `(Q=1, K=7)`)와 single (M=7 input, attention shape `(Q=7, K=7)`)이 다른 cuBLAS kernel을 호출할 가능성을 만든다. 이 drift는 mechanism적으로 회피 불가능이며 게이트가 아니다. 측정값은 regression monitoring·운영 noise floor 추정 용도.

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

# Tokenize
enc = tokenizer(PROMPT, return_tensors="pt")
ids = enc["input_ids"].to("cuda")            # (1, 6)
attn = enc["attention_mask"].to("cuda")      # (1, 6) all-ones

# (a) no-cache forward — 2.1·2.2 reference
set_all_seeds(42)
with torch.inference_mode():
    out_nocache = model(input_ids=ids, attention_mask=attn,
                         use_cache=False, output_hidden_states=True)

# (b) cache forward — 2.1·2.2 variant (DynamicCache 자동 생성)
set_all_seeds(42)
with torch.inference_mode():
    out_cache = model(input_ids=ids, attention_mask=attn,
                       use_cache=True, output_hidden_states=True)

# next_token_id (greedy from (b) prompt logits)
next_token_id = out_cache.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)

# ids_full = [prompt(6) + next_token(1)]
ids_full = torch.cat([ids, next_token_id], dim=1)                         # (1, 7)
attn_padded = torch.cat([attn, torch.zeros_like(next_token_id)], dim=1)   # [1,1,1,1,1,1,0]
attn_single = torch.ones_like(ids_full)                                   # [1]*7

# (c) padded path — 2.3A (use_cache=True, mask[6]=0)
set_all_seeds(42)
with torch.inference_mode():
    out_padded = model(input_ids=ids_full, attention_mask=attn_padded,
                        use_cache=True)
# out_padded.past_key_values: DynamicCache, layer당 K/V shape (1, 8, 7, 128)

# (d) single path — 2.3A reference + 2.3B reference (use_cache=True)
set_all_seeds(42)
with torch.inference_mode():
    out_single = model(input_ids=ids_full, attention_mask=attn_single,
                        use_cache=True)

# (e) operational split — 2.3B (prefill + M=1 decode)
set_all_seeds(42)
with torch.inference_mode():
    out_prefill = model(input_ids=ids, attention_mask=attn,
                         use_cache=True)
    # decode: 명시 attention_mask + cache_position (option E 패턴 유지)
    decode_attn_mask = torch.ones((1, 7), dtype=torch.long, device="cuda")
    decode_cache_position = torch.arange(6, 7, device="cuda")
    out_decode = model(input_ids=next_token_id,
                       past_key_values=out_prefill.past_key_values,
                       attention_mask=decode_attn_mask,
                       cache_position=decode_cache_position,
                       use_cache=True)
# operational_logits = out_decode.logits[:, -1, :]  shape (1, vocab)
# single reference     = out_single.logits[:, 6, :] shape (1, vocab)

# --- 2.3A 비교 ---
for i in range(num_hidden_layers):
    pk = out_padded.past_key_values.key_cache[i]   # (1, 8, 7, 128)
    sk = out_single.past_key_values.key_cache[i]   # (1, 8, 7, 128)
    assert torch.equal(pk[:, :, :6, :], sk[:, :, :6, :])
    # V도 동일하게
```

### Tensor shape 명세 (CLAUDE.md §6.1)

| 변수 | shape | 비고 |
|---|---|---|
| `ids` | `(1, 6)` | prompt input_ids (BOS 포함) |
| `attn` | `(1, 6)` | prompt attention_mask all-ones |
| `next_token_id` | `(1, 1)` | greedy argmax of (b) prompt logits |
| `ids_full` | `(1, 7)` | concat(ids, next_token_id) — padded·single 공통 input |
| `attn_padded` | `(1, 7)` | `[1, 1, 1, 1, 1, 1, 0]` — 2.3A padded path |
| `attn_single` | `(1, 7)` | `[1]*7` — 2.3A single path |
| `decode_attn_mask` | `(1, 7)` | `[1]*7` — 2.3B decode (cached 6 + new 1 시점) |
| `decode_cache_position` | `(1,)` | `[6]` — 2.3B decode 위치 |
| `out_nocache.logits` / `out_cache.logits` | `(1, 6, 32000)` | 2.1 비교 대상 |
| `out_nocache.hidden_states[i]` / `out_cache.hidden_states[i]` | `(1, 6, 4096)` | 33개, 2.2 비교 대상 |
| `out_padded.past_key_values.key_cache[i]` | `(1, 8, 7, 128)` | GQA: H_kv=8, D=128. 2.3A 비교 대상 (`[:6]`만) |
| `out_single.past_key_values.key_cache[i]` | `(1, 8, 7, 128)` | 동일 |
| `out_single.logits` | `(1, 7, 32000)` | 2.3B reference (`[:, 6, :]`) |
| `out_prefill.past_key_values.key_cache[i]` | `(1, 8, 6, 128)` | 2.3B prefill cache |
| `out_decode.logits` | `(1, 1, 32000)` | 2.3B operational decode |

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
    "2.3A_padded_cache_kv_equiv": {
      "passed": true,
      "gate": "torch.equal",
      "per_layer": [
        {"layer": 0,
         "k_match": true, "v_match": true,
         "k_sha_padded": "...", "k_sha_single": "...",
         "v_sha_padded": "...", "v_sha_single": "...",
         "k_max_abs_diff": 0.0, "k_mean_abs_diff": 0.0,
         "v_max_abs_diff": 0.0, "v_mean_abs_diff": 0.0}
      ],
      "mismatched_layers": []
    },
    "2.3B_operational_split_drift": {
      "measured": true,
      "max_abs_diff": 0.0,
      "mean_abs_diff": 0.0,
      "argmax_match": true,
      "topk_overlap_k5": 5,
      "drift_budget_exceeded": false,
      "drift_budget_threshold": 1e-4,
      "note": "measurement only, no pass/fail gate. cuBLAS shape-dependent dispatch (C-3/C-4/C-6/C-7)."
    }
  },
  "all_invariants_passed": true
}
```

`all_invariants_passed = 2.1 PASS AND 2.2 PASS AND 2.3A PASS`. **2.3B는 측정만이라 게이트 제외**.

## 보고서 작성 가이드

`docs/reports/step_02_dynamic_cache_report.md` 작성 (Markdown, `docs/design/report_style.md`). 섹션:
1. **요약** — invariant 4종 (2.1·2.2·2.3A: PASS/FAIL, 2.3B: 측정값)
2. **목표·통과 기준** — 옵션 B 채택 경위 (C-3/C-4/C-6/C-7 mechanism)
3. **환경** — vast.ai instance id, dph, driver, transformers 4.51.3. 누적 진단 round 비용 ($)
4. **수정 / 신규 파일** — `run_dynamic_cache_check.py` 신규, fork 무수정 유지
5. **Tensor shape 명세** — 표 위와 동일
6. **구현 핵심** — fork 무수정, DynamicCache 외부 호출. 5 forward(a no-cache / b cache / c padded / d single / e operational split). `forward hook ❌`, `output_attentions=True ❌`. K/V 직접 접근 (`past_key_values[i].key_cache/value_cache`). padded·single 둘 다 use_cache=True
7. **Invariant 결과 표** — 2.3A per-layer K/V match (table), 2.3B drift 측정값
8. **결과 데이터** — summary.json 발췌
9. **환경 간 비교** — Step 1과 동일 정책 (vast.ai 단독)
10. **알려진 한계** — 다음 항목 명시:
    - 직전 작업 0의 K_total mismatch 분석은 다른 구조(prefill cache + padded decode 2단계)에 대한 것. 옵션 B(단일 padded forward + use_cache=True, cache empty 시작)에는 적용 ❌ — 정정 사실 명시.
    - 2.3A는 K/V projection 단계 bitwise. attention 내부 GEMM은 DynamicCache append-only로 padded(cache + new) vs single(cache) shape 강제 차이 → logits까지 bitwise는 mechanism적으로 불가능, Step 2 검증 범위 밖. 2.3B로 정량화.
    - DynamicCache 버전 의존 (4.51.3 기준). v4.56 이후 API 변경 시 재검증.
11. **다음 step** — Step 3 (ChunkedKVStore 자료구조 정확성, 현재 stub — 진입 전 확장 필요)

## 다음 step 게이트

- [ ] `results/step_02/vastai/summary.json` `all_invariants_passed: true` (2.1 + 2.2 + 2.3A 모두 PASS, 2.3B 제외)
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
- **옵션 B 채택 경위 (2026-05-16)**: C-3~C-7 진단으로 cross-shape cuBLAS dispatch가 mechanism으로 확정. 이전 2.3A 명세(split prefill+decode = single full forward)는 bitwise 불가능. atol 완화는 사용자 전제("연산 순서·정밀도 통제 시 A=B")와 충돌하므로 spec을 둘로 분리:
  - 2.3A = padded vs single cache K/V[:6] bitwise (same-shape 구성)
  - 2.3B = 운영 split drift 측정 (gate ❌)
- **2.3A의 검증 범위 한계**: K/V projection 단계는 bitwise 가능하지만, attention 내부 GEMM은 DynamicCache append-only 구조상 padded(cache + new) vs single(cache)이 강제 shape 차이를 갖는 영역으로 들어가면 bitwise 불가. 그러나 옵션 B는 **단일 forward call** 구조라 이 한계와 무관 — Mechanism justification 참조.
- **실패 시 진단 순서**:
  1. `src/compblend/modeling/FORK_HASH.txt`로 fork byte hash 확인 (Step 1처럼) — fork 무변조 검증.
  2. padded·single의 layer별 K/V `[:6]`를 단계별 비교 — 어느 layer에서 갈라지는지 식별.
  3. DynamicCache 내부 상태 (`cache.update` 후 stored K/V)를 직접 추출해 padded·single 양쪽의 해당 위치 K/V와 element-wise 비교. position 6의 masked dummy slot K/V는 비교 대상 ❌.
  4. padded path의 attention_mask shape·cache_position 처리·position 6 masked dummy 영향 검증 (causal mask만으로도 0~5의 attention에는 영향이 없어야 함).
  5. **fork 코드 수정으로 통과시키기 ❌** — 왜 다른지 먼저 규명.
