# Step 1 — fork 동치성 검증 보고서 (fork된 코드 = HF 표준 forward)

## 1. 요약

HF transformers 4.51.3의 `modeling_mistral.py`를 `src/compblend/modeling/`로 fork(import 12줄만 상대→절대 변환, forward 본문 1089줄 byte 무수정)하고, fork된 코드의 forward가 HF 표준 forward와 **bitwise 동일**함을 확인했다. **invariant 1.1·1.2·1.3 모두 통과** ✅ — logits, 모든 layer hidden state, 모든 layer의 q/k/v projection 출력이 일치. 우리 코드 베이스 안에서 같은 결과를 내는 토대가 확보됐고, fork·로딩이 충실함이 확인됐다.

`all_invariants_passed: true`. CacheBlend 로직은 들어가지 않았다 (Step 4+).

## 2. 목표와 통과 기준

| Invariant | 명제 | 결과 |
|---|---|---|
| 1.1 | 우리 fork forward와 HF 표준 forward의 logits SHA-256 동일 | ✅ PASS |
| 1.2 | 모든 layer hidden state SHA-256 동일 (embedding 포함 33개) | ✅ PASS |
| 1.3 | 모든 layer의 q/k/v projection 출력 element-wise 동일 (RoPE 적용 전) | ✅ PASS |

## 3. 환경 정보

| 항목 | 값 |
|---|---|
| 실험 환경 | vast.ai A100-SXM4-80GB (instance `36787683`, 검증 후 destroy) |
| GPU / driver | NVIDIA A100-SXM4-80GB / 570.211.01 |
| dph_total | 할당 시 $1.007/h (`show instance`는 $1.139/h로 표시 — §10 참조) |
| 가동 시간 | ≈ 9.3분 |
| Python / PyTorch / transformers | 3.10 / 2.10.0+cu128 (CUDA 12.8) / 4.51.3 |
| 모델 | mistralai/Mistral-7B-Instruct-v0.2 |
| dtype / attention | float32 / eager |
| 결정론 설정 | `use_deterministic_algorithms(True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

## 4. 수정 / 신규 파일

| 파일 | 변경 | 사유 |
|---|---|---|
| `src/compblend/modeling/modeling_mistral.py` | 신규 (fork) | transformers 4.51.3 `modeling_mistral.py` — import 12줄 상대→절대, 본문 무수정 |
| `src/compblend/modeling/__init__.py` | 신규 | `MistralForCausalLM`, `MistralModel` export |
| `src/compblend/modeling/FORK_HASH.txt` | 신규 | source/fork sha256 + 변환된 import 12줄 기록 (진단 1단계용) |
| `tasks/step_01_fork_equivalence/run_fork_equivalence_check.py` | 신규 | invariant 1.1/1.2/1.3 검증 스크립트 |
| `results/step_01/vastai/summary.json` | 신규 | 검증 결과 (vast.ai에서 생성) |
| `tasks/step_01_fork_equivalence.md` | 수정 | stub → 자체완결 spec 확장, §fork/"Step 1 원칙" import 변환 반영 |

## 5. Tensor shape 명세

Mistral-7B-Instruct-v0.2: `H=4096`, `H_q=32`, `H_kv=8`, `D=128` (DECISIONS.md §3.8). prompt `"The capital of France is"` → `T=6` (BOS 포함).

| 변수 | shape | dtype | 비고 |
|---|---|---|---|
| `logits` | `(1, 6, 32000)` | float32 | invariant 1.1 — SHA-256 비교 |
| `hidden_states[i]` (33개) | `(1, 6, 4096)` | float32 | invariant 1.2 — layer별 SHA-256. i=0 embedding, i=1..32 layer 출력 |
| `q_proj` 출력 | `(1, 6, 4096)` | float32 | invariant 1.3 — H_q·D = 32·128. RoPE 적용 전 |
| `k_proj` / `v_proj` 출력 | `(1, 6, 1024)` | float32 | invariant 1.3 — H_kv·D = 8·128 (GQA). RoPE 적용 전 |

## 6. 구현 핵심

- **fork 방식**: transformers 원본은 전부 상대 import(`from ...activations` 등). `src/compblend/modeling/`에서 상대 import가 깨지므로 **import 12줄만 절대 경로로 변환**(`from ...X` → `from transformers.X`, `from .configuration_mistral` → `from transformers.models.mistral.configuration_mistral`). **forward 본문 1089줄은 byte 무수정** — `diff`로 정확히 12줄(import만) 차이 검증, `FORK_HASH.txt`에 source sha256 + 12줄 before/after 기록.
- **호출 방식**: `compblend.modeling.MistralForCausalLM.from_pretrained` (our) vs `transformers.AutoModelForCausalLM.from_pretrained` (HF 표준) — 동일 weight·입력.
- **순차 로드**: HF 로드 → forward → logits·hidden·q/k/v를 CPU 이동 → `del` + `empty_cache()` → our 로드 → forward → CPU 텐서끼리 비교. fp32×2=56GB 동시 적재 회피.
- **검증 계측 (외부 forward hook, fork 코드 무수정)**: 각 layer의 `q_proj`/`k_proj`/`v_proj`에 `register_forward_hook` 96개(32 layer × 3) 등록 → 모듈 출력(RoPE 적용 전)을 즉시 CPU 캡처. hidden state는 `output_hidden_states=True`로 33개 추출. hook 코드는 `run_fork_equivalence_check.py` 안에만.
- **비교 방식**: 1.1/1.2 = SHA-256 (`tensor.detach().cpu().to(fp32).numpy().tobytes()`, Step 0과 동일 기준), 1.3 = `torch.equal` (element-wise 정확 비교).
- `use_cache=False`, `attn_implementation="eager"`, fp32.

## 7. Invariant 검증 결과

| Invariant | 검증 내용 | 결과 |
|---|---|---|
| 1.1 | `our_logits_sha256` = `hf_logits_sha256` = `d338ec6a350dc6d7f81307c1479737cde13e7740d78ecf360b93ce4d23573c0b` | ✅ PASS |
| 1.2 | `n_hidden_states=33`, `mismatched_layers: []` (33개 전부 SHA-256 일치) | ✅ PASS |
| 1.3 | `n_layers=32`, `mismatched: []` (96개 q/k/v projection 전부 `torch.equal` True) | ✅ PASS |

## 8. 결과 데이터 (`results/step_01/vastai/summary.json`)

| 필드 | 값 |
|---|---|
| `all_invariants_passed` | `true` |
| `fork_source` | `transformers/models/mistral/modeling_mistral.py @ 4.51.3 (import 문 외 byte 무수정)` |
| `transformers_version` | `4.51.3` |
| `invariants.1.1.our/hf_logits_sha256` | `d338ec6a350dc6d7…` (동일) |
| `invariants.1.2.n_hidden_states` | `33` |
| `invariants.1.3.n_layers` | `32`, `mismatched: []` |

## 9. 환경 간 비교

local_a100 교차 검증(invariant 0.3)은 미수행 — vast.ai 단독 진행 (사용자 결정, Step 0과 동일 정책).

부수적 관찰: 이번 fork forward의 `hf_logits_sha256`·`our_logits_sha256`이 모두 `d338ec6a350dc6d7…`로, **직전 Step 0(다른 인스턴스 36769033)의 logits SHA-256과 동일**하다. 같은 prompt·model·fp32/eager 조건에서 vast.ai A100-SXM4-80GB 인스턴스가 달라도 logits가 bitwise 재현됨이 부수적으로 확인됐다. 단 이는 강한 invariant 0.3(vast.ai ↔ local_a100)을 대체하지 않으며, 0.3은 사용자 결정 시 별도 round로 유지한다.

## 10. 알려진 한계 / 의심스러운 부분

- ⚠️ **trivially 통과의 의미**: fork가 (import 외) 무수정이고 양쪽이 같은 weight·입력을 쓰므로 invariant 1.1~1.3이 통과하는 것은 정상이다. 이 통과는 "구현이 맞다"가 아니라 **"fork·로딩이 충실하다"**만 의미한다.
- 🔵 **fork의 transformers 내부 모듈 의존**: fork 단일 파일은 `modeling_utils`, `activations`, `rope_utils`, `cache_utils` 등 transformers 내부 모듈에 다수 의존. Step 4 이후 CacheBlend 로직 진입 시 추가 fork가 필요할 수 있음 — 그 시점에 별도 결정.
- 🔵 **dph_total 표시 불일치**: 할당 시 `search offers`는 $1.007/h, `show instance`는 $1.139/h로 표시. 콘솔이 authoritative. 원인 추적은 별도 round.
- 🔵 **git push -u 누락**: `step/step_01_layerwise_forward` 첫 push 시 upstream 미설정으로 2개 commit이 로컬에만 있었고, 인스턴스 clone 단계에서 발견·정정. `vast_helper`에 step 브랜치 push 가드 추가는 별도 round.
- 🔵 **확장본 §결과 저장 형식 예시의 `fork_source` 문구**: 코드(`run_fork_equivalence_check.py`)는 "(import 문 외 byte 무수정)"으로 정정했으나 task 파일 예시 블록은 아직 "(무수정)" — 별도 [meta] round에서 정정 예정.
- 🔵 **invariant 1.2 JSON key 정정**: Step 1 명명 정정 round에서 invariant 1.2의 JSON key를 `1.2_layerwise_hidden_equiv` → `1.2_per_layer_hidden_equiv`로 변경. 데이터 값(SHA, boolean)은 무변경, key 이름만 정정.

## 11. 다음 step

Step 2 — HF DynamicCache forward = no-cache forward (bitwise 일치). `tasks/step_02_dynamic_cache.md`는 현재 **stub 상태**(30줄)이므로, Step 1과 마찬가지로 진입 전 자체완결 task 파일로 확장이 필요하다.
