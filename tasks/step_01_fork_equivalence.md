# Step 1 — fork 동치성 검증 (fork된 코드 = HF 표준 forward, no cache)

> Self-contained task. 이 파일만 읽고도 작업 가능해야 한다.
> 2026-05-15: stub에서 본격 task 파일로 확장. 해석은 **(A) 무수정 fork**로 확정,
> 검증 계측은 **옵션 1 (외부 forward hook, fork 코드 무수정)**로 확정.

---

## 목표

HF transformers 4.51.3의 `modeling_mistral.py`를 `src/compblend/modeling/`로 **무수정 fork**한다.
우리 repo 안의 fork된 코드로 forward를 돌렸을 때 HF 표준(transformers 설치본)과 **bitwise 일치**함을 확인한다.

이 단계는 CacheBlend 로직을 넣지 않는다 — 코드를 그대로 가져오기만 한다. 포인트는
(1) 우리 코드 베이스 안에서 같은 결과를 내는 토대 확보, (2) layer 단위 검증 하니스 구축.
실제 modeling 수정(RoPE re-rotation, chunk concat 등)은 Step 4+에서 시작한다.

## Step 1 원칙 (반드시 지킬 것)

- **Step 1 원칙: `src/compblend/modeling/` 안의 코드는 import 문 외 byte 무수정. import은 상대→절대 경로 변환만 허용, 추가/제거 ❌. 검증 계측은 외부 forward hook으로만.**
- **invariant 1.3은 RoPE 전 q/k/v 기준. RoPE 후 검증은 이후 step에서 필요 시 추가.**

## 사전 조건

- Step 0 통과 (tag `step_00_done`, main에 merge됨)
- Branch: `step/step_01_layerwise_forward` (main에서 분기)
- vast.ai 인스턴스: 작업 중 `scripts/vast_helper.py`로 할당 (disk_space 필터 + destroy 대화형 패치 적용본 — commit `e39dd54`, `239e53d`)
- 모델 캐시: Mistral-7B-Instruct-v0.2 (인스턴스 셋업 시 다운로드)
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`

## 통과 기준 / Invariants

fork가 무수정이고 양쪽이 같은 weight·같은 입력을 쓰므로 세 invariant 모두 **trivially 통과해야 한다**.
하나라도 실패하면 fork가 충실하지 않거나 로딩이 잘못된 것 — 즉시 추적한다.

### Invariant 1.1 — Forward 동등성
같은 `input_ids`에 대해 우리 fork forward와 HF 표준 forward의 logits SHA-256 동일.
```
sha256(our_model(x).logits) == sha256(hf_model(x).logits)
```
조건: fp32, `attn_implementation="eager"`, `use_cache=False`.

### Invariant 1.2 — Layer-by-layer 동등성
`output_hidden_states=True`로 추출한 모든 layer의 hidden state SHA-256 동일.
```
∀ i ∈ [0, num_layers]: sha256(our_h_i) == sha256(hf_h_i)
```
(`hidden_states` tuple 길이 = `num_hidden_layers + 1`, embedding 출력 포함.)

### Invariant 1.3 — q/k/v projection 동등성
모든 layer의 `q_proj` / `k_proj` / `v_proj` 모듈 **출력**(= RoPE 적용 전 q/k/v)이 element-wise 동일.
```
∀ layer i, ∀ p ∈ {q,k,v}: torch.equal(our_p_i, hf_p_i)
```
RoPE 후 q/k는 검증하지 않는다 — 무수정 fork이므로 invariant 1.2(hidden state 일치)가 RoPE 통과를 간접 보증한다.

## 구현 사양

### fork

- **소스**: 설치된 transformers 4.51.3의 `transformers/models/mistral/modeling_mistral.py` (1101줄).
- **대상**: `src/compblend/modeling/modeling_mistral.py` — **import 문 외 byte 무수정. import은 상대→절대 경로 변환만 허용, 추가/제거 ❌.**
  - transformers 원본은 전부 상대 import(`from ...activations` 등 11개 + `from .configuration_mistral` 1개 = 12개)를 쓴다. `src/compblend/modeling/`에서는 상대 import가 깨지므로 이 12줄만 절대 경로로 변환한다 (`from ...X` → `from transformers.X`, `from .configuration_mistral` → `from transformers.models.mistral.configuration_mistral`). 본문 1089줄은 byte 무수정.
  - 파일 상단에 출처 주석 1줄만 추가 허용.
  - import 변환 후 transformers 설치본의 내부 모듈(base 클래스, RoPE 헬퍼, config 등)을 그대로 사용한다 — fork 대상 ❌. fork = "modeling_mistral.py 단일 파일이 우리 소유"라는 뜻이지 transformers 전체 재구현이 아니다.
  - 변환 검증: 변환 후 원본과 `diff` → 정확히 12줄(import만) 차이임을 확인하고 `FORK_HASH.txt`에 source sha256 + 12줄 before/after를 기록한다 (invariant 실패 시 진단 1단계용).
- `src/compblend/modeling/__init__.py` — `MistralForCausalLM`, `MistralModel` export.
- `src/compblend/modeling/FORK_HASH.txt` — source sha256 + fork sha256 + 변환된 import 12줄 기록.
- assert/hook/dump를 fork 디렉토리 안에 추가 ❌ (그건 Step 4+).

### 호출 방식 ("our forward" vs "HF 표준")

- **our**: `from compblend.modeling import MistralForCausalLM` → `MistralForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")`
- **HF 표준**: `from transformers import AutoModelForCausalLM` → `AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, attn_implementation="eager")`
- 두 모델은 같은 weight를 로드한다. 우리 클래스는 HF 클래스의 무수정 복사이므로 `from_pretrained`가 별도 등록 없이 동작한다.
- **순차 로드 기본**: HF 표준 모델 로드 → forward → 결과(logits·hidden_states·hook 캡처)를 CPU로 이동 → `del` + `torch.cuda.empty_cache()` → our 모델 로드 → forward → CPU 텐서끼리 비교. 이유: fp32×2 = 56GB는 A100 80GB에서 빠듯하고 hidden state·q/k/v hook 캡처를 더하면 OOM 위험. Step 1은 시간 압박 없음.
- 동시 로드(두 모델을 GPU에 함께 적재)는 옵션 — 메모리 여유를 확인한 경우에만.

### 검증 계측 (옵션 1 — 외부 forward hook, fork 코드 무수정)

모든 계측은 `tasks/step_01_fork_equivalence/run_fork_equivalence_check.py` 안에서만. `src/compblend/modeling/`에 추가 ❌.

- **1.1 / 1.2**: 각 모델을 `model(**ids, output_hidden_states=True, use_cache=False)`로 1회 forward → `out.logits`(1.1), `out.hidden_states`(1.2) 비교.
- **1.3**: forward 전에 각 layer의 `self_attn.q_proj` / `k_proj` / `v_proj`에 `register_forward_hook` 등록 → hook이 모듈 **출력**(projection 결과, RoPE 적용 전)을 layer별로 CPU에 캡처. our·hf 각각 순차로 캡처하고, 두 forward가 끝난 뒤 layer별 `torch.equal` 비교. forward 후 hook 제거.

### 새 파일

| 파일 | 내용 |
|---|---|
| `src/compblend/modeling/__init__.py` | `MistralForCausalLM`, `MistralModel` export |
| `src/compblend/modeling/modeling_mistral.py` | transformers 4.51.3 `modeling_mistral.py` fork (import 12줄 상대→절대, 본문 무수정) |
| `src/compblend/modeling/FORK_HASH.txt` | source sha256 + fork sha256 + 변환된 import 12줄 기록 |
| `tasks/step_01_fork_equivalence/run_fork_equivalence_check.py` | 검증 스크립트 (순차 로드, forward hook, invariant 1.1/1.2/1.3 검증, summary.json 작성) |

`src/compblend/__init__.py`는 이미 존재. 디렉토리는 `mkdir -p src/compblend/modeling tasks/step_01`.

### 결정론 / shape 규칙 (CLAUDE.md §6 준수)

- `run_fork_equivalence_check.py` entry point에서 seed 고정 + `torch.use_deterministic_algorithms(True)` 등 (Step 0 `run_determinism_check.py`와 동일 패턴).
- tensor 변수에 shape 주석. hidden state `(B, T, H)`, q/k/v_proj 출력 `(B, T, H_q*D)` 또는 `(B, T, H_kv*D)` — 실제 shape는 구현 시 확인 후 주석.

## 실행 방법

MacBook에서 코드 작성 → commit → push. vast.ai에서 실행:
```bash
# vast.ai (Claude가 ssh로)
cd compblend3 && git pull
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tasks/step_01_fork_equivalence/run_fork_equivalence_check.py --out results/step_01/vastai/
# 결과 git push → MacBook git pull로 회수
```
로컬 A100 검증은 이번 step 미수행 (vast.ai 단독). 환경 간 비교는 사용자 결정 시 별도 round.

## 결과 저장 형식

`results/step_01/vastai/summary.json`:
```json
{
  "step": 1,
  "env_tag": "vastai",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "torch_dtype": "float32",
  "attention_implementation": "eager",
  "transformers_version": "4.51.3",
  "fork_source": "transformers/models/mistral/modeling_mistral.py @ 4.51.3 (import 문 외 byte 무수정)",
  "prompt": "The capital of France is",
  "invariants": {
    "1.1_forward_logits_equiv": {
      "passed": true,
      "our_logits_sha256": "...",
      "hf_logits_sha256": "..."
    },
    "1.2_per_layer_hidden_equiv": {
      "passed": true,
      "n_hidden_states": 33,
      "mismatched_layers": []
    },
    "1.3_qkv_projection_equiv": {
      "passed": true,
      "n_layers": 32,
      "mismatched": []
    }
  },
  "all_invariants_passed": true
}
```
핵심 필드 (`compare_results.py`가 사용): `all_invariants_passed`, `invariants.*.passed`, `*_sha256`.

## 보고서 작성 가이드

`docs/reports/step_01_fork_equivalence_report.md` 작성 (Markdown, `docs/design/report_style.md` 따름). 필수 섹션:

1. **요약** — fork 무수정 여부 + invariant 3종 PASS/FAIL
2. **목표와 통과 기준** — invariant 1.1/1.2/1.3
3. **수정 / 신규 파일** (table)
4. **Tensor shape 명세** (table: hidden state, q/k/v_proj 출력)
5. **구현 핵심** — fork 방식(무수정 + 출처 주석), 호출 방식, forward hook 계측
6. **Invariant 검증 결과** (table, ✅/❌ badge)
7. **결과 데이터** — `summary.json` 핵심
8. **알려진 한계** — RoPE 후 q/k 미검증(1.2가 간접 보증), local_a100 미검증, transformers 버전 고정 의존성
9. **다음 step** — Step 2 (HF DynamicCache forward = no-cache forward)

## 다음 step 게이트

- [ ] `results/step_01/vastai/summary.json`의 `all_invariants_passed: true`
- [ ] `docs/reports/step_01_fork_equivalence_report.md` 작성
- [ ] 사용자 리뷰 승인
- 통과 시 §7.1 절차대로 `main`에 `--no-ff` merge + tag `step_01_done` + 브랜치 삭제 → Step 2 진입

## 작업 순서

1. **이 확장본을 `main`에 `[meta]` commit** (task 파일 확장은 메타 문서 변경 — §7.1 예외 (a)). push.
2. `git checkout -b step/step_01_layerwise_forward` (main에서 분기).
3. `src/compblend/modeling/` fork + `tasks/step_01_fork_equivalence/run_fork_equivalence_check.py` 작성. py_compile + import smoke test (MacBook).
4. commit + push → vast.ai 인스턴스 할당·셋업 → 실행 → 결과 회수 → 인스턴스 destroy.
5. 보고서 작성 → PROGRESS.md 갱신 → commit → 사용자 리뷰 대기.

## 솔직성 노트

- fork가 무수정이면 invariant 1.1~1.3은 trivially 통과가 정상. 통과했다고 "구현이 맞다"가 아니라 "fork·로딩이 충실하다"만 의미한다 — 보고서에 이 점을 명확히.
- Fork 단일 파일은 transformers 내부 모듈(`modeling_utils`, `activations`, `rope_utils` 등)에 다수 의존. Step 4 이후 CacheBlend 로직 진입 시 추가 fork 필요할 수 있음 — 그 시점에 별도 결정.
- **invariant 실패 시 진단 순서**: (1) `modeling_mistral.py` byte hash 확인, (2) 두 model 인스턴스의 config·state_dict 동등성 확인, (3) `MistralForCausalLM` 직접 vs `AutoModelForCausalLM` dispatch 차이 확인. fork 코드 수정으로 통과시키기 ❌ — 왜 다른지 먼저 규명.
- transformers 4.51.3에 고정된 fork다. 버전 올리면 재fork 필요 — `fork_source`에 버전 명시로 추적.
