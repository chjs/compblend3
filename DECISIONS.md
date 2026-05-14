# DECISIONS.md — compblend3 프로젝트 결정사항 (v2)

> 이 문서는 프로젝트의 모든 핵심 결정사항과 그 근거를 기록한다.
> 컨텍스트 리셋 후 새 세션은 GOAL.md → PROGRESS.md → 이 파일 순으로 참조한다.
> 이 파일의 결정사항을 바꾸려면 명시적으로 사용자와 합의해야 한다.

---

## 1. 프로젝트 목표

### 1.1 최종 목표
**CompBlend** — 압축된 KV cache를 chunk reuse 시나리오에서 재사용하는 시스템.
구성요소: Gated HKVD, pluggable compression backend (1차 = KVzip), RoPE-aware position realignment.

### 1.2 현재 phase의 목표
**검증 가능한 CacheBlend의 HF transformers 재구현.**

### 1.3 실험 시나리오
"system prompt + doc_1 + doc_2 + ... + doc_N + user query"
- doc들의 크기, 갯수, 순서가 다양함
- 모든 doc이 답에 필요 (Loong 데이터셋 특성)
- 문서 순서가 바뀌어도 blending 결과 F1이 robust해야 함

### 1.4 이전 시도의 핵심 문제
CompBlend-old, compblend2 모두 같은 문제로 실패:
- **(a) 코드 검증이 어려움** — CacheBlend 구현이 원본과 정말 같은지 확신 부족
- **(b) F1 성능이 논문대로 안 나옴**
- 이 둘은 같은 문제의 두 얼굴: 검증이 안 되니 디버깅 불가

---

## 2. 모델

### 2.1 타겟 모델
- **Mistral-7B-Instruct-v0.2** (1차 검증)
  - 근거: 원본 CacheBlend 논문 모델, 논문 Table 직접 비교 가능
- **Llama-3.1-8B** (일반화 검증)
  - 근거: 알고리즘이 다른 아키텍처에서도 동작하는지 확인

### 2.2 아키텍처 차이 주의
- GQA: 둘 다 사용
- **Sliding window**:
  - **Mistral-7B-Instruct-v0.2**: 과거 config에 `sliding_window=32768`이 있었으나, 현재 HF config 기준 `sliding_window=null` (완전 비활성). HF 커밋 메시지 인용: "never meant to support sliding window attention".
  - **Llama-3.1-8B**: sliding window 없음.
  - **구현 규칙**: forward 진입 시 `config.sliding_window` 값을 assert/logging. sliding window가 실제 활성화된 모델이 들어오면 명시적으로 에러 (현재 phase에서 지원 ❌).
- RoPE theta: Mistral 1e6, Llama-3.1 5e5
- Max context: Mistral 32K (실용 8K~16K), Llama-3.1 128K

---

## 3. 기술 스택

### 3.1 3머신 구조

| 역할 | 머신 | OS / Arch | GPU | 용도 | 사용 빈도 |
|---|---|---|---|---|---|
| **Claude Code 실행** | MacBook Air M2 | macOS, ARM64 | 없음 (MPS는 미사용) | 코드 작성/편집, git, SSH로 vast.ai 트리거 | 항상 |
| **실험 실행 (primary)** | vast.ai 인스턴스 | Ubuntu 22.04/24.04, x86_64 | A100-SXM4 80GB | 모든 step의 실험, 결과 JSON 생성 | 항상 |
| **추가 검증 (occasional)** | 사용자 로컬 A100 | Ubuntu 24.04, x86_64 | A100-SXM4 80GB | "꼭 필요한 검증"만 별도 실행 | 가끔 |

### 3.2 환경별 패키지

**MacBook (Claude Code)**:
- Python 3.10 (uv)
- PyTorch (CPU 빌드, MPS 미사용) — smoke test 용도만, 실제 forward 안 함
- transformers (코드 정적 분석용)
- GPU 의존 패키지 ❌

**vast.ai + 로컬 A100 (실험 환경)**:
- Python 3.10 (uv)
- PyTorch 2.10.0+cu128
- CUDA 12.8
- transformers 4.51+ (Step 0에서 정확한 minor 버전 확정)
- flash-attn 2.7.x~2.8.x (Phase 6에서야 필요)

### 3.3 환경 결정 근거
이전 합의(KVzip 기반 torch 2.3.0 + CUDA 12.1)는 **폐기**.
이유: 사용자가 LMCache 호환을 위해 이미 torch 2.10.0+cu128로 작업 중. 환경을 사용자에게 맞춤.
KVzip 호환은 Phase 7에서 별도 확인. 안 되면 그때 patch.

### 3.4 환경 태그
환경별 결과를 구분하기 위한 태그 (`COMPBLEND_ENV_TAG` 환경변수 또는 자동 감지):
- `macbook` — Claude Code 실행 환경, 실험 결과 생성 ❌ (smoke test만)
- `vastai` — primary 실험 환경, 모든 step의 결과 JSON 생성
- `local_a100` — 가끔의 추가 검증 환경, 사용자가 결정한 step만 실행

### 3.5 Attention backend
- **Phase 1**: `eager` (검증 단계)
  - 근거: 결정론적 실행 가능, fp32 + deterministic mode에서 bitwise 비교 가능
- **Phase 6+**: `flash_attention_2` (KVzip 호환 단계)
  - 근거: KVzip이 flash-attn 2 기반
- **Phase 6 진입 시 추가 검증**: eager-CacheBlend ↔ flash-attn2-CacheBlend numerical 일치 (atol ~1e-3)

### 3.6 구현 방식: HF modeling 파일 fork
- HF의 `modeling_mistral.py`, `modeling_llama.py`를 우리 repo로 복사 후 직접 수정
- monkey-patch ❌
- 근거: 코드가 한 군데에 다 보임, assert 직접 박기 쉬움, transformers 버전 변화에 fragile하지 않음

### 3.7 Tokenization Contract

CacheBlend/LMCache 디버깅에서 가장 흔한 함정이 토큰화 차이다. 이 섹션의 규칙은 **모든 step에서 강제**.

**기본 원칙**:
- 모든 비교 실험은 **string prompt가 아니라 `input_ids` 기반**으로 수행.
- system / 각 doc / query는 **독립적으로 tokenize한 뒤 concat**.
- "한 번에 string concat 후 tokenize" 와 "각각 tokenize 후 concat"은 결과가 다를 수 있음 — 후자만 사용.

**Separator**:
- 각 doc 사이에는 **고정 separator token sequence** 삽입.
- 1차 separator: 빈 시퀀스 (separator 없음). 단순 concat.
  - 근거: 우리는 Loong 데이터셋 사용. 원본 데이터의 문서가 이미 자연스러운 경계를 가짐.
  - LMCache의 `LMCACHE_BLEND_SPECIAL_STR=" # # "`는 LMCache의 chunk 경계 마커로, 알고리즘에 본질적이지 않음.
- separator를 추가할지는 Step 4 진입 시 다시 결정. 추가 시 정확한 token id 시퀀스를 DECISIONS.md에 명시.

**Chat template**:
- **검증 단계 (Step 0~7)**: chat template ❌. raw token sequence만 사용.
  - 근거: chat template은 모델별로 다르고 (Mistral `[INST]`, Llama `<|start_header_id|>`), 알고리즘 검증 단계에서 디버깅 노이즈.
- **Step 8 (F1 측정)**: chat template 적용 여부 + 정확한 template 문자열을 task 파일에 명시.

**BOS / EOS**:
- BOS: 전체 prompt 맨 앞에만 1회 (system 앞).
- 각 doc tokenize 시 `add_special_tokens=False` 사용 → 중간에 BOS/EOS 끼지 않게.
- EOS: 생성 시작 전 prompt에는 ❌. 생성 stop 조건으로만 사용.

**모든 실험에서 기록할 것** (results/step_XX/{env}/tokenization.json):
- 사용한 tokenizer (HF id, revision)
- 각 chunk의 token_ids
- chunk 경계 (token offset)
- separator id 시퀀스 (있다면)
- BOS/EOS 위치
- chat template 적용 여부

LMCache reference와의 비교 시 양쪽이 **동일한 input_ids, position_ids, attention_mask, chunk boundary**를 사용했음을 명시적으로 확인.

### 3.8 KV Cache Data Model

CacheBlend / HF DynamicCache / ChunkedKVStore의 layout을 정확히 명세.

**Layer별 K/V shape** (단일 batch 가정, B=1):

| Stage | Shape | dtype | 비고 |
|---|---|---|---|
| HF 표준 (DynamicCache 내부) | `(B, H_kv, T, D)` | fp32 (검증), bf16 (Phase 6+) | `H_kv=num_key_value_heads`, GQA repeat 적용 전 |
| Attention 입력 (q dot k) | `(B, H_q, T, D)` | 동상 | k가 `repeat_interleave(H_q // H_kv, dim=1)` 거친 후 |
| Chunk export (저장) | `(H_kv, T_chunk, D)` per layer per chunk | 동상 | B 차원 제거, chunk별 분리 |
| Blended cache (Step 4+) | `(B, H_kv, T_total, D)` | 동상 | 모든 chunk concat 후 |

기호:
- `B`: batch (1)
- `H_q`: num_attention_heads (Mistral: 32)
- `H_kv`: num_key_value_heads (Mistral: 8, GQA ratio 4)
- `T`: sequence length
- `D`: head_dim (Mistral: 128)

**Chunk metadata** (각 chunk마다):
```python
@dataclass
class ChunkMeta:
    chunk_id: str           # 예: "system", "doc_0", "doc_1", ...
    token_ids: list[int]    # 이 chunk의 토큰 id
    original_offset: int    # 원본 prefill 시의 시작 position
    new_offset: int         # blend 시의 시작 position
    original_length: int    # 토큰 수
    is_cacheable: bool      # query는 False, system/doc은 True
    is_permanent_hit: bool  # system은 True, doc은 False
```

**HF cache 호환성**:
- HF DynamicCache는 `key_cache: list[Tensor]`, `value_cache: list[Tensor]` 구조 (layer별).
- 우리 ChunkedKVStore는 내부적으로 **chunk별로 분리** 저장하지만, forward 호출 시 **DynamicCache 인터페이스로 변환** 후 모델에 전달.
- 변환 시점에 RoPE re-rotation 적용 (Step 4.1).

**StaticCache / QuantizedCache / OffloadedCache 미지원**:
- Phase 1~6에서는 **DynamicCache만 지원**.
- Phase 7 (KVzip 통합) 시 quantized cache layout과의 호환 결정.

**GQA repeat_interleave 기준**:
- KV cache는 **GQA repeat 적용 전** 상태로 저장 (`H_kv`).
- attention 계산 직전에만 repeat 적용 (메모리 절약).

---

## 4. CacheBlend 알고리즘 범위

### 4.1 In-scope (전체 구현)
- 다중 chunk KV cache 결합
- HKVD(High KV Deviation) 기반 선택적 recompute
- Positional encoding 재계산 (RoPE re-rotation)
- Layer별 selection ratio 스케줄링

### 4.2 진행 방식
한 번에 다 만들지 않고 **Step별로 invariant를 통과시키며 추가**.

### 4.3 Prefill vs Decode
1차 목표는 prefill 중심. Generation 통합은 prefill 검증 완료 후 결정.

---

## 5. Chunk 정책

### 5.1 Chunk 단위: Doc 단위
- 1 doc = 1 chunk
- 가변 길이 허용
- 근거: 사용자 워크로드와 자연스럽게 일치, hit ratio 추적 직관적

### 5.2 System prompt 처리
- **항상 position 0 고정**
- **영구 캐시 hit으로 처리**

### 5.3 User query 처리
- **항상 마지막 위치**
- **캐싱 안 함** (매번 새로움)

### 5.4 Hit ratio 측정
둘 다 측정:
- **Token-level**: `sum(cached_doc_lengths) / total_input_tokens`
- **Doc-level**: `cached_docs / total_docs`

---

## 6. 검증 전략

### 6.1 핵심 원칙: Invariant-first development
- 모든 기능은 invariant를 **먼저** 정의 후 구현
- Invariant는 unit test로 자동 검증 (deterministic)
- Invariant 통과 못 하면 commit 안 함

### 6.2 핵심 원칙: Bottom-up gating
Step N 통과 못 하면 Step N+1 진행 금지. 각 step에 사용자 리뷰 게이트.

### 6.3 검증 단계 (Step 0 ~ 8)

| Step | 목표 | 통과 기준 |
|---|---|---|
| 0 | HF eager forward 결정론 확인 | seed 고정 시 동일 입력 → 동일 출력 (bitwise, 같은 환경 내) |
| 1 | Our layerwise forward = HF 표준 forward (no cache) | bitwise 일치 |
| 2 | HF DynamicCache forward = no-cache forward | bitwise 일치 |
| 3 | ChunkedKVStore 자료구조 정확성 | concat → DynamicCache 변환 → forward, Step 2와 bitwise 일치 |
| 4 | N chunks 따로 prefill → concat = vanilla full prefill | 4.1: RoPE re-rotation 정확성, 4.2: N-chunk concat = vanilla (atol 명세) |
| 5 | 1 chunk reuse = vanilla forward | atol 일치 |
| 6 | N chunks reuse, recompute_ratio=1.0 = vanilla forward | atol 일치 |
| 7 | HKVD 알고리즘 정확성 | numpy oracle과 동일 선택 index |
| 8 | recompute_ratio<1.0 적용, Loong F1 측정 | F1 곡선, 논문 패턴 비교 |

### 6.4 모든 의심을 invariant로 변환
막연한 우려가 들면 자동 검증 가능한 명제로 변환:
- "RoPE가 잘못된 것 같다" → "Layer L에서 cached K에 R(p_new)R(-p_old) 적용 후 fresh K와 element-wise 일치"
- "Attention mask가 이상하다" → "Q×K^T의 mask 위치가 정확히 -inf"
- "HKVD 선택이 이상하다" → "numpy oracle algorithm과 동일 index 출력"

### 6.5 Bitwise 가능성
- **같은 HF 환경 안**: fp32 + `torch.use_deterministic_algorithms(True)` + eager로 가능
- **HF ↔ vLLM**: 구조적으로 불가 (paged memory, kernel fusion 경계, mask 시점 차이)
- **vast.ai ↔ 로컬**: 가능성 있음 (같은 driver/CUDA wheel이면). SHA-256 우선, atol 1e-5 fallback.

### 6.6 단계적 검증 입력 확장
```
Stage 0: 단일 토큰, 1 layer       → backend sanity check
Stage 1: 짧은 문장 (~50 tokens)    → full model, prefill만
Stage 2: system + doc1            → 단일 chunk
Stage 3: system + doc1 + doc2     → 2 chunks (CacheBlend 핵심 케이스)
Stage 4: N chunks, doc 순서 셔플   → workload 일반화
Stage 5: 실제 데이터셋             → Loong F1 비교
```

---

## 7. Reference 저장소

### 7.1 코드 분석용 (실행 ❌)
- **YaoJiayi/CacheBlend**: 논문 저자 초기 구현 (vLLM 포크, `vllm_blend/`)
- **CompBlend-old**: 이전 시도, 결과는 있었으나 신뢰 부족
- **compblend2**: 직전 실패 시도, CLAUDE.md 워크플로우 패턴 참고

### 7.2 실제 실행 reference

**LMCache reference repository pinning** (모든 필드를 Phase 0 task에서 확정):

| 항목 | 값 | 확정 시점 |
|---|---|---|
| Repo | `chjs/LMCache` | 확정됨 |
| Branch | `fix/cacheblend-vllm-v0.17.1-compat` | 확정됨 |
| Commit SHA | **TBD** (Phase 0에서 `git ls-remote` 또는 web_fetch로 확정) | Phase 0 |
| vLLM 버전 | v0.17.1 (LMCache README 기준) | Phase 0 확정 |
| vLLM commit SHA | **TBD** | Phase 0 |

**vLLM patch**:
- LMCache examples/blend_kv_v1/README.md 의 정확한 patch 내용을 Phase 0에서 web_fetch로 읽어 `patches/lmcache-vllm-cacheblend.patch` 로 저장.
- 본 DECISIONS.md에는 의도만 기록, 실제 diff는 patch 파일로.
- 우리 메모리 (`gpu_worker.py`의 `VLLMModelTracker.register_model`)와 ChatGPT 의견 (`init_worker_distributed_environment` + `ensure_kv_transfer_initialized`)이 다르므로, **실제 LMCache README가 진실 공급원**.
- 참조 URL: https://github.com/chjs/LMCache/blob/fix/cacheblend-vllm-v0.17.1-compat/examples/blend_kv_v1/README.md

**관련 task**:
- Phase 0 task 0.7 신설: LMCache pinning 확정 + patch 회수.

### 7.3 최종 통합 대상 (Phase 7)
- **KVzip** (snu-mllab/KVzip): CompBlend의 첫 backend

### 7.4 평가 데이터셋
- **Loong** (MozerWang/Loong, EMNLP 2024 Oral)
  - 평균 11개 문서/instance, 모든 문서가 답에 필요 (Leave No Document Behind)
  - Level 1 (Spotlight Locating)부터 검증 → Level 4 (Chain of Reasoning) 최종
  - 영어, Academic Papers 도메인부터
  - Context: Mistral은 ~16K 이하 instance만 (32K context window 안전 마진)

### 7.5 논문
- CacheBlend: arxiv.org/pdf/2405.16444
- CompBlend (자체 초안): 본 repo `notes/compblend_paper_draft.pdf` (Samsung Best Paper Award 제출본)

---

## 8. 인프라

### 8.1 3머신 구조 + SSH 자동화

- **MacBook M2** (Claude Code 실행): 코드 작성/편집/git, vast.ai SSH 트리거
- **vast.ai A100-SXM4 80GB** (primary 실험): 모든 step 실험 실행
- **사용자 로컬 A100 80GB** (occasional 검증): 사용자가 "꼭 필요한 검증"이라 판단한 step만 별도 실행

### 8.2 SSH 자동화 규칙

Claude Code(MacBook)가 vast.ai에 SSH로 직접 명령 트리거. 사용자 중간 개입 없음.

규칙 3개:
1. **SSH alias 고정**: 사용자 `~/.ssh/config`에 `Host vast` alias 정의. Claude는 `ssh vast '...'` 만 사용. IP/포트 inline ❌.
2. **인스턴스 상태 ping 먼저**: 매 step 시작 시 `ssh vast 'nvidia-smi'` 가벼운 확인. 인스턴스가 죽어있으면 사용자에게 켜달라고 요청 후 멈춤. **Claude가 자동으로 인스턴스 spawn 시도 ❌**.
3. **시크릿은 vast.ai의 .env에서만**: SSH 명령 인라인에 `HF_TOKEN=hf_...` 같은 시크릿 절대 ❌. vast.ai의 `.env`는 인스턴스 셋업 시 한 번 채워두고 그걸 source.

### 8.3 파괴적 명령

vast.ai는 가상 인스턴스라 자유. `rm -rf`, `pip uninstall`, 인스턴스 내 어떤 명령도 사용자 사전 승인 없이 OK.
(인스턴스 자체의 destroy / stop은 사용자 권한 — Claude가 vast.ai API 호출 ❌)

### 8.4 결과 회수 워크플로우

vast.ai에서 실험 → `results/step_XX/vastai/summary.json` 작성 → vast.ai에서 git add + commit + push → MacBook에서 git pull로 회수.
scp 사용 ❌ (git 한 군데로 통일).

### 8.5 환경별 결과 분리
- `results/step_XX/vastai/` — 모든 step에서 채워짐
- `results/step_XX/local_a100/` — 사용자가 결정한 step만, 대부분 비어 있을 수 있음
- `results/step_XX/macbook/` — 존재하지 않음 (smoke test 결과만 별도로)

### 8.6 결과 비교 강도
1. **1순위: SHA-256 bitwise 일치** (vastai vs local_a100, 운 좋으면 일치)
2. **2순위: atol 1e-5 일치** (환경 차이가 있을 때 fallback)
3. **3순위: logit top-k 일치** (Step 8 같은 task metric 단계)

`local_a100` 결과가 없는 step은 `vastai` 단독으로 invariant 검증 진행.

---

## 9. 작업 진행 원칙

### 9.1 컨텍스트 리셋 친화 (Claude Code 작업 시)
- Claude Code는 MacBook M2에서 실행. 매 세션은 컨텍스트 리셋 후 git pull한 상태에서 시작 가능해야 함.
- 모든 상태는 파일에 (GOAL.md, DECISIONS.md, CLAUDE.md, PROGRESS.md, task 파일들)
- Task 단위는 작게, 한 컨텍스트 안에서 끝나도록
- 각 task는 self-contained (해당 task 파일만 읽고도 작업 가능)
- 매 task 후 commit + PROGRESS.md 업데이트
- 새 세션 진입점: GOAL.md → PROGRESS.md → 다음 task 파일

### 9.1a Step 작업의 SSH 흐름 (typical)

```
MacBook에서 새 세션 시작 (Claude Code)
   ↓
GOAL.md, PROGRESS.md, tasks/step_XX_*.md 읽기
   ↓
ssh vast 'nvidia-smi' 로 인스턴스 살아있는지 확인
   ↓ (살아있음)
MacBook에서 코드 작성/편집 (src/, tasks/step_XX/)
   ↓
git add + commit + push
   ↓
ssh vast 'cd compblend3 && git pull && python tasks/step_XX/...'
   ↓
실험 결과가 vast.ai의 results/step_XX/vastai/summary.json 에 저장
   ↓
ssh vast 'cd compblend3 && git add results docs && git commit -m "[step_XX] results" && git push'
   ↓
MacBook에서 git pull (결과 회수)
   ↓
보고서 docs/reports/step_XX_report.html 작성 (MacBook에서)
   ↓
git add + commit + push
   ↓
사용자 리뷰 대기
```

### 9.2 매 step 필수 산출물 (compblend2 패턴)
1. **docs/reports/step_XX_report.html** — HTML 한글, semantic structure, Gmail-compatible tables, color badges
2. **docs/prompts/step_XX_prompt.md** — 해당 step 시작 시 사용자 프롬프트 기록
3. **PROGRESS.md** 업데이트 — 완료 표시, 다음 step 명시
4. **results/step_XX/{vastai,local}/summary.json** — 표준 포맷 결과

### 9.3 보고서 필수 내용
- modified source files (table)
- tensor shapes (table)
- cache layout
- RoPE handling
- recompute logic
- attention masking
- unit tests 결과 (table)
- risks
- next step

### 9.4 리뷰 워크플로우
- Step 완료 → 보고서 작성 → 사용자 리뷰 → 승인 → 다음 step
- 리뷰 안 받고 다음 step 진행 금지
- 리뷰 형태: README의 보고서를 사용자가 직접 읽음, GitHub issue 또는 직접 코멘트

### 9.5 솔직성 원칙
- 확신 없는 정보는 "확인 필요"로 표시
- 할 수 없는 일은 명확히 밝힘
- Negative result도 그대로 보고 (paper-headline 위해 데이터 굽지 않음)
- F1이 안 나오면 어디서 안 나오는지 추적 가능한 구조 유지

### 9.6 Git workflow
- `main` 브랜치는 항상 검증 통과한 상태만 유지
- 각 step은 `step/step_XX_<short-desc>` 브랜치에서 진행
- Step 완료 → PR/merge → tag `step_XX_done`
- Commit 메시지: `[step_XX] <description>` 형태
- Claude Code는 commit까지 자동, push는 사용자 승인 시

### 9.7 문서 언어
- **한글 우선** (compblend2 정책 계승)
- 외부 라이브러리 함수명/클래스명, 논문 인용, 파일 경로, 코드 심볼은 원문 유지
- tensor shape 표기는 원문 (예: `(B, H_kv, T, D)`)

---

## 10. 비목표 (이번 phase에서 안 함)

- vLLM/LMCache와의 **bitwise 일치는 목표가 아니다** (구조적으로 불가). Numerical 동등(atol)과 task quality 동등(F1)이 목표.
- vLLM/LMCache와의 **sampling output 일치는 목표가 아니다**. 우리 검증은 prefill 결정성 + 같은 입력에서의 동등성 위주.
- TTFT 최적화 (1차 목표는 correctness)
- Multi-token decode 최적화 (first-token-then-fresh-prefill 방식 우선)
- **Multi-GPU / Tensor Parallel은 1차 목표에서 제외**. Phase 7 이후 KVzip + multi-GPU 결정 시 재논의.
- Quantization 기반 compressor 통합 (Phase 8+)
- **Compressed KV 상태에서의 직접 blending은 1차 목표에서 제외** — Phase 7 진입 시 §11에서 다시 결정.
- 사용자가 명시적으로 합의하지 않은 추가 기능 — premature abstraction 금지

---

## 11. KVzip Integration Hypothesis (Phase 7)

KVzip-compressed KV를 CacheBlend와 어떻게 결합할지의 가설.
**Phase 7 진입 시 최종 결정**, 그 전까지는 가설 단계.

| Phase | 구현 가설 | 위험 |
|---|---|---|
| 1차 (Phase 7 초반) | **Decompress 후 blend**: KVzip-compressed KV를 uncompressed full KV layout으로 복원 → 우리 CacheBlend 적용 | KVzip의 압축 이점 일부 손실. 다만 correctness 검증 용이. |
| 2차 (Phase 7 후반) | **Retain layout으로 blend**: KVzip이 보존한 sparse token만 dense layout으로 옮긴 후 blend | layout 변환 비용. retained token이 chunk boundary와 정렬되는지 확인 필요. |
| 3차 (Phase 8) | **Compressed/sparse layout에서 직접 blend**: 압축 상태 그대로 blending | 가장 어렵. KV 위치 추적, RoPE re-rotation 정확성 등 추가 invariant 필요. |

**1차 가설을 default로 진행** (구현 용이성). 2차/3차는 Phase 7 진입 후 결과 보고 결정.

**현재 phase (Step 0~8)에서는 영향 없음** — uncompressed full KV 기준으로 진행하므로 KVzip 통합 방식이 무엇이든 결과는 같음.

이 가설을 미리 명시하는 이유:
- ChatGPT 검토(2026-05-14)에서 "안 정해두면 cache data model 설계가 흔들릴 수 있다"는 지적.
- Phase 7 진입 시 잊지 않도록 닻 역할.

---

## 12. Phase 분할

| Phase | 포함 step | 목표 |
|---|---|---|
| **Phase 0** | 환경 셋업 | uv venv, torch 2.10+cu128, 모델 다운로드, Loong manifest, LMCache pinning |
| **Phase 1** | Step 0~3 | HF forward + cache 자료구조 토대 |
| **Phase 2** | Step 4~6 | CacheBlend 핵심 (multi-chunk concat, selective reuse) |
| **Phase 3** | Step 7 | HKVD 알고리즘 |
| **Phase 4** | Step 8 (Mistral) | F1 실험 (Loong Level 1) |
| **Phase 5** | Llama-3.1-8B 일반화 | 모델 확장 |
| **Phase 6** | flash-attn 2 백엔드 전환 | KVzip 호환 준비 |
| **Phase 7** | KVzip 통합 (CompBlend 첫 backend) | §11 가설 중 하나 선택 |
| **Phase 8** | Gated HKVD + 추가 backends | CompBlend 완성 |

---

## 13. 변경 이력

- **2026-05-14 v1**: 초안 작성 (KVzip 호환 torch 2.3.0 + CUDA 12.1 기반)
- **2026-05-14 v2**: 환경 정보 갱신 후 폐기/변경
  - 사용자 환경: Ubuntu 24.04, nvcc 12.8, torch 2.10.0+cu128
  - LMCache 호환을 위해 사용자 환경에 맞춤
  - 검증 단계 Step 2~3 신설 (KV cache 자료구조 명시적 검증)
  - compblend2 워크플로우 패턴 (mandatory reporting, HTML 한글 보고서) 계승
  - 이전 시도 분석: CompBlend-old + compblend2 모두 "검증 어려움 + F1 안 나옴"으로 실패
  - 환경별 결과 분리 + 비교 스크립트 도입
- **2026-05-14 v3**: 3머신 구조로 정정
  - Claude Code는 MacBook M2에서 실행 (vast.ai ❌). 실험만 vast.ai. 사용자 로컬 A100은 occasional 검증용.
  - SSH 자동화: Claude가 `ssh vast '...'`로 직접 트리거. 사용자 중간 개입 없음.
  - SSH 규칙 3개: alias 고정, 인스턴스 ping 먼저 (자동 spawn ❌), 시크릿은 vast.ai .env에서만.
  - 파괴적 명령은 가상 인스턴스라 자유 (단 인스턴스 자체 destroy는 사용자 권한).
  - 환경 태그: macbook / vastai / local_a100 셋으로 분리.
- **2026-05-14 v4**: ChatGPT 검토 의견 Tier 1 반영
  - §2.2 Mistral v0.2 sliding window 정정 (완전 비활성, HF 커밋 메시지 인용)
  - §3.7 Tokenization Contract 신설 (input_ids 기반, separator 정책, chat template 끄기, BOS/EOS 규칙)
  - §3.8 KV Cache Data Model 신설 (layer별 shape, ChunkMeta, DynamicCache 변환, GQA 기준)
  - §7.2 LMCache reference에 commit SHA pinning placeholder, Phase 0 task 0.7 신설 예고
  - §10 비목표에 HF↔vLLM bitwise 일치 ❌, sampling output 일치 ❌, multi-GPU ❌, compressed 직접 blend ❌ 명시
  - §11 KVzip Integration Hypothesis 신설 (3-phase 가설, 1차 = decompress 후 blend default)
  - 미해결로 남긴 것: §11 최종 결정은 Phase 7 진입 시. recompute_ratio=1.0 invariant 전제조건은 step_06 task 파일에 반영.
