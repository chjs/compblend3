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
| 1 | fork 동치성 검증 (fork된 코드 = HF 표준 forward, no cache) | bitwise 일치 |
| 2 | HF DynamicCache forward = no-cache forward | bitwise 일치 |
| 3 | ChunkedKVStore 자료구조 정확성 | concat → DynamicCache 변환 → forward, Step 2와 bitwise 일치 |
| 4 | N chunks 따로 prefill → concat = vanilla full prefill | 4.1: RoPE re-rotation 정확성, 4.2: N-chunk concat = vanilla (atol 명세) |
| 5 | 1 chunk reuse = vanilla forward | atol 일치 |
| 6 | N chunks reuse, recompute_ratio=1.0 = vanilla forward | atol 일치 |
| 7 | HKVD 알고리즘 정확성 | numpy oracle과 동일 선택 index |
| 8 | recompute_ratio<1.0 적용, Loong F1 측정 | F1 곡선, 논문 패턴 비교 |

> Step 1 행: 당초 "Our layerwise forward"로 명명했으나 실제 작업은 fork 무수정(import 외) 동치성 검증이고 layerwise forward 작성이 아님 — Step 2 진입 전 "fork 동치성 검증"으로 정정 (§13 v10).

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

### 8.1 3머신 구조

- **MacBook M2** (Claude Code 실행): 코드 작성/편집/git, vast.ai SSH/API 트리거
- **vast.ai A100-SXM4 80GB** (primary 실험): 모든 step 실험 실행. **step별 신규 할당** — 영구 임대 ❌.
- **사용자 로컬 A100 80GB** (occasional 검증): 사용자가 "꼭 필요한 검증"이라 판단한 step만 별도 실행

vast.ai 인스턴스는 step 시작 시 새로 할당하고 step 종료 시 destroy한다 (§8.4). 이전 가정(장기 임대된 단일 인스턴스를 ping으로 깨워 쓴다)은 폐기.

### 8.2 SSH 자동화 규칙

Claude Code(MacBook)가 vast.ai에 SSH로 직접 명령 트리거. 사용자 중간 개입 없음.

규칙 3개:
1. **SSH alias 고정**: `~/.ssh/config`의 `Host vast` alias 사용. Claude는 `ssh vast '...'` 만 사용. IP/포트 inline ❌. step별 인스턴스 할당 시 Claude가 이 `Host vast` 블록만 추가/갱신한다 (다른 항목 건드리지 ❌, 백업 권장 — §8.4의 `ssh_alias_register`).
2. **인스턴스 확인 먼저, 없으면 자동 할당**: 매 step 시작 시 인스턴스 상태 확인. 인스턴스가 없거나(step별 할당이라 대부분의 경우) 죽어있으면 **Claude가 자동으로 새 인스턴스를 할당**한다 (§8.4). ping(`ssh vast 'nvidia-smi'`)은 할당 후 SSH 연결/GPU 확인용.
3. **시크릿은 인라인 ❌**: SSH 명령 인라인이나 stdout/log에 `HF_TOKEN`, `VAST_API_KEY` 등 시크릿 값 절대 ❌. vast.ai의 `.env`는 인스턴스 셋업 시 한 번 채워두고 그걸 source. `VAST_API_KEY`도 동일 규칙 — `echo $VAST_API_KEY` 같은 명령 금지.

### 8.3 vast.ai 명령·API 사용 정책

- **인스턴스 내부 명령**: vast.ai는 가상 인스턴스라 자유. `rm -rf`, `pip uninstall`, 인스턴스 내 어떤 명령도 사용자 사전 승인 없이 OK.
- **vast.ai API**: Claude가 `VAST_API_KEY`로 vast.ai API/CLI를 자유롭게 사용한다 — 인스턴스 검색·할당·destroy 포함 (§8.4).
- **시크릿**: `VAST_API_KEY`/`HF_TOKEN` 등 시크릿 값은 stdout/log/명령 인라인에 노출 ❌.
- 이전 규정("인스턴스 destroy/stop은 사용자 권한 — Claude가 API 호출 ❌")은 폐기.

### 8.4 인스턴스 lifecycle

vast.ai 인스턴스는 step 단위로 살았다 죽는다. Claude가 전 과정을 자동 관리한다.

- **할당**: step 시작 시 Claude가 A100-SXM4 80GB 인스턴스를 자동 할당. **사용자 승인 게이트 ❌**.
- **destroy**: step 완료(결과 git push 회수 후) 시 Claude가 인스턴스를 destroy. **사용자 승인 게이트 ❌**.
- **destroy 확인**: 사용자가 vast.ai 콘솔에서 별도로 확인 (Claude의 책임 영역 밖).
- **비용 모니터링**: 사용자 책임. Claude는 비용을 신경 쓰지 않는다.
- **안전장치 최소화**: 이전 프로젝트에서 안전 과잉이 비효율을 낳은 경험에 따라, 인스턴스 관리에 사용자 확인 게이트를 두지 않는다.
- 도구: `scripts/vast_helper.py` (Step 0 진입 직전 구현).

### 8.5 결과 회수 워크플로우

vast.ai에서 실험 → `results/step_XX/vastai/summary.json` 작성 → vast.ai에서 git add + commit + push → MacBook에서 git pull로 회수.
scp 사용 ❌ (git 한 군데로 통일).

### 8.6 환경별 결과 분리
- `results/step_XX/vastai/` — 모든 step에서 채워짐
- `results/step_XX/local_a100/` — 사용자가 결정한 step만, 대부분 비어 있을 수 있음
- `results/step_XX/macbook/` — 존재하지 않음 (smoke test 결과만 별도로)

### 8.7 결과 비교 강도
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
보고서 docs/reports/step_XX_report.md 작성 (MacBook에서)
   ↓
git add + commit + push
   ↓
사용자 리뷰 대기
```

### 9.2 매 step 필수 산출물 (compblend2 패턴)
1. **docs/reports/step_XX_report.md** — Markdown 한글, GitHub-native 렌더링, 이모지 badge (✅ ⚠️ ❌ 🔵 ⬜)
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
- `main` 브랜치는 항상 검증 통과한 상태만 유지. Step 작업 중 `main` 직접 commit ❌ — 예외: (a) 정책/규칙 변경 또는 step 진입 전 task 파일 신설·확장(CLAUDE.md·DECISIONS.md·PROGRESS.md·tasks/*.md 등 메타·스펙 문서), (b) Phase 단위 작업(step 번호 없는 환경 셋업).
- 각 step은 `step/step_XX_<short-desc>` 브랜치에서 진행. Sub-step도 같은 step 브랜치에 commit 누적, 별도 브랜치 ❌.
- Step 완료 + 사용자 리뷰 승인 후: 로컬 `git merge --no-ff` 로 `main`에 병합 (merge commit 유지 — git log에 step 단위 그룹 가시화) → tag `step_XX_done` → `git push origin main step_XX_done` → step 브랜치 삭제(로컬+원격). **PR 생성 ❌** (솔로 프로젝트, 로컬 merge + push).
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
- **2026-05-14 v5**: vast.ai 인스턴스 관리 워크플로우 변경 (사용자 결정)
  - §8 인프라 전면 재정리: vast.ai는 **step별 신규 할당** (영구 임대 가정 폐기)
  - §8.1 제목 "3머신 구조 + SSH 자동화" → "3머신 구조", step별 할당 명시
  - §8.2 SSH 규칙 2 변경: 인스턴스 없거나 죽으면 Claude가 자동 할당 (과거: 사용자에게 요청 후 멈춤)
  - §8.3 정정: "파괴적 명령" → "vast.ai 명령·API 사용 정책". Claude가 vast.ai API 자유 사용 (과거: "API 호출 ❌"). 시크릿 값 노출 ❌ 유지·강화 (`VAST_API_KEY` 포함)
  - §8.4 신설: 인스턴스 lifecycle — Claude 자동 할당/destroy, **사용자 승인 게이트 ❌**, destroy 확인은 사용자가 콘솔에서, 비용 모니터링은 사용자 책임, 안전장치 최소화
  - 사용자 선호 인용: "안전 과잉이 비효율" / "destroy 확인은 내가 한다" / "비용 모니터링은 사용자 책임"
  - 구 §8.4~8.6 → §8.5~8.7로 번호 이동 (본문 동일)
  - tasks/phase_00_setup.md: Phase 0 완료 게이트에서 0-B(vast.ai) 항목 제거 — Step 0 시작 직전 진행
  - PROGRESS.md: Phase 0-A / 0-D 완료 표시, Phase 0-B 재분류, 변경 이력 v5
  - scripts/vast_helper.py 신설 (인터페이스 명세 placeholder, Step 0 진입 직전 구현). scripts/vast_run.sh는 SSH 명령 래퍼로 그대로 유지 (기능 겹침 없음)
  - CLAUDE.md §3.1/§3.2/§3.3/§4/§4.3/§11 동기 수정 (DECISIONS.md §8 참조 형태, 규칙 본문은 복사 ❌)
  - patches/lmcache-vllm-cacheblend: LMCache README가 unified diff가 아닌 수동 삽입 코드블록이라 `.md`로 저장 (`.patch` ❌). 관련 경로 참조 수정
  - 정직성 기록: LMCache README 1차 회수에 쓴 WebFetch가 원문에 없는 소제목을 임의 추가 → `gh api`+`curl` raw 교차검증으로 바로잡음
- **2026-05-14 v6**: 세션 종료 표준 요약문 양식 신설 (사용자 결정)
  - CLAUDE.md §5.4 신설: "매 세션 종료 시 표준 요약문" — 세션 종료 신호 시점에 핸드오프용 plain-text 요약을 마지막 응답에 출력
  - 청자는 다음 세션의 Claude, 용도는 (a) 현재 상태 파악 (b) 다음 세션 진입 프롬프트 생성
  - 양식 8개 섹션 (세션 범위 / 완료된 작업 / 미완·보류 / 다음 세션 진입점 / 의사결정 대기 / 정직성 노트 / vast.ai 사용 / 변경된 핵심 파일), brief 보고와 별개로 출력, diff·긴 인용·표 ❌, 사실만
- **2026-05-14 v7**: 보고서 양식 HTML → Markdown 전환 (사용자 결정)
  - §9.2 / CLAUDE.md §5.2 / docs/design/report_style.md: "HTML 한글, Gmail-compatible tables, color badges" → "Markdown 한글, GitHub-native 렌더링, 이모지 badge (✅ ⚠️ ❌ 🔵 ⬜)"
  - docs/design/report_style.md 전체 재작성: HTML/inline CSS/Gmail table 마크업 가이드 제거 → Markdown table 문법 + 이모지 badge 매핑
  - repo 전역 `docs/reports/*.html` 참조 17곳 → `.md`. history/log 3곳(§13 v2 이력, docs/prompts/phase_00_prompt.md 2곳)은 append-only 원칙상 의도적 유지
  - 기존 docs/reports/phase_00_setup_report.html은 변환하지 않고 그대로 둠 (one-off legacy, 시간 비효율)
  - 검토 채널: GitHub에서 직접 (이메일 ❌)
- **2026-05-15 v8**: 브랜치 워크플로우 명확화 (Step 0 진입 시점)
  - §9.6 / CLAUDE.md §7.1: "PR/merge" → 로컬 `git merge --no-ff` (PR 생성 ❌), tag + `git push origin main step_XX_done` + step 브랜치 삭제(로컬+원격) 단계 명시
  - main 직접 commit ❌ 규칙 + 예외 (a) 정책/규칙 변경(메타 문서) (b) Phase 단위 작업, sub-step은 같은 step 브랜치 누적
  - CLAUDE.md §7.2에 `[meta]` commit prefix 정의 (정책/메타 문서·Phase 단위 작업용)
  - Phase 0가 main에 직접 들어간 것은 정책 누락이었으나 이미 반영됨 — revert ❌, Step 0부터 적용
- **2026-05-15 v9**: §7.1/§9.6 main 직접 commit 예외 (a)에 "step 진입 전 task 파일 신설·확장" 명시
  - Step 1 stub → 자체완결 task 파일 확장을 main에 `[meta]` commit하기 위함 — task 파일은 step 브랜치 분기 전에 확장되므로 step 브랜치에 둘 수 없음
  - 기존 예외 (a)는 "정책/규칙 변경 (CLAUDE.md·DECISIONS.md·PROGRESS.md 등)"만 명시 → task 파일 미포함이라 한 줄 보강. CLAUDE.md §7.1 / DECISIONS.md §9.6 동기 수정
- **2026-05-15 v10**: Step 1 명명 정정 — "layerwise" → "fork_equivalence"
  - Step 1에서 한 일은 fork 무수정(import 외) 동치성 검증이고 layerwise forward 작성이 아님. "layerwise"는 "layerwise forward 작성"으로 오독될 소지 → 혼동 방지 위해 정정. 진짜 layerwise forward는 Step 4(CacheBlend)에서 작성 예정 — 이름 충돌 방지 위해 Step 2 진입 전(정정 비용 최소 시점)에 정리.
  - 파일/디렉토리 rename(git mv, history 유지): `tasks/step_01_layerwise_forward.md`→`step_01_fork_equivalence.md`, `tasks/step_01/`→`tasks/step_01_fork_equivalence/`, `run_layerwise_check.py`→`run_fork_equivalence_check.py`, `docs/reports/step_01_layerwise_report.md`→`step_01_fork_equivalence_report.md`. 내부 참조·§6.3 표 Step 1 행 정정.
  - invariant 1.2 JSON key 정정: `1.2_layerwise_hidden_equiv` → `1.2_per_layer_hidden_equiv` (3곳: task 스키마 예시·run script·summary.json). 데이터 값(SHA·boolean) 무변경, key 이름만 정정.
  - 보존: tag `step_01_done`(step 번호 기반, 이름 무관)·commit message history·`results/step_01/` 디렉토리·summary.json 데이터 값·step 브랜치명 `step/step_01_layerwise_forward`(merge 후 삭제될 ref).
- **2026-05-15 v11**: Step 2 진입 전 `[meta]` round — 보류 항목 3건 정리
  - (c) `tasks/step_01_fork_equivalence.md` §결과 저장 형식 예시의 `fork_source` 문구 정정: "(무수정)" → "(import 문 외 byte 무수정)" (Step 1 보고서 §10에서 예고된 항목).
  - (d) CLAUDE.md §4.5 신설(권장): task 파일 확장 시 외부 코드 의존 가정의 사전 확인 (Step 1 fork 단일 파일 import 가정 실패 사례에서 도출).
  - (e) CLAUDE.md §4.5 (같은 §): round 중간 결정 변경 시 초반 기록 재검토 (Step 1 명명 정정 round의 JSON key 보존 목록 부정확 사례에서 도출).
  - 분리: (a) `scripts/vast_helper.py` push `-u` 가드는 §7.2 `[meta]` 정의(코드는 메타·스펙 문서 아님)에 부합 안 함 — Step 2 step 브랜치 첫 commit으로 처리. (b) dph_total 불일치 추적은 다음 vast.ai 실행 시 데이터 추가 수집 후. (f) `[meta]` prefix 정의·분류 검토는 명명 round로 분리.
  - step 브랜치 마지막 commit으로 처리(`[meta]` prefix) — Step 1 파일 대부분이 미merge 상태라 main 직접 작업 불가, Step 1 merge 시 정정된 이름으로 main 반영.
