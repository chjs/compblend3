# PROGRESS.md — compblend3 진행 상태

> 매 step 완료 시 업데이트한다.
> 새 세션 진입 시 두 번째로 읽는 파일 (GOAL.md 다음).

---

## 현재 상태

**Phase**: Phase 1 (Step 0~3) — Step 3 merge·tag 완료. Phase 1 종결. Phase 2 (Step 4~6) 진입 준비
**완료**: Phase 0 전체 ✅ + Step 0 (tag `step_00_done`) ✅ + Step 1 (tag `step_01_done`) ✅ + Step 2 (tag `step_02_done`) ✅ + Step 3 (tag `step_03_done`) ✅
**Next**: Step 4 진입 (작업 0 chunk padding 정책 검증, DECISIONS §13 v13 사전 가정 확정)
**Next task file**: `tasks/step_04_*.md` (현재 stub — 진입 전 확장)
**Branch**: `main` (Step 4 진입 시 step/step_04_* 브랜치 신설)

---

## Phase 진행도

| Phase | 상태 | 비고 |
|---|---|---|
| **Phase 0** — 환경 셋업 | ✅ 완료 | MacBook venv ✅ + LMCache pinning ✅ + 사용자 리뷰 승인 (2026-05-14). vast.ai 셋업은 Step 0 작업 중 (Phase 0-B) |
| Phase 1 — Step 0~3 (HF forward + cache) | 🔵 진행 중 | Step 0·1·2 tag 완료, Step 3 final gate PASS + report 작성 완료, 사용자 리뷰 후 merge/tag 대기 |
| Phase 2 — Step 4~6 (CacheBlend 핵심) | ⬜ 대기 | |
| Phase 3 — Step 7 (HKVD) | ⬜ 대기 | |
| Phase 4 — Step 8 (Mistral F1) | ⬜ 대기 | |
| Phase 5 — Llama-3.1-8B 일반화 | ⬜ 대기 | |
| Phase 6 — flash-attn 2 전환 | ⬜ 대기 | |
| Phase 7 — KVzip 통합 | ⬜ 대기 | |
| Phase 8 — Gated HKVD + 추가 backends | ⬜ 대기 | |

범례: ⬜ 대기 / 🔵 진행 중 / ✅ 완료 / ⚠️ 막힘

---

## Step 진행도 (Phase 1~4)

| Step | 목표 | 상태 | Report | 결과 |
|---|---|---|---|---|
| 0 | HF eager forward 결정론 | ✅ | step_00_determinism_report.md | invariant 0.1/0.2 ✅ (tag step_00_done) |
| 1 | fork 동치성 검증 (fork된 코드 = HF 표준, no cache) | ✅ | step_01_fork_equivalence_report.md | invariant 1.1/1.2/1.3 ✅ (tag step_01_done) |
| 2 | HF DynamicCache forward + padded cache K/V (옵션 B) | ✅ | step_02_dynamic_cache_report.md | invariant 2.1/2.2/2.3A ✅, 2.3B drift 6.20e-06 (gate ❌) (tag step_02_done) |
| 3 | ChunkedKVStore 정확성 + HF Cache 인터페이스 호환성 | ✅ | step_03_chunked_kv_store_report.md | invariant 3.1·3.2·3.3A·3.3B 모두 PASS ✅ (vast.ai max_abs=0.0) (tag step_03_done) |
| 4 | N chunks 따로 prefill → concat (RoPE re-rotation) | ✅ | step_04_multi_chunk_concat_report.md | 4.1 atol 1e-6 / 4.2 bitwise / 4.3 measurement (drift max=8.46) (tag step_04_done) |
| 5 | 1 chunk reuse = vanilla | ✅ | step_05_one_chunk_reuse_report.md | 5.1·5.2·5.3·5.4 모두 bitwise PASS (tag step_05_done) |
| 6 | N chunks reuse, recompute_ratio=1.0 = vanilla | ✅ | step_06_n_chunks_reuse_full_recompute_report.md | API contract + 6.1 bitwise PASS (max_abs=0.0) (tag step_06_done) |
| 7 | HKVD oracle 일치 | ✅ | step_07_hkvd_oracle_report.md | 7.1·7.2·7.3·7.4·7.5 PASS (MacBook CPU, vast.ai 사용 ❌). HKVD formula는 CC 자율 채택, Step 8 진입 전 사용자 검토 권장 (tag step_07_done) |
| 8 | Loong F1 측정 (Mistral) | ⬜ | - | - |

---

## Phase 0 세부 task (완료)

Phase 0는 step 번호가 붙지 않은 환경 셋업. 3머신을 차례로 셋업.

### Phase 0-A: MacBook (Claude Code 실행 환경)

| Task | 상태 | 비고 |
|---|---|---|
| 0-A.1 — uv venv 생성, Python 3.10, transformers 설치 | ✅ | `setup/install_macbook.sh` 실행 완료 (torch 2.10.0 CPU, transformers 4.51.3) |
| 0-A.2 — `~/.ssh/config`에 `Host vast` alias 정의 | ⬜ | Step 0 시작 시 `scripts/vast_helper.py`가 인스턴스 할당 후 자동 등록 |
| 0-A.3 — `ssh vast 'echo hello'` 로 alias 동작 확인 | ⬜ | Step 0 시작 시 Phase 0-B에서 확인 |
| 0-A.4 — `python scripts/check_env.py` (macbook 모드) | ✅ | `results/phase_00/macbook/env_check.json` all_ok=true, FAIL 0건 |

### Phase 0-B: vast.ai 인스턴스 (primary 실험 환경) — Step 0 작업 중 수행됨

| Task | 상태 | 비고 |
|---|---|---|
| 0-B.1 — 인스턴스 생성 (A100-SXM4 80GB) | ✅ | `vast_helper.py` 자동 할당. instance 36769033 (검증 후 destroy) |
| 0-B.2 — `setup/install_vastai.sh` 실행 (Claude가 ssh로) | ✅ | torch 2.10+cu128, transformers 4.51.3 |
| 0-B.3 — `.env` 전송 (HF_TOKEN, GITHUB_PAT) | ✅ | `vast_helper.py`가 화이트리스트 키만 stdin 전송 (VAST_API_KEY 제외) |
| 0-B.4 — Mistral-7B-Instruct-v0.2 다운로드 | ✅ | 27.5GB |
| 0-B.5 — Loong clone + manifest 생성 | ⬜ | Step 0 불필요 — Step 8 진입 시 수행 |
| 0-B.6 — `scripts/check_env.py` 실행 (vastai 모드) | ✅ | 전 항목 통과. `results/phase_00/vastai/env_check.json` |
| 0-B.7 — `scripts/sanity_forward.py` 실행 | ⬜ | 생략 — Step 0의 `run_determinism_check.py`가 실질 forward 검증 |

### Phase 0-C: 로컬 A100 (occasional 검증 환경) — 사용자가 결정한 step에서만

| Task | 상태 | 비고 |
|---|---|---|
| 0-C.1 — `setup/install_local.sh` (사용자가 직접) | ⬜ | 한 번만, 로컬 A100 머신에서 |
| 0-C.2 — `scripts/check_env.py` (local_a100 모드) | ⬜ | 전부 OK 기대 |

Phase 0-C는 Phase 0-A, 0-B와 다르게 **첫 검증이 필요한 step 직전에** 셋업해도 OK.

### Phase 0-D: LMCache reference pinning — Claude가 MacBook에서 web_fetch

| Task | 상태 | 비고 |
|---|---|---|
| 0-D.1 — `chjs/LMCache` branch의 최신 commit SHA 회수 | ✅ | SHA `9f8aa4d6…`, vLLM v0.17.1 SHA `95c0f928…` |
| 0-D.2 — vLLM patch 정확한 형태 회수 (README.md 읽기) | ✅ | `gh api`+`curl` 교차검증. unified diff 아님 → `patches/lmcache-vllm-cacheblend.md` 저장 |
| 0-D.3 — `notes/lmcache_pinning.md` 작성 | ✅ | 모든 SHA + patch 경로 기록 |
| 0-D.4 — commit + push | ✅ | commit `367f418` |

상세는 `tasks/phase_00_setup.md` Phase 0-D 섹션.

### Phase 0 완료 게이트

- [x] MacBook: `results/phase_00/macbook/env_check.json` OK (macbook tag)
- [x] LMCache pinning: `notes/lmcache_pinning.md` + `patches/lmcache-vllm-cacheblend.md` commit됨
- [x] 사용자 리뷰 승인 (2026-05-14)

vast.ai 환경 검증 / sanity_forward / Loong manifest는 Phase 0 게이트가 아니라 **Step 0 시작 직전 Phase 0-B**에서 확인 (DECISIONS.md §8.4 — 인스턴스 step별 할당).
로컬 A100 셋업도 Phase 0 게이트의 일부가 아님. Step 0 시작 직전에 셋업 권장.

---

## 다음 행동 (Next actions)

Step 3 final gate PASS 완료 + 보고서 작성 완료. 남은 것:

1. **사용자 리뷰** — `docs/reports/step_03_chunked_kv_store_report.md` 검토 (GitHub)
2. 승인 시: `git checkout main && git merge --no-ff step/step_03_chunked_kv_store` → `git tag step_03_done` → `git push origin main step_03_done` → step 브랜치 삭제 (로컬+원격)
3. Step 4 진입 — 작업 0 (CacheBlend chunk padding 정책 검증, DECISIONS §13 v13 사전 가정 확정)

별도 round 보류: 다른 진단 스크립트의 RoPE hook 첫 call 캡처 defect.

## 다음 세션 첫 행동

- Step 3 사용자 리뷰 대기. 승인 시 §7.1 절차 (`main --no-ff` merge + tag `step_03_done` + 브랜치 삭제) → Step 4 작업 0 (chunk padding 정책 검증) 진입.

---

## 최근 변경 이력

- **2026-05-14**: 초기 PROGRESS.md 작성. Phase 0 시작 전 상태.
- **2026-05-14 (v4)**: ChatGPT 검토 의견 Tier 1 반영
  - Phase 0-D (LMCache pinning) 신설
  - DECISIONS.md §3.7 (Tokenization Contract), §3.8 (KV Cache Data Model), §11 (KVzip Integration Hypothesis) 추가
  - step_06 task stub 보강 (invariant 6.1/6.2 분리, 전제조건 명시)
  - Mistral v0.2 sliding window 정정
- **2026-05-14 (v5)**: Phase 0-A·0-D 완료 + vast.ai 워크플로우 변경
  - Phase 0-A (MacBook 셋업) 완료: env_check all_ok, FAIL 0건
  - Phase 0-D (LMCache pinning) 완료: SHA 회수, patch는 unified diff 아닌 수동 삽입 코드블록 → `.md`로 저장
  - vast.ai를 step별 신규 할당 방식으로 전환 (Claude 자동 할당/destroy). DECISIONS.md §8 재정리, §8.4 신설
  - Phase 0-B를 Phase 0 게이트에서 분리 (Step 0 시작 직전 진행)
  - `scripts/vast_helper.py` 신설 (인터페이스 명세 placeholder)
- **2026-05-14 (v6)**: 보고서 양식 HTML → Markdown 전환
  - DECISIONS.md §9.2 / CLAUDE.md §5.2 / `docs/design/report_style.md`: HTML+Gmail-compatible tables+color badges → Markdown+GitHub-native 렌더링+이모지 badge (✅ ⚠️ ❌ 🔵 ⬜)
  - repo 전역 `docs/reports/*.html` 참조 17곳 → `.md`. history/log 3곳(DECISIONS.md §13 v2, `docs/prompts/phase_00_prompt.md` 2곳)은 의도적 유지
  - 기존 `docs/reports/phase_00_setup_report.html`은 변환하지 않고 그대로 둠. 검토는 GitHub에서 (이메일 ❌). DECISIONS.md §13 v7 기록
- **2026-05-15 (v7)**: Phase 0 완료 + Step 0 진입
  - Phase 0 완료 게이트 3개 통과 (사용자 리뷰 승인 2026-05-14)
  - 현재 step = Step 0 (HF eager forward 결정론 검증), 브랜치 `step/step_00_determinism_check`
  - CLAUDE.md §7.1 / DECISIONS.md §9.6 브랜치 워크플로우 보강 (--no-ff merge, PR ❌, 브랜치 삭제, main 직접 commit 예외 (a)(b)), `[meta]` commit prefix 정의 — DECISIONS.md §13 v8
- **2026-05-15 (Step 0 진행)**: `scripts/vast_helper.py` 구현(commit `009e1b6`). 첫 vast.ai 인스턴스 할당 시도 — id 36764141 (A100-SXM4 80GB, $0.641/h), 할당·`Host vast` alias·SSH·GPU(driver 570.172.08) 확인 OK. `install_vastai.sh`가 디스크 부족으로 실패(`/` overlay 12GB, torch 2.10+cu128 설치 중 No space left) → 안전 규칙대로 즉시 destroy. 원인 추정: offer의 disk 용량 부족 + `--disk 200` 미반영. 다음: `search offers`에 disk 용량 필터 추가 후 재시도.
- **2026-05-15 (Step 0 완료)**: HF eager forward 결정론 검증 통과. vast.ai A100-SXM4-80GB(instance 36769033)에서 Mistral-7B-Instruct-v0.2 fp32/eager forward — invariant 0.1(같은 입력 3회 SHA-256 동일 `d338ec6a…`)·0.2(다른 입력 다른 SHA `7cc4f432…`) 모두 ✅. 결과 `results/step_00/vastai/summary.json`, 보고서 `docs/reports/step_00_determinism_report.md`. 셋업 중 인프라 이슈 2건 발견·수정: disk 작은 offer(→`disk_space>=100` 필터+가드 commit `e39dd54`), `vastai destroy` 대화형(→stdin "y"+검증 commit `239e53d`). 인스턴스 destroy 완료(잔존 0). local_a100 교차검증 미수행(별도 round). 사용자 리뷰 대기.
- **2026-05-15 (Step 0 merge + `[meta]`)**: `step/step_00_determinism_check` → main `--no-ff` merge(`e801232`), tag `step_00_done`, 브랜치 삭제. `[meta]`: HF_HOME 빈 값 처리(`ef776fc`), Step 1 task 파일 확장 + §7.1/§9.6 예외 (a)에 task 파일 명시(`a5b34b1`, DECISIONS §13 v8·v9).
- **2026-05-15 (Step 1 완료)**: fork 동치성 검증 (fork된 코드 = HF 표준 forward) 통과. transformers 4.51.3 `modeling_mistral.py`를 `src/compblend/modeling/`로 fork(import 12줄 상대→절대, 본문 1089줄 byte 무수정). vast.ai A100-SXM4-80GB(instance 36787683)에서 invariant 1.1(logits SHA-256)·1.2(hidden state 33개 SHA-256)·1.3(q/k/v projection 96개 torch.equal) 모두 ✅. logits SHA `d338ec6a…` — Step 0과 동일. 결과 `results/step_01/vastai/summary.json`, 보고서 `docs/reports/step_01_fork_equivalence_report.md`. commit `4dcb1aa`(fork)·`494c13e`(검증 스크립트)·`b4f49da`(결과). 인스턴스 destroy 완료. 사용자 리뷰 대기.
- **2026-05-15 (Step 1 명명 정정)**: `layerwise` → `fork_equivalence` rename. Step 1은 fork 무수정(import 외) 동치성 검증이고 layerwise forward 작성이 아님 — "layerwise"가 "layerwise forward 작성"으로 오독될 소지가 있어 Step 2 진입 전(정정 비용 최소 시점)에 정리. 진짜 layerwise forward는 Step 4(CacheBlend)에서 작성 예정. 파일/디렉토리 rename(git mv) + 내부 참조 정정. invariant 1.2 JSON key는 `1.2_layerwise_hidden_equiv`→`1.2_per_layer_hidden_equiv`로 정정(데이터 값 무변경). tag `step_01_done`·commit message history·summary.json 데이터 값·step 브랜치명은 보존. DECISIONS.md §13 v10.
- **2026-05-15 (Step 1 merge + Step 2 진입 전 `[meta]` round)**: Step 1 → main `--no-ff` merge(`0cede3f`), tag `step_01_done`, 브랜치 삭제. `[meta]` round 3건: (c) task 파일 `fork_source` 예시 "(무수정)"→"(import 문 외 byte 무수정)" 정정, (d) CLAUDE.md §4.5 권장 "외부 코드 의존 사전 확인" 메모, (e) CLAUDE.md §4.5 권장 "round 중간 결정 변경 시 초반 기록 재검토" 메모. (a) `vast_helper` push `-u` 가드는 §7.2 `[meta]` 정의(코드 미부합) 사유로 Step 2 step 브랜치 첫 commit으로 분리. DECISIONS.md §13 v11.
- **2026-05-15 (Step 2 task 파일 확장)**: stub(30줄) → 자체완결 spec. 작업 0(외부 코드 의존 사전 확인, CLAUDE.md §4.5 (d) 첫 적용)에서 transformers 4.51.3은 legacy Tuple 미지원·`Cache` 서브클래스만 허용 확인 → Step 2 비교 surface 명확. Invariant 2.1(logits SHA-256)·2.2(layer hidden SHA-256)·2.3A(split vs single forward, **bitwise 1차 → atol 1e-6 fallback → 실패 시 보고**). prompt 동일, decode = greedy argmax, `set_all_seeds(42)` 매 forward 직전. DECISIONS.md §13 v12.
- **2026-05-16 (Step 2 진단 round C-3~C-7 + invariant 2.3 옵션 B 채택)**: 초기 실행에서 2.3A FAIL (max_abs=6.20e-06). 5 round 진단으로 원인 식별:
  - **C-3** prefix KV drift 분포 분석 (case=C, layer 1부터 drift). 결과: `artifacts/c3_diagnosis/`, commit `60d3e18`.
  - **C-4** Layer 0 intra-op divergence localization. 결과: 첫 발산 = `02_q_proj` (max_abs=2.384e-06), k_proj/v_proj는 bitwise. `artifacts/c4_layer0_intra_op/`, commit `5270e30`.
  - **C-6** 4 조건 명시 검증 (입력·eager·정밀도·deterministic). 모두 ✅ 통과 + q_proj 발산 재현. `artifacts/c6_input_eager_precision_deterministic/`, commit `fb1a9e5`.
  - **C-7** padded shape + position 정보 검증. **F2 결과: split_padded vs single q_proj first 6 max_abs=0 bitwise** — GEMM input shape (M=6 vs M=7)이 cuBLAS의 "동일 순서" 위반 원인 확정. position_ids·RoPE는 3-way bitwise (정확). `artifacts/c7_padded_shape_position_info/`, commit `36de610`.
  - **결론**: cuBLAS shape-dependent kernel dispatch가 mechanism. atol 완화는 사용자 전제 위배 → 명세를 분리.
  - **옵션 B (2026-05-16 결정)**: 2.3A를 "padded forward(M=7, mask=[1*6,0], use_cache=True) vs single forward(M=7, mask=[1]*7, use_cache=True)의 DynamicCache K/V[:6] bitwise"로 재정의. Gate = `torch.equal`. 2.3B 신설(운영 split forward drift 측정, gate ❌). 둘 다 단일 forward + cache empty 시작 → attention shape `(Q=7, K=7)` 양쪽 동일 → same-shape bitwise 가능. mechanism justification은 task 파일 §2.3A에 명시.
  - **task 파일·script·DECISIONS 갱신**: `tasks/step_02_dynamic_cache.md` §2.3A 옵션 B 정의 + §2.3B 신설 + 의사 코드·Tensor shape·summary.json schema·보고서 가이드·솔직성 노트 갱신. `tasks/step_02_dynamic_cache/run_dynamic_cache_check.py` 재작성 (5 forward: a no-cache / b cache / c padded / d single / e operational split, forward hook ❌, K/V 직접 접근). DECISIONS.md §13 v13 (옵션 B 결정 + CacheBlend chunk padding 사전 가정). 누적 vast.ai 비용은 보고서 §3 환경에서 정산.
- **2026-05-17 (Step 2 실험 + 보고서 작성)**: vast.ai A100-SXM4-80GB (instance `36876915`, dph_total `$1.21/h`, running 228초, 추정 비용 ~$0.16) 에서 옵션 B 검증 — **invariant 2.1·2.2·2.3A 모두 PASS** ✅ (32 layer × K/V `torch.equal` 통과, mismatched=[]). 2.3B drift 측정: `max_abs=6.20e-06`, `argmax_match=True`, `top5_overlap=5/5`, `drift_budget_exceeded=False`. decode 토큰 `5465="Paris"` (Step 0/1 일치). `all_invariants_passed=true`. 결과 commit `84e7d63`. 보고서 `docs/reports/step_02_dynamic_cache_report.md` 작성 (11 섹션) — §6 mechanism interpretation으로 옵션 B의 same-shape (Q=7, K=7) 구조와 prefill cache + padded decode 2-step 구조 (Q=7, K=13)의 구분 명시, §7 operational drift의 mechanism 정량화, §9 CacheBlend chunk padding 사전 메모와의 정합, §10에서 직전 작업 0의 K_total mismatch 분석 정정 명시. 사용자 리뷰 대기. Step 2 누적 vast.ai 비용 ~$1.0 추정 (옵션 A/E + C-3/C-4/C-6/C-7 + 본 실행), 정확 비용은 콘솔 참조.
- **2026-05-17 (Step 2 merge + Step 3 진입)**: Step 2 사용자 리뷰 승인 → main `--no-ff` merge (`71f9c03`), tag `step_02_done`, 브랜치 삭제 (로컬+원격). Step 3 진입 — 작업 0 (사전 확인): HF Cache 인터페이스 surface 직접 확인 (5 abstract methods), modeling_mistral.py 호출 패턴 (`update`/`get_seq_length`/`get_max_cache_shape`/`isinstance(Cache)`), DECISIONS §3.8의 ChunkedKVStore 명세 그대로 채택. 별도 결정 2건: (a) drift_budget Step 3 미적용 (모든 invariant same-shape), (b) chunk padding 정책 Step 4 작업 0으로 이연. Step 3 scope β 채택 — invariant 4개 (3.1 + 3.2 + 3.3A + 3.3B). 해석 A 채택 — Cache 상속 ❌, dataclass-like container. 작업 1·2·3·4 일괄 처리: task 파일 stub → 자체완결 12 섹션 확장, `src/compblend/cache.py` 신규 (ChunkMeta + ChunkedKVStore), `tasks/step_03_chunked_kv_store/run_chunked_kv_store_check.py` 신규, MacBook smoke 3.1·3.2·3.3A PASS. branch `step/step_03_chunked_kv_store` 생성·commit·push (`e2f9c00`).
- **2026-05-17 (Step 3 hygiene + 3.3B vast.ai)**: hygiene commit (`72d1b5b`) — summary.json gate 분리 (`local_smoke_gate_passed`·`step_03_final_gate_passed`·`all_invariants_passed`) + 콘솔 3분기 + 3.3B diagnostic 13 fields + `_classify_3_3b_failure` 자동 판정. 알고리즘 ❌ 수정. MacBook smoke 재실행 PASS. vast.ai A100-SXM4-80GB (instance `36936503`, dph_total `$1.21/h`, running 322초, 추정 비용 ~$0.15) 에서 3.3B 검증 — **invariant 3.1·3.2·3.3A·3.3B 모두 PASS** ✅, `step_03_final_gate_passed=True`, `all_invariants_passed=True`. `logits_a_sha256 == logits_b_sha256 = e581d7f715cffb63...`, `max_abs_diff=0.0`, `mean_abs_diff=0.0`. decode 토큰 `5465="Paris"` (Step 0/1/2 일치). `failure_case=""` (Case 1/2/3 모두 배제). 결과 commit `e9449f0`. 인스턴스 destroy, 잔존 0개.
- **2026-05-17 (Step 3 보고서 작성)**: `docs/reports/step_03_chunked_kv_store_report.md` 작성 (12 섹션) — §1 Summary, §2 Goal/Scope, §3 Environment (vast.ai + MacBook + 누적 비용 추정 Step 0~3 ~$1.36), §4 Implementation overview (해석 A · 5-step 절차 · Tensor shape), §5 Invariants/Gates (4 invariant + 3 gate fields), §6 MacBook smoke, §7 vast.ai 3.3B (diagnostic fields + failure case 자동 판정), §8 Key findings (6건), §9 Mechanism interpretation (왜 3.3B가 bitwise인가 — 3.1 + 결정론 prefill의 derived guarantee), §10 Limitations (11건), §11 Step 4 implications (작업 0 chunk padding 정책 검증), §12 Artifacts/commits/next. 필수 영문 문장 3건 (Cache 상속 ❌ · drift budget 미적용 · RoPE/chunk padding 이연) §4.2 / §1·§8 / §2·§10 배치. summary.json source of truth 재확인 — 직전 보고와 충돌 없음. 사용자 리뷰 대기. `main` merge·tag·브랜치 삭제는 별도 round.
- **2026-05-18 (Step 3 merge·tag·삭제 + Phase 1 종결)**: 사용자 standing approval — overnight round 진입. Step 3 final gate PASS sanity 재확인 (step_03_final_gate_passed=true, logits SHA match, max_abs=0.0, failure_case=""). main `--no-ff` merge (`527fb92`), tag `step_03_done`, step 브랜치 삭제 (로컬+원격). 잔존 vast.ai 0개. **Phase 1 (Step 0~3) 종결**. Step 4 진입 준비.

(이후 매 step 완료 시 여기에 한 줄씩 추가)
