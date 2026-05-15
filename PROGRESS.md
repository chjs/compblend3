# PROGRESS.md — compblend3 진행 상태

> 매 step 완료 시 업데이트한다.
> 새 세션 진입 시 두 번째로 읽는 파일 (GOAL.md 다음).

---

## 현재 상태

**Phase**: Phase 1 (Step 0~3) — Step 1 실험 통과, 사용자 리뷰 대기
**완료**: Phase 0 전체 ✅ + Step 0 (결정론, tag `step_00_done`) ✅ + Step 1 (fork 동치성 검증) invariant 1.1/1.2/1.3 통과 ✅
**Next**: Step 1 사용자 리뷰 → 승인 시 `main`에 `--no-ff` merge + tag `step_01_done` + 브랜치 삭제 → Step 2 진입
**Next task file**: `tasks/step_02_dynamic_cache.md` (현재 stub — 진입 전 확장 필요)
**Branch**: `step/step_01_layerwise_forward` (리뷰 승인 후 merge·삭제)

---

## Phase 진행도

| Phase | 상태 | 비고 |
|---|---|---|
| **Phase 0** — 환경 셋업 | ✅ 완료 | MacBook venv ✅ + LMCache pinning ✅ + 사용자 리뷰 승인 (2026-05-14). vast.ai 셋업은 Step 0 작업 중 (Phase 0-B) |
| Phase 1 — Step 0~3 (HF forward + cache) | 🔵 진행 중 | Step 0 진행 중 |
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
| 1 | fork 동치성 검증 (fork된 코드 = HF 표준, no cache) | 🔵 | step_01_fork_equivalence_report.md | invariant 1.1/1.2/1.3 ✅, 리뷰 대기 |
| 2 | HF DynamicCache forward = no-cache forward | ⬜ | - | - |
| 3 | ChunkedKVStore 정확성 | ⬜ | - | - |
| 4 | N chunks 따로 prefill → concat = vanilla | ⬜ | - | - |
| 5 | 1 chunk reuse = vanilla | ⬜ | - | - |
| 6 | N chunks reuse, recompute_ratio=1.0 = vanilla | ⬜ | - | - |
| 7 | HKVD oracle 일치 | ⬜ | - | - |
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

Step 1 실험 통과 (invariant 1.1/1.2/1.3 ✅), `step/step_01_layerwise_forward` 브랜치에 commit 완료. 남은 것:

1. **사용자 리뷰** — `docs/reports/step_01_fork_equivalence_report.md` 검토
2. 승인 시: `git checkout main && git merge --no-ff step/step_01_layerwise_forward` → `git tag step_01_done` → `git push origin main step_01_done` → step 브랜치 삭제(로컬+원격)
3. Step 2 진입 — `tasks/step_02_dynamic_cache.md`는 현재 stub(30줄), 진입 전 자체완결 task 파일로 확장 필요

로컬 A100 교차검증(invariant 0.3)은 미수행 — 사용자 결정 시 별도 round.

별도 round 보류: `vast_helper` push `-u` 가드 / dph_total 불일치 추적 / 확장본 `fork_source` 예시 정정 `[meta]` / CLAUDE.md "외부 코드 의존 사전 확인" 메모.

## 다음 세션 첫 행동

- Step 1 사용자 리뷰 대기. 승인 시 `main`에 `--no-ff` merge + tag `step_01_done` + 브랜치 삭제 → Step 2 (task 파일 stub 확장부터)

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

(이후 매 step 완료 시 여기에 한 줄씩 추가)
