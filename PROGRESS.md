# PROGRESS.md — compblend3 진행 상태

> 매 step 완료 시 업데이트한다.
> 새 세션 진입 시 두 번째로 읽는 파일 (GOAL.md 다음).

---

## 현재 상태

**Phase**: Phase 0 (환경 셋업) — 거의 완료
**완료**: Phase 0-A (MacBook 셋업) ✅ + Phase 0-D (LMCache pinning) ✅
**Next**: Phase 0 사용자 리뷰 → 승인 시 Step 0 진입. Phase 0-B (vast.ai 셋업)는 Step 0 시작 직전 `scripts/vast_helper.py`로 자동 수행 (Phase 0 게이트 제외).
**Next task file**: `tasks/step_00_determinism_check.md`
**Branch**: 미정 (Step 0 진입 시 `step/step_00_determinism_check` 생성)

---

## Phase 진행도

| Phase | 상태 | 비고 |
|---|---|---|
| **Phase 0** — 환경 셋업 | 🔵 **IN PROGRESS** | MacBook venv ✅ + LMCache pinning ✅. vast.ai 셋업·모델 다운로드·Loong manifest는 Step 0 시작 시 (Phase 0-B) |
| Phase 1 — Step 0~3 (HF forward + cache) | ⬜ 대기 | |
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
| 0 | HF eager forward 결정론 | ⬜ | - | - |
| 1 | Our layerwise forward = HF 표준 (no cache) | ⬜ | - | - |
| 2 | HF DynamicCache forward = no-cache forward | ⬜ | - | - |
| 3 | ChunkedKVStore 정확성 | ⬜ | - | - |
| 4 | N chunks 따로 prefill → concat = vanilla | ⬜ | - | - |
| 5 | 1 chunk reuse = vanilla | ⬜ | - | - |
| 6 | N chunks reuse, recompute_ratio=1.0 = vanilla | ⬜ | - | - |
| 7 | HKVD oracle 일치 | ⬜ | - | - |
| 8 | Loong F1 측정 (Mistral) | ⬜ | - | - |

---

## Phase 0 세부 task (현재 진행 중)

Phase 0는 step 번호가 붙지 않은 환경 셋업. 3머신을 차례로 셋업.

### Phase 0-A: MacBook (Claude Code 실행 환경)

| Task | 상태 | 비고 |
|---|---|---|
| 0-A.1 — uv venv 생성, Python 3.10, transformers 설치 | ✅ | `setup/install_macbook.sh` 실행 완료 (torch 2.10.0 CPU, transformers 4.51.3) |
| 0-A.2 — `~/.ssh/config`에 `Host vast` alias 정의 | ⬜ | Step 0 시작 시 `scripts/vast_helper.py`가 인스턴스 할당 후 자동 등록 |
| 0-A.3 — `ssh vast 'echo hello'` 로 alias 동작 확인 | ⬜ | Step 0 시작 시 Phase 0-B에서 확인 |
| 0-A.4 — `python scripts/check_env.py` (macbook 모드) | ✅ | `results/phase_00/macbook/env_check.json` all_ok=true, FAIL 0건 |

### Phase 0-B: vast.ai 인스턴스 (primary 실험 환경) — Step 0 시작 직전 진행, Phase 0 게이트 제외

| Task | 상태 | 비고 |
|---|---|---|
| 0-B.1 — 인스턴스 생성 (A100-SXM4 80GB) | ⬜ | 사용자 결정 |
| 0-B.2 — `setup/install_vastai.sh` 실행 (Claude가 ssh로) | ⬜ | uv venv, torch 2.10+cu128, transformers |
| 0-B.3 — `.env`에 HF_TOKEN 채움 (사용자가 직접) | ⬜ | 시크릿이라 ssh inline ❌ |
| 0-B.4 — Mistral-7B-Instruct-v0.2 다운로드 (Claude가 ssh로) | ⬜ | `scripts/download_models.py` |
| 0-B.5 — Loong clone + manifest 생성 (Claude가 ssh로) | ⬜ | `scripts/build_loong_manifest.py` |
| 0-B.6 — `scripts/check_env.py` 실행 (vastai 모드) | ⬜ | 전부 OK 기대 |
| 0-B.7 — `scripts/sanity_forward.py` 실행 | ⬜ | top1 token이 합리적인지 |

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
- [ ] 사용자 리뷰 승인

vast.ai 환경 검증 / sanity_forward / Loong manifest는 Phase 0 게이트가 아니라 **Step 0 시작 직전 Phase 0-B**에서 확인 (DECISIONS.md §8.4 — 인스턴스 step별 할당).
로컬 A100 셋업도 Phase 0 게이트의 일부가 아님. Step 0 시작 직전에 셋업 권장.

---

## 다음 행동 (Next actions)

Phase 0-A (MacBook 셋업) + Phase 0-D (LMCache pinning) + Phase 0-A 보고서 작성 완료. 남은 것:

1. **사용자 리뷰** → Phase 0 완료 게이트 승인 대기
2. 승인 시 Step 0 진입. **Step 0 시작 시 Phase 0-B를 `scripts/vast_helper.py`로 자동 수행**:
   - 인스턴스 할당 (A100-SXM4 80GB) → `~/.ssh/config` `Host vast` 자동 등록
   - `setup/install_vastai.sh` 실행 → `.env`에 HF_TOKEN 전송
   - `check_env.py` (vastai) / 모델 다운로드 / Loong manifest / `sanity_forward`
   - Step 0 실험 → 결과 git push → 인스턴스 destroy

Phase 0-C (로컬 A100 셋업)는 사용자가 별도로 진행. Phase 0 완료 게이트의 일부 ❌, Step 0 진입 직전에 권장.

## 다음 세션 첫 행동

- Phase 0 리뷰 완료 후 Step 0 진입 (`scripts/vast_helper.py` 구현 + 첫 vast.ai 인스턴스 할당부터)

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

(이후 매 step 완료 시 여기에 한 줄씩 추가)
