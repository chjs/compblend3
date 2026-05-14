# Phase 0 — 세션 프롬프트 기록

이 파일은 Phase 0 (환경 셋업) 작업 세션의 시작 프롬프트와 진행 중 추가 지시를 기록한다.
세션: 2026-05-14, MacBook (Claude Code).

---

## 세션 시작 프롬프트 (요지)

compblend3 프로젝트 첫 진입 세션. repo는 첫 commit만 된 상태.
CLAUDE.md §0 순서대로 GOAL.md → PROGRESS.md → DECISIONS.md → task 파일을 읽고
프로젝트를 파악한 뒤 (a) 현재 상태 요약 (b) 첫 행동 제안 (c) 진행 전 확인 필요 사항을 보고.

진행 규칙:
- 어떤 능동적 행동(파일 생성/수정, ssh, git commit 등)도 명시적 승인 없이 ❌
- 의심스러우면 추측하지 말고 질문
- 작업 언어 한글

## 진행 중 추가 지시 (시간순)

1. **환경 정보 + vast.ai 정책 정정**: vast.ai 인스턴스는 step별 신규 할당.
   `VAST_API_KEY`가 MacBook `.env`에 있음 → Claude가 vast.ai API 직접 사용 가능.
   인스턴스 할당/destroy를 Claude가 자동 관리, 사용자 승인 게이트 없음, 안전장치 최소화.
   단 `VAST_API_KEY` 값은 stdout/log 노출 ❌.
   이번 세션 범위: Phase 0-A (MacBook 셋업) + Phase 0-D (LMCache pinning)만.
   Phase 0-B는 Step 0 시작 시점, Phase 0-C는 미룸.
   Phase 0-A 마무리 시 워크플로우 변경 패치 5종 제안 요청.

2. **A·B 동시 진행 승인**: A = install_macbook.sh + check_env.py, B = LMCache web_fetch 2건.
   각 단계 명령 실행 전 명시적 승인. 한 번에 여러 단계 묶기 ❌.

3. **Step 1 (LMCache 파일 작성) 세부 지시**: README 원문을 gh api + curl 두 방법으로
   재회수해 비교. unified diff인지 코드블록인지 판단 후 `.patch`/`.md` 결정.
   notes/lmcache_pinning.md + patches/* 작성 후 commit.

4. **Step 3 (워크플로우 패치 제안) 세부 지시**: 5종 패치를 diff만 제안 (실제 수정 ❌).
   DECISIONS.md §8 / tasks/phase_00_setup.md / PROGRESS.md / vast_helper / §13 v5.
   누락 가능성·정직성 노트 포함.

5. **CLAUDE.md 6번째 패치 추가 승인** + vast_helper.py는 Python 확정.
   §8.3 흡수 판단 OK. 6종 diff를 before/after로 작성 보고.

6. **(vi) CLAUDE.md 정정 2건** 후 승인. 나머지 (i)~(v) + (vii) .env.example
   diff 본문 요청. 자신 없는 부분은 원문 인용 + 제안 형태로.

7. **7종 패치 적용 + commit + push** 승인. 이후 보고는 brief 양식
   (한 문단 / 변경 파일 path / commit / 의도와 다른 부분 / 다음 행동 한 줄).

8. **Step 2 (Phase 0-A 보고서) 진행 지시**: report_style.md 따라 HTML 작성,
   prompt 기록, PROGRESS.md 최종 갱신 (Phase 진행도 표 비고 컬럼 정정 포함),
   env_check.json 함께 commit. 한글 인코딩 확인. 구 워크플로우 흔적 검토는
   다음 세션으로 미루고 PROGRESS.md에 메모.

## 산출물

- `results/phase_00/macbook/env_check.json`
- `notes/lmcache_pinning.md`, `patches/lmcache-vllm-cacheblend.md`
- `scripts/vast_helper.py` (placeholder)
- `docs/reports/phase_00_setup_report.html`, `docs/prompts/phase_00_prompt.md`
- 워크플로우 문서 7종 정정 (DECISIONS / CLAUDE / PROGRESS / phase_00_setup / .env.example)
- commit: `367f418`, `319b0f3`, + 본 보고서 commit
