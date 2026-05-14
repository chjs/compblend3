# Step 0 — 세션 프롬프트 기록

이 파일은 Step 0 (HF eager forward 결정론 확인) 작업 세션의 시작 프롬프트와 진행 중 추가 지시를 기록한다.
세션: 2026-05-14~15, MacBook (Claude Code).

---

## 세션 시작 프롬프트 (요지)

Phase 0 사용자 리뷰 완료 + 사전 점검(보고서 검토, vast.ai 잔액/A100 가용성, HF Mistral access) 통과.
Phase 0 완료 게이트 승인 → Step 0 진입.

CLAUDE.md §0 순서대로 필수 reading 후 (a) Step 0 현재 상태 (b) 첫 행동 제안 (c) 확인 필요 사항 보고.
진행 전 사용자 승인 대기. 작업 언어 한글.

## 진행 중 추가 지시 (시간순)

1. **브랜치 워크플로우 적용**: Phase 0까지 누락됐던 §7.1/§9.6 브랜치 규칙을 Step 0부터 적용.
   step별 브랜치 분기 → 작업 → 리뷰 승인 후 로컬 `git merge --no-ff` + tag + 브랜치 삭제, PR ❌.
   §7.1/§9.6에 `--no-ff`·"PR ❌"가 명확하지 않으면 보강 (main 직접 commit, 정책 예외).

2. **5개 사전 질문 답변**: vastai 인증 = `vastai set api-key`, HF cache = 매번 재다운로드,
   GitHub 인증 = PAT+HTTPS(.env 통째 전송), 브랜치 워크플로우 = 보강 승인,
   로컬 A100 검증 = vast.ai 단독 진행 (환경 간 비교는 별도 round).

3. **3-1~3-5 즉시 진행**: PROGRESS/CLAUDE/DECISIONS 보강 → main에 2 commit + push →
   `step/step_00_determinism_check` 브랜치 분기 → `vast_helper.py` 4개 함수 한 번에 구현.
   메타 정리 round 금지, "step 진행 막는가"만 묻고 안 막으면 메모 후 진행.

4. **첫 인스턴스 할당 승인**: step 브랜치 push + `python scripts/vast_helper.py` 실행.
   막히면 인스턴스 즉시 destroy 후 보고, 5분 이상 진척 없으면 멈추고 보고,
   매 단계 결과를 PROGRESS.md에 기록.

5. **디스크 부족 디버깅**: 첫 인스턴스가 disk 작은 offer에 잡혀 install 실패 →
   `disk_space>=100` 필터 + 디스크 가드 수정 후 재할당 승인.

6. **메모리 지시**: "다음부턴 디스크 가용공간 미리 잘 체크하고 일을 시작하도록 기억해줘" →
   memory `precheck-disk-before-long-ops` 저장.

7. **잔존 인스턴스 확인**: 36764141이 첫 destroy(rc=0)에도 살아있음 발견 →
   `vastai destroy instance`가 대화형([y/N])임을 확인, stdin "y" 전달로 실제 destroy.
   `destroy_instance()`에 대화형 처리 + 검증 추가.

## 산출물

- `tasks/step_00/run_determinism_check.py` (신규)
- `scripts/vast_helper.py` (placeholder → 구현 + disk/destroy 수정)
- `results/step_00/vastai/summary.json`, `results/phase_00/vastai/env_check.json`
- `docs/reports/step_00_determinism_report.md`, `docs/prompts/step_00_prompt.md`
- memory: `precheck-disk-before-long-ops`
- commit: `009e1b6`, `142d46d`, `e39dd54`, `239e53d`, `4fccf0d`, `243deda`, + 본 보고서 commit
  (main: `2e1e69d`, `a7f465f`)
