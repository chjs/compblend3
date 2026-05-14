# docs/context/session_resume.md — 새 세션 컨텍스트 복원 가이드

> 컨텍스트 리셋 후 새 Claude 세션을 시작할 때, 이전 작업 흐름을 정확히 복원하기 위한 가이드.
> **사용자가 읽는 파일**이다. 첫 메시지로 무엇을 보낼지 알려준다.

---

## 0. 왜 이 파일이 필요한가

compblend3는 **컨텍스트 리셋 친화** 설계다. 모든 작업 상태가 파일로 저장된다.
하지만 새 Claude 세션은 그 파일들의 존재를 모른다. 어떻게 읽을지 모른다.

이 파일은 그 빈 칸을 메운다. 새 세션 첫 메시지로 무엇을 보낼지 알려준다.

또한 중요한 한계도 명시: **문서에 적힌 결정사항은 복원되지만, 문서에 적히지 않은 대화 맥락은 복원되지 않는다.**

---

## 1. 시나리오별 첫 메시지

새 세션은 보통 다음 셋 중 하나에서 시작된다:

| 환경 | 특징 | 권장 첫 메시지 |
|---|---|---|
| **Claude Code on MacBook** | git clone된 디렉토리 통째로 접근, ssh로 vast.ai 트리거 | §1.1 |
| **Claude.ai 웹/모바일** | 단일 메시지 인터페이스, web_fetch만 가능 | §1.2 |
| **이전 세션이 끊긴 직후** (Claude Code) | 작업 중간 컨텍스트 부족 | §1.3 |

### 1.1 Claude Code on MacBook (가장 일반적인 경우)

```
compblend3 프로젝트, 새 세션입니다.

다음 순서로 파일을 읽어주세요:
1. GOAL.md
2. PROGRESS.md
3. CLAUDE.md
4. PROGRESS.md의 'Next task file'에 적힌 task 파일

읽은 후 현재 상태를 한 줄로 요약하고 다음 행동을 제안해주세요.
진행 전 제 승인을 기다려주세요.
```

이 한 메시지면 충분하다. CLAUDE.md §0이 모든 후속 순서를 안내한다.

> vast.ai 인스턴스는 step 시작 시 Claude가 `scripts/vast_helper.py`로 자동 할당한다
> (DECISIONS.md §8.4). 세션 시작 시점에 인스턴스 존재 여부를 확인할 필요 없음 —
> step에 진입할 때 없으면 새로 할당된다.

### 1.2 Claude.ai 웹/모바일 새 채팅

이 환경에서는 `git clone`이 안 되고 `web_fetch`만 가능하다.
사용자가 GitHub raw URL을 명시해야 한다.

```
compblend3 프로젝트입니다. https://github.com/chjs/compblend3 의 파일들을 web_fetch로 다음 순서로 읽어주세요:

1. https://raw.githubusercontent.com/chjs/compblend3/main/CLAUDE.md
2. https://raw.githubusercontent.com/chjs/compblend3/main/GOAL.md
3. https://raw.githubusercontent.com/chjs/compblend3/main/PROGRESS.md
4. https://raw.githubusercontent.com/chjs/compblend3/main/DECISIONS.md
5. PROGRESS.md의 'Next task file' 항목에 적힌 task 파일도 같은 방식으로 읽어주세요

읽은 후 현재 상태를 한 줄로 요약하고, 다음 행동을 제안해주세요.
```

브랜치명(main)이 다르면 그에 맞게 URL을 수정한다. 진행 중인 step 브랜치라면 `main` 대신 `step/step_XX_...`.

### 1.3 Claude Code on MacBook 끊김 (컨텍스트 부족으로 강제 종료)

이 경우 PROGRESS.md에 "다음 행동"이 정확히 기록되어 있어야 한다 (CLAUDE.md §4.2 규칙).
step 도중 끊겼고 vast.ai 인스턴스가 아직 destroy되지 않았다면, PROGRESS.md에
인스턴스 정보와 진행 상태가 적혀 있어야 한다.

```
compblend3 프로젝트, 이전 세션이 중간에 끊긴 상태입니다.

먼저 다음 확인해주세요:
1. PROGRESS.md의 '다음 행동' 섹션 — 어디서 멈췄는지
2. git status — MacBook에서 미완 변경사항
3. (step 도중 끊겼고 인스턴스가 살아있을 경우) ssh vast 'cd compblend3 && git status'
   — `ssh vast` 가 실패하면 인스턴스가 이미 없는 것이므로 해당 step을 처음부터 재실행
4. (인스턴스가 살아있을 경우) ssh vast 'tmux list-sessions' — 백그라운드 실험 확인

상황 요약 후 작업 재개 전에 제 승인을 기다려주세요.
```

---

## 2. 컨텍스트 복원의 한계 (정직성 섹션)

### 2.1 복원되는 것

문서에 적힌 모든 것:
- 프로젝트 목표 (GOAL.md)
- 모든 결정사항과 근거 (DECISIONS.md)
- 작업 규칙 (CLAUDE.md)
- 진행 상태 (PROGRESS.md)
- 각 step의 작업 지시 (tasks/*.md)
- 보고서 (docs/reports/*.html)
- 결과 데이터 (results/*/summary.json)

### 2.2 복원되지 않는 것

이전 세션의 **대화 맥락 중 문서로 적어두지 않은 것**:
- 사용자의 선호도 (예: "한 번에 다 만들어라" vs "단계적으로")
- 후보 옵션 중 왜 그것을 골랐는지의 미세한 추론
- "이것도 고민했는데 안 했다"의 부정형 결정
- 사용자의 임시 어조나 우선순위 (오늘은 빨리 진행하고 싶다 등)

→ 새 세션의 Claude가 이런 걸 모를 수 있다. 다시 물어볼 가능성이 있다.
→ 중요하다고 느낀 맥락이 있다면 **그때그때 DECISIONS.md 또는 task 파일에 명시적으로 기록**해야 한다.

### 2.3 환경 차이로 인한 미세한 변동

새 세션의 Claude는:
- 다른 Claude 인스턴스 (model version 동일해도)
- 같은 문서를 읽어도 약간 다른 우선순위로 해석 가능
- 보고서 작성 스타일이 미세하게 다를 수 있음

→ 큰 결정은 동일하게 복원되지만, 사소한 표현 선택은 일관되지 않을 수 있음. 정상이다.

---

## 3. 사용자가 세션 종료 전에 해야 할 일

다음 세션이 깔끔하게 시작되려면 종료 전에:

- [ ] **PROGRESS.md 갱신**: 현재 phase, next step, next task file 정확히
- [ ] **미완 작업의 "다음 행동" 기록**: 어느 함수 작성 중이었는지, 어느 invariant 못 통과했는지
- [ ] **중요한 결정사항이 새로 생겼다면 DECISIONS.md에 추가**
- [ ] **이번 세션에서 새로 학습한 함정이 있다면 troubleshooting.md에 기록**
- [ ] **git commit** (또는 commit할 수 없다면 그 이유를 PROGRESS.md에)

이 5개가 안 되어 있으면 다음 세션이 헤맨다. 문서가 부실하면 컨텍스트 리셋 친화 설계가 작동하지 않는다.

---

## 4. 새 세션의 Claude가 자주 물어볼 만한 것 (미리 답 준비)

새 세션을 시작하면 Claude가 다음을 물을 가능성이 높다. 답을 미리 준비해두면 빠르다.

- "이전 step의 결과를 로컬 A100에서도 재현하셨나요? compare_results.py 결과는?"
- "PROGRESS.md에 적힌 next step으로 바로 진행할까요?"
- "참조 저장소(LMCache, KVzip, Loong)는 어디에 clone되어 있나요?"
- "`.env`에 `VAST_API_KEY` / `HF_TOKEN`이 채워져 있나요?" (인스턴스 자동 할당에 필요)

이 답들이 PROGRESS.md 또는 task 파일에 있으면 묻지 않는다.

> vast.ai 인스턴스 존재 여부·`Host vast` alias 등록은 Claude가 step 시작 시
> `scripts/vast_helper.py`로 자동 처리하므로 더 이상 묻지 않는다 (DECISIONS.md §8.4).

---

## 5. 비상 시: 모든 게 꼬였을 때

이전 세션의 작업이 어떻게 됐는지 정말 모르겠다면:

```
compblend3 프로젝트입니다. 작업 상태가 불명확합니다.

다음을 순서대로 해주세요:
1. git log --oneline -20 으로 최근 commit 확인
2. git status 로 미완 변경사항 확인
3. PROGRESS.md 와 가장 최근 step의 docs/reports/step_XX_report.html 비교
4. 차이가 있다면 어느 쪽이 맞는지 추정하고 저에게 알려주세요

확정 전까지 어떤 파일도 수정하지 마세요.
```

이 모드는 "read-only로 상태 진단"이다. Claude가 임의로 복구를 시도하지 않게 막는다.

---

## 6. 이 파일 자체의 유지

이 파일도 시간이 지나면 outdated 될 수 있다.
- 새 브랜치 명명 규칙이 바뀌면 §1.2의 URL 예시 갱신
- 새 시나리오가 생기면 §1에 추가
- 자주 발생하는 새 세션 질문이 있으면 §4에 추가

이 파일의 변경은 다른 메타 문서보다 자유롭게 해도 된다 — 더 정확한 가이드가 되는 방향이라면.
