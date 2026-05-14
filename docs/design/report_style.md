# docs/design/report_style.md — 보고서 Markdown 스타일 가이드

`docs/reports/*.md` 작성 시 따를 스타일 규칙. GitHub에서 직접 렌더링되어 검토된다.

> **양식 정책 (2026-05-14 전환, DECISIONS.md §13 v7)**: 보고서는 Markdown으로 작성한다.
> 검토는 GitHub에서 한다 (이메일 ❌). 기존 `docs/reports/phase_00_setup_report.html`은
> 변환하지 않고 그대로 둔다 (one-off legacy).

---

## 1. 필수 요소

- **언어**: 한글 (외부 라이브러리명, 파일 경로, tensor shape 표기는 원문)
- **포맷**: Markdown — GitHub-native 렌더링. 외부 CSS / 원시 HTML / JavaScript ❌
- **Heading 계층**: `#` 제목 1개, `##` 섹션, `###` 하위 (semantic 계층 유지)
- **Table-heavy**: 수정 파일, tensor shape, test 결과, risk 등은 모두 Markdown table
- **이모지 badge**: 상태 표시는 이모지로 (아래 §3)

---

## 2. Markdown table

GitHub-native table 문법을 쓴다. 정렬이 필요하면 헤더 구분선의 콜론으로.

```markdown
| 파일 | 변경 내용 | 이유 |
|---|---|---|
| `tasks/step_00/run_determinism_check.py` | 신규 | Step 0 결정론 검증 |
| `src/compblend/config.py` | `recompute_ratio` 추가 | Step 6 전제 |
```

- 코드 심볼·파일 경로·tensor shape는 백틱으로 감싼다.
- 셀 안에서 줄바꿈이 필요하면 내용을 쪼개거나 별도 문단으로 (원시 `<br>` ❌).
- 열이 많아 넓어지면 열 수를 줄이거나 섹션을 나눈다. 가로 스크롤 가정 ❌.

---

## 3. 상태 badge (이모지)

| 의미 | 이모지 | 사용처 |
|---|---|---|
| PASS / OK | ✅ | invariant 통과, test 성공 |
| FAIL / BAD | ❌ | invariant 실패, test 실패 |
| RISK / WARN | ⚠️ | 알려진 한계, 의심스러운 부분 |
| INFO / NOTE | 🔵 | 참고, 환경 의존 정보, 진행 중 |
| 대기 | ⬜ | 상태 표기 시 |

table 셀이나 문장 안에 그대로 쓴다. 예: `| 0.1 | 같은 입력 3회 → SHA-256 동일 | ✅ |`

---

## 4. 보고서 표준 섹션 순서

1. **요약** (1~2 문단, 가장 위에)
2. **목표와 통과 기준** (이번 step의 invariant)
3. **수정 파일** (table)
4. **Tensor shape 명세** (table, 해당 시)
5. **구현 핵심** (RoPE, attention masking, recompute logic 중 해당하는 것)
6. **Unit test 결과** (table with ✅ / ❌ badge)
7. **결과 데이터** (summary.json 핵심 발췌)
8. **환경 간 비교** (vastai vs local, 해당 시)
9. **알려진 한계 / 의심스러운 부분** ← 정직성 섹션
10. **다음 step**

---

## 5. 절대 하지 말 것

- 보고서 본문을 코드 블록 안에 통째로 덤프 ❌
- 원시 HTML / 외부 CSS / JavaScript ❌ (순수 Markdown만)
- 영문으로 본문 작성 ❌ (한글 정책)
- "잘 됐다고 생각합니다" 같은 주관적 표현 ❌ (objective metric으로)

---

## 6. 자동화 (선택)

`scripts/make_report.py`로 summary.json → Markdown 자동 변환도 가능하지만,
처음에는 수동으로 작성하며 패턴을 확립한 후 자동화한다. premature abstraction 금지.
