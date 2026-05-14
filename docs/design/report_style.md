# docs/design/report_style.md — 보고서 HTML 스타일 가이드

`docs/reports/*.html` 작성 시 따를 스타일 규칙. compblend2 패턴 계승.

---

## 1. 필수 요소

- **언어**: 한글 (외부 라이브러리명, 파일 경로, tensor shape 표기는 원문)
- **Semantic HTML**: `<h1>`, `<h2>`, `<section>` 계층 구조
- **Table-heavy**: 수정 파일, tensor shape, test 결과, risk 등은 모두 `<table>`
- **Inline CSS**: Gmail 호환 (외부 stylesheet 금지)
- **Color badges**: PASS / FAIL / RISK / INFO

---

## 2. 권장 컨테이너 CSS

```css
body {
  max-width: 940px;
  margin: 24px auto;
  padding: 0 16px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
  color: #1f2937;
}
h1 { font-size: 24px; margin-top: 0; }
h2 { margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; font-size: 20px; }
h3 { font-size: 16px; margin-top: 20px; }
pre {
  background: #f5f5f5;
  padding: 10px 12px;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 12.5px;
}
.section {
  margin: 28px 0;
  padding: 20px;
  border: 1px solid #d1d5db;
  border-radius: 10px;
  background: #ffffff;
}
.callout {
  background: #fef3c7;
  border-left: 4px solid #f59e0b;
  padding: 10px 14px;
  margin: 12px 0;
  border-radius: 4px;
}
```

---

## 3. Gmail 호환 table 마크업

`<table>`, `<th>`, `<td>` **각각에** border / background를 inline style 로 작성한다.
Gmail이 `<style>` block의 일부 속성을 무시하기 때문.

```html
<div style="width:100%; overflow-x:auto; margin:16px 0;">
  <table style="
    width:100%;
    border-collapse:collapse;
    border:2px solid #374151;
    background-color:#ffffff;
    font-size:14px;
  ">
    <caption style="caption-side:top; text-align:left; font-weight:700; margin-bottom:8px; color:#111827;">
      수정 파일 목록
    </caption>
    <thead>
      <tr>
        <th style="border:1px solid #374151; background-color:#e5e7eb; padding:10px 12px; text-align:left;">파일</th>
        <th style="border:1px solid #374151; background-color:#e5e7eb; padding:10px 12px; text-align:left;">변경 내용</th>
        <th style="border:1px solid #374151; background-color:#e5e7eb; padding:10px 12px; text-align:left;">이유</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border:1px solid #6b7280; background-color:#ffffff; padding:10px 12px; font-family:Consolas,monospace;">tasks/step_00/run_determinism_check.py</td>
        <td style="border:1px solid #6b7280; background-color:#ffffff; padding:10px 12px;">신규</td>
        <td style="border:1px solid #6b7280; background-color:#ffffff; padding:10px 12px;">Step 0 결정론 검증</td>
      </tr>
    </tbody>
  </table>
</div>
```

---

## 4. 상태 badge

```html
<span style="
  display:inline-block;
  padding:2px 8px;
  border:1px solid #065f46;
  background-color:#d1fae5;
  color:#065f46;
  font-weight:700;
  border-radius:999px;
  font-size:12px;
">PASS</span>
```

색 팔레트:

| 의미 | border | background | color |
|---|---|---|---|
| PASS / OK | `#065f46` | `#d1fae5` | `#065f46` |
| FAIL / BAD | `#991b1b` | `#fee2e2` | `#991b1b` |
| RISK / WARN | `#92400e` | `#fef3c7` | `#92400e` |
| INFO / NOTE | `#1e3a8a` | `#dbeafe` | `#1e3a8a` |

---

## 5. 보고서 표준 섹션 순서

1. **요약** (1~2 문단, 가장 위에)
2. **목표와 통과 기준** (이번 step의 invariant)
3. **수정 파일** (table)
4. **Tensor shape 명세** (table, 해당 시)
5. **구현 핵심** (RoPE, attention masking, recompute logic 중 해당하는 것)
6. **Unit test 결과** (table with PASS/FAIL badge)
7. **결과 데이터** (summary.json 핵심 발췌)
8. **환경 간 비교** (vastai vs local, 해당 시)
9. **알려진 한계 / 의심스러운 부분** ← 정직성 섹션
10. **다음 step**

---

## 6. 절대 하지 말 것

- `<pre>` 안에 보고서 본문 통째로 덤프 ❌
- 외부 CSS / JavaScript 참조 ❌
- 영문으로 본문 작성 (compblend2 step 003a 이후 정책) ❌
- "잘 됐다고 생각합니다" 같은 주관적 표현 ❌ (objective metric으로)

---

## 7. 자동화 (선택)

`scripts/make_report.py` 를 만들어 summary.json → HTML 자동 변환도 가능하지만,
처음에는 수동으로 작성하면서 패턴 확립 후 자동화. premature abstraction 금지.
