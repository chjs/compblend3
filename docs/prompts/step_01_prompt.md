# Step 1 — 세션 프롬프트 기록

이 파일은 Step 1 (Our layerwise forward = HF 표준 forward) 작업의 시작 프롬프트와
진행 중 추가 지시를 기록한다. 세션: 2026-05-15, MacBook (Claude Code).

---

## 시작 상황

Step 0 완료(merge·tag `step_00_done`) 후 Step 1 진입. `tasks/step_01_layerwise_forward.md`는
stub 상태였음 — 본격 작업 전 자체완결 task 파일로 확장이 필요했다.

## 진행 중 결정·지시 (시간순)

1. **Step 1 해석 확정 (A)**: stub이 (A) HF 코드 무수정 fork인지 (B) forward 루프 교체인지
   GOAL/PROGRESS/DECISIONS/stub 본문에서 근거 조사 → stub line 12 "코드를 그대로
   가져오기만 한다 — CacheBlend 로직은 아직 안 들어감"이 결정적 → **(A) 무수정 fork** 확정.

2. **확장본 작성 + 검증 계측 = 옵션 1**: stub → 자체완결 task 파일 확장. 검증 계측은
   옵션 1(외부 forward hook, fork 코드 무수정). 순차 로드 기본. 확장본을 main에
   `[meta]` commit (§7.1 예외 (a)에 "step 진입 전 task 파일 신설·확장" 추가 — 미포함이라 보강).

3. **fork 단일 파일 가정 오류 → import 12줄 변환 결정**: 확장본 §fork는 원본이
   `from transformers...` 절대 import라 가정했으나 — 실제로는 전부 상대 import(`from ...X`).
   verbatim fork는 `src/compblend/modeling/`에서 import 불가. 사용자 승인:
   **import 12줄만 상대→절대 변환, forward 본문 1089줄 byte 무수정.** "Step 1 원칙"을
   "import 문 외 byte 무수정. import은 상대→절대 경로 변환만 허용, 추가/제거 ❌"로 정정.

4. **smoke test — meta device**: smoke test 4 `MistralForCausalLM(cfg)`는 7B 파라미터를
   실제 할당(~28GB)해 MacBook OOM 위험 → `torch.device("meta")`로 메모리 할당 없이
   `__init__` 경로만 검증 (검증 목적 동일 충족).

5. **1.3 비교 방식 = torch.equal 확정**: 체크리스트 item e 문구가 "SHA-256 ... q+k+v"로도
   읽혔으나, 확장본 §검증 계측 명세대로 **1.3은 `torch.equal`** 유지 (summary.json 1.3
   schema에 hash 필드 없음과도 일치).

6. **vast.ai 실행**: instance 36787683 할당 → 셋업 → `run_layerwise_check.py` 실행 →
   invariant 1.1·1.2·1.3 모두 PASS → 결과 회수 → destroy. 가동 ≈9.3분.
   과정 중 `git push -u` 누락 발견·정정, dph_total 표시 불일치($1.007 vs $1.139) 관찰.

## 산출물

- `src/compblend/modeling/{modeling_mistral.py, __init__.py, FORK_HASH.txt}` (fork)
- `tasks/step_01/run_layerwise_check.py` (검증 스크립트)
- `results/step_01/vastai/summary.json` (all_invariants_passed: true)
- `docs/reports/step_01_layerwise_report.md`, `docs/prompts/step_01_prompt.md`
- `tasks/step_01_layerwise_forward.md` (stub → 자체완결 확장)
- commit (step 브랜치): `4dcb1aa`, `494c13e`, `b4f49da`, + 본 보고서 commit
  (main `[meta]`: `a5b34b1`)

## 별도 round 보류 항목

- `vast_helper`에 step 브랜치 push `-u` 가드 추가
- dph_total 표시 불일치 원인 추적
- 확장본 §결과 저장 형식 예시의 `fork_source` 문구 정정 ([meta])
- CLAUDE.md "외부 코드 의존 가정은 명세 commit 전 사전 확인" 메모
