# tasks/

각 작업 단위(Step)의 self-contained 작업 지시서.

## 명명 규칙

- `phase_00_setup.md` — Phase 0 환경 셋업 (step 번호 없음)
- `step_00_determinism_check.md` — Step 0
- `step_01_layerwise_forward.md` — Step 1
- ...

Sub-task 가 필요한 경우:
- `step_04a_rope_rerotation.md`
- `step_04b_n_chunk_concat.md`

## 각 task 파일의 필수 섹션

1. **목표** — 이번 step이 왜 필요한가
2. **사전 조건** — 이 step 진입 전에 만족되어야 할 것 (이전 step 완료 등)
3. **통과 기준 / Invariants** — 명확한 자동 검증 가능 명제로
4. **구현 사양** — 어떤 파일을 만들/수정하나, 함수 시그니처
5. **실행 방법** — 명령어 (vast.ai + 로컬 양쪽)
6. **결과 저장 형식** — `results/step_XX/{env}/summary.json` 의 정확한 schema
7. **보고서 작성 가이드** — `docs/reports/step_XX_report.html`에 포함할 내용
8. **다음 step 게이트** — 통과 조건 + 사용자 리뷰 요청 형식

## 진행 순서

```
phase_00_setup.md     ← 현재 위치
   ↓
step_00_determinism_check.md
   ↓
step_01_layerwise_forward.md
   ↓
step_02_dynamic_cache.md
   ↓
step_03_chunked_kv_store.md
   ↓
step_04_multi_chunk_concat.md
   ↓
step_05_one_chunk_reuse.md
   ↓
step_06_n_chunks_reuse_full_recompute.md
   ↓
step_07_hkvd_oracle.md
   ↓
step_08_loong_f1_mistral.md
```

이후 Phase 5~8은 진행 중에 task 파일 추가.
