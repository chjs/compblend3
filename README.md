# compblend3

**HuggingFace Transformers 기반 CacheBlend의 검증 가능한 재구현, 그리고 CompBlend(Gated HKVD + 압축 KV reuse)로 확장.**

## 프로젝트 핵심

- 이전 두 시도(CompBlend-old, compblend2)에서 막혔던 문제: **코드 검증이 어려웠고 CacheBlend의 F1이 논문대로 안 나옴**
- compblend3는 **검증 가능한 구조를 먼저** 만든 후에 기능을 추가한다.

## 빠른 시작

새 세션 / Claude Code 진입 시 다음 순서로 읽으세요:

1. **`GOAL.md`** — 왜 하는가 (1분)
2. **`PROGRESS.md`** — 어디까지 됐는가, 다음 task (1분)
3. **`CLAUDE.md`** — 작업 규칙 (필요시)
4. **`DECISIONS.md`** — 모든 결정사항과 근거 (참조용)
5. **`tasks/step_XX_*.md`** — 현재 step의 구체적 작업 지시

**컨텍스트 리셋 후 새 세션을 시작하는 사용자라면**: `docs/context/session_resume.md` 에 환경별 첫 메시지 템플릿이 있습니다.

## 디렉토리

```
compblend3/
├── README.md
├── GOAL.md
├── DECISIONS.md
├── CLAUDE.md
├── PROGRESS.md
├── .gitignore
├── pyproject.toml
├── docs/
│   ├── context/         # session_resume.md(세션 복원 가이드). project_goal.md·current_status.md는 GOAL.md·PROGRESS.md redirect stub
│   ├── setup/           # macbook_setup.md, vastai_setup.md, local_a100_setup.md, troubleshooting.md
│   ├── reports/         # step_XX_report.html (HTML 한글)
│   ├── prompts/         # step_XX_prompt.md
│   └── design/          # 설계 노트, invariant 정의
├── tasks/               # step_XX_*.md — self-contained 작업 지시
├── src/compblend/       # 실제 구현 (Python package)
├── tests/               # unit + integration tests
├── scripts/             # compare_results.py, vast_run.sh, setup helpers
├── setup/               # install_macbook.sh, install_vastai.sh, install_local.sh
├── results/             # results/step_XX/{vastai,local_a100}/ 결과 분리
├── notes/               # 분석 노트
└── data/                # 데이터셋 manifest
```

## 환경 (3머신 구조)

| 역할 | 머신 | 환경 | 사용 빈도 |
|---|---|---|---|
| **Claude Code 실행** | MacBook Air M2 | macOS ARM64, GPU 없음. 코드 작성/git/SSH 전용. | 항상 |
| **실험 (primary)** | vast.ai A100 80GB | Ubuntu, Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8 | 항상 |
| **추가 검증 (occasional)** | 사용자 로컬 A100 80GB | Ubuntu 24.04, 동일 환경 | 사용자 결정한 step만 |

상세는 `docs/setup/`.

## 워크플로우 (SSH 자동화)

Claude Code(MacBook)가 vast.ai에 SSH로 직접 명령 트리거. 사용자 중간 개입 최소화.

```
MacBook (Claude Code)
  ├─ 코드 작성/편집
  ├─ git push
  ├─ ssh vast 'cd compblend3 && git pull && python tasks/step_XX/...'
  ├─ ssh vast로 결과 git push (vast.ai에서)
  ├─ git pull (결과 회수)
  └─ 보고서 작성 + git push
```

사용자 로컬 A100은 가끔의 검증용. 대부분 step에서는 vastai 단독으로 진행.

## 참조 저장소 (별도 clone, 본 repo에 포함 ❌)

| 저장소 | 역할 |
|---|---|
| [YaoJiayi/CacheBlend](https://github.com/YaoJiayi/CacheBlend) | 논문 매핑 reference (코드 읽기만) |
| [chjs/LMCache fix branch](https://github.com/chjs/LMCache/tree/fix/cacheblend-vllm-v0.17.1-compat) | vLLM/CacheBlend 실행 reference |
| [snu-mllab/KVzip](https://github.com/snu-mllab/KVzip) | 최종 통합 대상 (Phase 7) |
| [MozerWang/Loong](https://github.com/MozerWang/Loong) | 평가 데이터셋 |
| [chjs/CompBlend-old](https://github.com/chjs/CompBlend-old) | 이전 시도 (참고만, 답습 금지) |
| [chjs/compblend2](https://github.com/chjs/compblend2) | 직전 시도 (참고만, 답습 금지) |

## 진행 원칙

1. **Invariant-first** — 모든 기능은 invariant를 먼저 정의하고 구현
2. **Bottom-up gating** — Step N 통과 못 하면 Step N+1 진행 금지
3. **Mandatory reporting** — 매 step 끝에 보고서 + 사용자 리뷰
4. **솔직성** — 의심스러우면 명시, 추측은 "확인 필요" 표시

## 라이선스

(미정 — Phase 8 즈음 결정)
