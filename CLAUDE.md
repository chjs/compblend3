# CLAUDE.md — compblend3 Claude Code 작업 가이드

이 파일은 Claude Code의 작업 규칙이다. 매 세션 시작 시 읽는다.

---

## 0. 세션 시작 시 필수 reading 순서

매 세션 첫 행동:

1. **`GOAL.md`** — 우리가 왜 이걸 하는가 (1분)
2. **`PROGRESS.md`** — 어디까지 됐고 다음은 무엇인가 (1분)
3. **현재 step의 `tasks/step_XX_*.md`** — 구체적 작업 지시
4. **이전 step의 `docs/reports/step_(XX-1)_report.html`** — 직전 완료 내용 (필요시)
5. **`DECISIONS.md`** — 결정사항 참조 (필요시)

이 순서를 건너뛰지 않는다. 작업 drift 방지를 위한 닻이다.

**참고**: `docs/context/session_resume.md` 는 사용자가 새 세션을 어떻게 시작해야 하는지 알려주는 파일이다. Claude가 직접 읽을 필요는 없지만, 사용자가 "세션 복원이 잘 안 됐다"고 할 때 가이드로 활용 가능.

---

## 1. 핵심 4원칙

### 원칙 1: Invariant-first development
모든 기능은 invariant를 **먼저 명시적으로 정의한 후** 구현한다.
- Invariant는 task 파일의 "통과 기준" 섹션에 명세
- Invariant는 unit test로 자동 검증 (deterministic)
- Invariant 통과 못 하면 commit 안 함

### 원칙 2: Bottom-up gating
Step N 통과 못 하면 Step N+1 진행 금지.
- 각 step 끝에 사용자 리뷰 게이트 존재
- 통과 게이트 없이 다음 step으로 넘어가지 않음
- 절대 한 번에 여러 step을 묶어서 진행 ❌

### 원칙 3: Mandatory reporting
매 step 완료 시 다음을 모두 생성:
1. `docs/reports/step_XX_report.html` (HTML 한글, semantic structure, table-heavy)
2. `docs/prompts/step_XX_prompt.md` (step 시작 프롬프트 기록)
3. `PROGRESS.md` 업데이트
4. `results/step_XX/vastai/summary.json` (Claude Code 결과)
5. Git commit

### 원칙 4: 솔직성
- 확신 없는 정보는 "확인 필요"라고 적는다
- Negative result도 그대로 보고 (paper-headline 위해 데이터 굽지 않음)
- 의심스러우면 invariant로 변환해서 검증
- "잘 됐다고 생각합니다"보다 "테스트 X가 통과했습니다" 식의 객관적 표현

---

## 2. 개발 원칙 (compblend2 계승)

- **correctness-first** (성능보다 정확성)
- **minimal file count** (필요한 만큼만)
- **no premature abstraction** (당장 필요 없으면 abstract하지 않음)
- **no unnecessary helper/util files** (`utils.py` 같은 잡동사니 금지)
- **readable tensor flow** (shape 추적이 코드 읽기만으로 가능해야 함)
- **explicit KV shape documentation** (tensor 변수에 shape 주석 필수)
- **deterministic tests** (seed 고정, 환경 차이 명시)

---

## 3. 환경 (요약)

상세는 `docs/setup/`. 핵심 3머신 구조:

### 3.1 머신 역할

| 역할 | 머신 | 환경 |
|---|---|---|
| **Claude Code 실행** (지금 너!) | MacBook Air M2 | macOS, ARM64, GPU 없음. uv venv, Python 3.10, transformers/torch는 smoke test 용도만. |
| **실험 (primary)** | vast.ai A100-SXM4 80GB | Ubuntu, x86_64, Python 3.10, PyTorch 2.10.0+cu128, CUDA 12.8 |
| **추가 검증 (occasional)** | 사용자 로컬 A100 80GB | Ubuntu 24.04, 동일 환경 (vast.ai와 거의 일치) |

Claude Code는 MacBook에서 돌아간다. **GPU 의존 코드는 MacBook에서 실행 ❌, 항상 ssh vast로 전송**.

### 3.2 SSH 자동화 규칙

1. **SSH alias 고정**: `ssh vast '...'` 만 사용. IP/포트 inline ❌. 사용자가 `~/.ssh/config`에 `Host vast` 정의해둠.
2. **인스턴스 상태 ping 먼저**: 매 step 시작 시 `ssh vast 'nvidia-smi'` 가벼운 확인. 죽어있으면 사용자에게 켜달라고 요청 후 멈춤. **자동 spawn 시도 ❌**.
3. **시크릿은 vast.ai의 .env에서만**: SSH 명령 인라인에 `HF_TOKEN=...` 같은 시크릿 절대 ❌.

### 3.3 파괴적 명령

vast.ai는 가상 인스턴스라 자유. `rm -rf`, `pip uninstall` 등 인스턴스 내 어떤 명령도 사용자 사전 승인 없이 OK. 다만 인스턴스 자체의 destroy/stop은 사용자 권한이라 Claude가 vast.ai API 호출 ❌.

### 3.4 결과 회수

vast.ai에서 실험 → vast.ai에서 git add+commit+push → MacBook에서 git pull. scp ❌.

### 3.5 환경 태그

`COMPBLEND_ENV_TAG` 가 결정:
- `macbook` — Claude Code 환경 (smoke test만, 실험 결과 생성 ❌)
- `vastai` — primary 실험
- `local_a100` — occasional 검증

---

## 4. Step 워크플로우

각 step은 다음 흐름. **Claude Code는 MacBook에서 돌고, 실험은 ssh vast로 트리거**.

```
[새 세션 시작 — MacBook의 Claude Code]
   ↓
GOAL.md, PROGRESS.md, tasks/step_XX_*.md 읽기
   ↓
ssh vast 'nvidia-smi' — 인스턴스 살아있는지 ping
   ↓ (살아있음 — 죽어있으면 사용자에게 요청 후 멈춤)
[필요 시 sub-task로 분할]
   ↓
MacBook에서 구현 코드 작성 + unit test 작성
   ↓
git add + commit + push  (MacBook → GitHub)
   ↓
ssh vast 'cd compblend3 && git pull && python tasks/step_XX/run_*.py --out results/step_XX/vastai/'
   ↓
실험 종료 후 vast.ai에서 결과 자동 저장됨
   ↓
ssh vast 'cd compblend3 && git add results docs && git commit -m "[step_XX] results from vastai" && git push'
   ↓
MacBook에서 git pull로 결과 회수
   ↓
MacBook에서 docs/reports/step_XX_report.html 작성
   ↓
docs/prompts/step_XX_prompt.md 작성
   ↓
PROGRESS.md 업데이트
   ↓
git add + commit + push
   ↓
[사용자 리뷰 대기]
   ↓
사용자 승인 후 다음 step 진행
```

### 4.1 Sub-task 분할 기준
한 step이 컨텍스트 한 번으로 끝나지 않을 것 같으면 sub-task로 분할:
- 예: Step 4 → Step 4.1 (RoPE re-rotation), Step 4.2 (N-chunk concat)
- 각 sub-task도 같은 워크플로우 (보고서, prompt 기록, commit)
- Sub-task 분할 시 PROGRESS.md에 명시

### 4.2 컨텍스트 부족 시
- 작업 중 컨텍스트가 부족해지면 **즉시 중단**하고 PROGRESS.md에 정확한 상태 기록
- vast.ai에서 실험 진행 중이면 그 상태도 PROGRESS.md에 ("vast.ai에서 X 실험 진행 중, 결과 미회수")
- 새 세션에서 이어받기 가능하도록 "다음 행동" 명시
- 절반 한 채로 commit 금지

### 4.3 vast.ai 인스턴스가 죽어있을 때
1. 사용자에게 vast.ai 콘솔에서 인스턴스 켜달라고 요청
2. 인스턴스 살아난 후 `ssh vast 'cd compblend3 && git pull'` 로 최신 코드 동기화
3. 그 다음 실험 트리거

### 4.4 사용자 로컬 A100 검증
대부분 step에서는 vast.ai 단독으로 진행. 다음 경우에만 로컬 A100 검증 권장:
- Step 0 (결정론 확인) — 환경 간 SHA-256 비교로 토대 검증
- Step 8 (F1 측정) — 결과의 reproducibility 핵심

다른 step에서는 사용자가 필요하다고 판단하지 않는 한 vast.ai 결과만으로 진행 OK.

---

## 5. 보고서 작성 규칙 (compblend2 계승)

### 5.1 언어
- **한글** (소스코드 주석 포함)
- 함수명, 클래스명, 코드 심볼, 파일 경로, tensor shape 표기는 원문 유지
- 영어 필요 시 한 줄 아래에 한글 보충

### 5.2 HTML 보고서 (`docs/reports/step_XX_report.html`)
- Semantic HTML 구조: `<h1>`, `<h2>`, `<section>`
- Table-heavy (수정 파일, tensor shape, test 결과, risks)
- Inline CSS (Gmail-compatible)
- Color badges: PASS/FAIL/RISK/INFO

권장 CSS와 마크업 패턴은 `docs/design/report_style.md` 참조 (compblend2 패턴).

### 5.3 보고서 필수 섹션
1. **요약** (1-2 문단)
2. **목표와 통과 기준** (이번 step의 invariant)
3. **수정 파일** (table: 파일 / 변경 / 사유)
4. **Tensor shape 명세** (table: 변수 / shape / dtype / 비고)
5. **구현 핵심** (RoPE 처리, attention masking, recompute logic 중 해당하는 것)
6. **Unit test 결과** (table: test / 검증 / 결과 with PASS/FAIL badge)
7. **결과 데이터** (results/step_XX/vastai/summary.json 요약)
8. **알려진 한계 / 의심스러운 부분** (정직성)
9. **다음 step**

---

## 6. 코드 작성 규칙

### 6.1 Tensor shape 명세
모든 tensor 변수에 shape 주석:
```python
# q: (B, H_q, T_new, D)
# k: (B, H_kv, T_new, D)
q = self.q_proj(hidden_states).view(B, T, H_q, D).transpose(1, 2)
```

### 6.2 Assert로 invariant 박기
런타임 invariant는 assert로:
```python
assert q.shape == (B, H_q, T_new, D), f"q shape mismatch: {q.shape}"
assert k.dtype == v.dtype, f"k/v dtype mismatch: {k.dtype} vs {v.dtype}"
```

### 6.3 Magic number 금지
모든 hyperparameter는 config로:
- `recompute_ratio`, `hkvd_top_k`, `chunk_size_limit` 등
- `src/compblend/config.py`에 dataclass로 모음

### 6.4 결정론 보장
실험 스크립트 entry point에서:
```python
import torch, random, numpy as np
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 6.5 No premature abstraction
- "나중에 X도 필요할 수 있으니 추상화"를 하지 않는다
- 두 번 이상 같은 패턴이 나올 때 추상화 검토
- `utils.py`, `helpers.py` 같은 잡동사니 파일 금지

---

## 7. Git workflow

### 7.1 브랜치
- `main`: 검증 통과한 상태만
- `step/step_XX_<short-desc>`: 각 step 작업 브랜치
- Step 완료 → main에 merge → tag `step_XX_done`

### 7.2 Commit 메시지
```
[step_XX] <description>

- 변경 1
- 변경 2

invariant: <어떤 invariant 통과했는지>
results: results/step_XX/vastai/summary.json
```

### 7.3 자동/수동
- Claude Code는 commit까지 자동
- Push는 사용자 승인 시
- Force-push 절대 금지

---

## 8. 사용자와의 협업

### 8.1 사용자(로컬 A100) 역할
- 대부분의 step에서는 사용자 검증 불필요. vast.ai 단독으로 진행.
- **특정 step에서만** (Step 0, Step 8, 또는 사용자가 명시한 step) 로컬 A100에서 재현 검증
- 사용자가 로컬 A100에서 직접 git pull 후 실험 실행 → `results/step_XX/local_a100/` 에 결과 저장
- 비교 결과 검토 후 리뷰 승인

### 8.2 결과 비교
사용자가 실행: `python scripts/compare_results.py --step XX`
- `local_a100` 결과가 없으면 vastai 단독 검증으로 판정
- 1순위: SHA-256 일치
- 2순위: atol 1e-5
- 3순위: top-k logit 일치 (Step 8 같은 task metric)
- 불일치 시: 어느 layer/position에서 갈렸는지 자동 리포트

### 8.3 리뷰 요청
Step 완료 시 다음을 사용자에게 명시:
1. 완료된 작업 요약
2. `docs/reports/step_XX_report.html` 링크
3. `results/step_XX/vastai/summary.json` 핵심 수치
4. 알려진 한계 / 의심스러운 부분
5. 사용자 로컬 A100 검증이 필요한 step이면 그 사실 명시
6. 다음 step 진행 가능 여부 confirm 요청

---

## 9. 참조 저장소 활용

| 저장소 | 활용 방식 |
|---|---|
| YaoJiayi/CacheBlend | 논문 매핑 코드 읽기. **실행 ❌** |
| chjs/LMCache fix branch | vLLM 환경에서 reference 실행 (Phase 2~). 별도 venv 필요 |
| chjs/CompBlend-old | 이전 시도 참고. **답습 ❌** (특히 4b의 결과 노이즈 함정 주의) |
| chjs/compblend2 | 직전 실패 시도. CLAUDE.md 워크플로우 패턴만 참고 |
| snu-mllab/KVzip | Phase 7부터 통합 대상 |
| MozerWang/Loong | Step 8부터 평가 데이터셋 |

참조 저장소는 **별도 위치에 clone**, 본 repo 안에 vendor ❌.

---

## 10. 컨텍스트 리셋 친화 체크리스트

새 세션 진입 시:
- [ ] GOAL.md 읽음
- [ ] PROGRESS.md 읽음 (current step 확인)
- [ ] 현재 step의 tasks/step_XX_*.md 읽음
- [ ] 직전 step의 docs/reports/step_(XX-1)_report.html 훑음
- [ ] Working directory가 올바른 step 브랜치인지 확인

세션 종료 시:
- [ ] PROGRESS.md 업데이트 (현재 상태 정확히)
- [ ] 미완 작업이 있다면 "다음 행동" 명시
- [ ] Commit 한 후 종료 (또는 commit할 수 없는 상태면 그 이유 PROGRESS.md에)

---

## 11. 절대 하지 말 것

1. **Step을 묶어서 진행하지 말 것** (Step 0~3을 한 번에 하지 말 것)
2. **검증 없이 다음 step 진행하지 말 것** (invariant 통과 안 됐는데 넘어가는 것)
3. **F1 안 나오면 코드를 슬쩍 고치지 말 것** (왜 안 나오는지 추적 가능한 구조 유지)
4. **CompBlend-old, compblend2의 결과를 답습하지 말 것** (참고는 OK, 같은 함정 반복 ❌)
5. **추상화를 미리 하지 말 것** (당장 필요한 만큼만)
6. **확신 없이 "잘 됐습니다"라고 하지 말 것** (objective metric으로 표현)
7. **HF transformers의 forward를 monkey-patch하지 말 것** (modeling 파일 fork)
8. **paper-headline을 위해 데이터를 굽지 말 것** (negative result도 그대로 보고)
9. **MacBook에서 GPU 의존 실험 시도하지 말 것** (모든 forward는 ssh vast로)
10. **SSH 명령 인라인에 시크릿(HF_TOKEN 등) 박지 말 것** (vast.ai .env에서만)
11. **vast.ai 인스턴스가 죽었을 때 자동 spawn 시도하지 말 것** (사용자에게 요청)
12. **vast.ai API로 인스턴스 destroy/stop 시도하지 말 것** (사용자 권한)

---

## 12. 이 문서의 변경

이 파일을 수정해야 한다고 판단되면:
1. 수정 제안을 사용자에게 먼저 보고
2. 사용자 승인 후 수정
3. 변경 사항을 DECISIONS.md 변경 이력에도 기록
