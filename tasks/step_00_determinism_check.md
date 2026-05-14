# Step 0 — HF eager forward 결정론 확인

> Self-contained task. 이 파일만 읽고도 작업 가능해야 한다.

---

## 목표

HF transformers의 Mistral-7B-Instruct-v0.2 eager forward가 **같은 입력 + 같은 seed 조건에서 bitwise 재현 가능**함을 확인한다.

이것이 모든 후속 invariant 검증의 토대이다. 결정론이 보장되지 않으면 "정확함"을 정의할 수 없다.

## 사전 조건

- Phase 0 완료 (`PROGRESS.md` 의 Phase 0 게이트 모두 통과)
- 모델 캐시 존재
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 환경변수 설정

## 통과 기준 / Invariants

### Invariant 0.1 — 환경 내 결정론
같은 환경 안에서 3회 실행 시 logits 텐서가 **SHA-256 bitwise 동일**.

```
sha256(run1_logits) == sha256(run2_logits) == sha256(run3_logits)
```

### Invariant 0.2 — 부분 입력 안정성
서로 다른 입력에서는 다른 logits.
```
sha256(run1_logits_input_A) != sha256(run1_logits_input_B)
```
(이건 trivial하지만 invariant 0.1이 모든 입력에 대해 trivially 동일한 출력을 만들지 않는지 확인)

### Invariant 0.3 — 환경 간 동등성 (best-effort)
vast.ai와 로컬의 logits가 SHA-256 일치하면 가장 좋음.
일치하지 않으면 atol 1e-5 일치는 보장되어야 함.

## 구현 사양

### 새 파일

**`tasks/step_00/run_determinism_check.py`** — 실행 스크립트

```python
"""Step 0: HF eager forward 결정론 확인.

같은 prompt + 같은 seed로 3회 forward → logits SHA-256 비교.
서로 다른 prompt로 1회 forward → 다른 SHA-256 확인.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_all_seeds(seed: int = 42):
    """결정론을 위한 모든 seed 설정."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic():
    """PyTorch deterministic mode."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def logits_sha256(logits: torch.Tensor) -> str:
    """logits 텐서의 SHA-256."""
    # fp32로 변환해서 cpu numpy bytes 기반 hash
    arr = logits.detach().cpu().to(torch.float32).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def logits_summary(logits: torch.Tensor) -> dict:
    """logits 통계 요약 (비교용)."""
    arr = logits.detach().cpu().to(torch.float32)
    return {
        "shape": list(arr.shape),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "norm": float(arr.norm()),
        "last_token_top5_values": arr[0, -1].topk(5).values.tolist(),
        "last_token_top5_indices": arr[0, -1].topk(5).indices.tolist(),
    }


def run_forward(model, tokenizer, prompt: str, seed: int = 42) -> torch.Tensor:
    """단일 forward."""
    set_all_seeds(seed)
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**ids)
    return out.logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 출력 디렉토리 (예: results/step_00/vastai)")
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # 환경변수 체크
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정. 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' 후 재실행.")
        sys.exit(1)

    setup_deterministic()

    print(f"[1/4] 모델 로드: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).to("cuda").eval()

    # 두 개의 다른 prompt
    prompt_A = "The capital of France is"
    prompt_B = "Photosynthesis is the process by which"

    # Invariant 0.1: 같은 입력 × n_runs
    print(f"[2/4] Invariant 0.1 — 같은 입력 {args.n_runs}회 실행")
    runs_A = []
    for i in range(args.n_runs):
        logits = run_forward(model, tokenizer, prompt_A, seed=args.seed)
        h = logits_sha256(logits)
        s = logits_summary(logits)
        runs_A.append({"run": i + 1, "sha256": h, "summary": s})
        print(f"  run {i+1}: sha256={h[:16]}...")

    inv_0_1_pass = len(set(r["sha256"] for r in runs_A)) == 1

    # Invariant 0.2: 다른 입력
    print(f"[3/4] Invariant 0.2 — 다른 입력 1회 실행")
    logits_B = run_forward(model, tokenizer, prompt_B, seed=args.seed)
    sha_B = logits_sha256(logits_B)
    print(f"  prompt_B sha256={sha_B[:16]}...")

    inv_0_2_pass = sha_B != runs_A[0]["sha256"]

    # 결과 저장
    print(f"[4/4] 결과 저장")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "step": 0,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "torch_dtype": "float32",
        "attention_implementation": "eager",
        "seed": args.seed,
        "n_runs": args.n_runs,
        "invariants": {
            "0.1_same_input_deterministic": {
                "passed": inv_0_1_pass,
                "description": "같은 입력 + 같은 seed 3회 → logits SHA-256 동일",
            },
            "0.2_different_input_distinguishable": {
                "passed": inv_0_2_pass,
                "description": "다른 입력 → logits SHA-256 다름",
            },
        },
        "all_invariants_passed": inv_0_1_pass and inv_0_2_pass,
        # 비교 스크립트가 사용할 표준 필드
        "logits_sha256": runs_A[0]["sha256"],  # 대표 sha
        "logits_summary": runs_A[0]["summary"],
        "details": {
            "prompt_A": prompt_A,
            "prompt_B": prompt_B,
            "runs_A": runs_A,
            "prompt_B_sha256": sha_B,
        },
    }

    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_file}")

    print()
    if summary["all_invariants_passed"]:
        print("==> 모든 invariant 통과 ✅")
    else:
        print("==> Invariant 실패 ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### 디렉토리

```bash
mkdir -p tasks/step_00
mkdir -p results/step_00/{vastai,local}
```

## 실행 명령

**중요**: Claude Code는 MacBook에서 돌고, 실험은 vast.ai에서 실행한다. 양쪽 결과 모두 git으로 회수한다.

### vast.ai 실행 (Claude가 ssh로 트리거)

```bash
# MacBook에서 코드 commit + push 후
git add tasks/step_00 src
git commit -m "[step_00] determinism check script"
git push

# vast.ai에 코드 동기화
bash scripts/vast_run.sh sync

# vast.ai에서 실행
bash scripts/vast_run.sh run "python tasks/step_00/run_determinism_check.py --out results/step_00/vastai/"

# vast.ai에서 결과 push
bash scripts/vast_run.sh push "[step_00] determinism results from vastai"
# (자동으로 MacBook에서 git pull도 함)
```

### 사용자 로컬 A100 실행 (사용자가 결정한 경우, 직접 실행)

```bash
# 로컬 A100 머신에서
cd ~/work/compblend3
git pull
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
python tasks/step_00/run_determinism_check.py --out results/step_00/local_a100/

# 결과 push
git add results/step_00/local_a100/
git commit -m "[step_00] determinism results from local_a100"
git push
```

**Step 0은 환경 간 결정론 비교가 핵심이므로 로컬 A100 검증을 강력 권장.**

## 결과 저장 형식

`results/step_00/{env}/summary.json` 의 schema는 위 스크립트 안에 정의됨.

핵심 필드 (compare_results.py 가 사용):
- `logits_sha256` — 대표 SHA-256 (Tier 1)
- `logits_summary.{max,min,mean,norm}` — 수치 비교 (Tier 2)
- `invariants.0.1_same_input_deterministic.passed` — 환경 내 결정론
- `invariants.0.2_different_input_distinguishable.passed` — sanity check
- `all_invariants_passed` — 전체 PASS 여부

## 결과 비교

MacBook에서 실행:
```bash
python scripts/compare_results.py --step 0
```

기대:
- 양쪽 결과(vastai, local_a100) 모두 있는 경우:
  - Tier 1 PASS (SHA-256 일치) — 가장 좋은 결과
  - 또는 Tier 2 PASS (atol 1e-5)
  - 또는 Tier 3 PASS (token sequence)
- local_a100 결과가 없는 경우:
  - vastai 단독 invariant 검증으로 통과 판정

Step 0은 결정론이 핵심이므로 local_a100 검증을 권장. 그러나 사용자가 시간상 어렵다면 vastai 단독으로도 진행 가능.

## 보고서 작성 가이드

`docs/reports/step_00_determinism_report.html` 작성. 필수 섹션:

1. **요약** — 결정론 보장 여부 (PASS/FAIL badge)
2. **환경 정보** — torch, CUDA, GPU, deterministic mode (table)
3. **수정 파일** — `tasks/step_00/run_determinism_check.py` 추가 (table)
4. **Invariant 검증 결과** (table):
    | Invariant | 설명 | 결과 |
    |---|---|---|
    | 0.1 | 같은 입력 3회 → SHA-256 동일 | PASS/FAIL badge |
    | 0.2 | 다른 입력 → SHA-256 다름 | PASS/FAIL badge |
5. **결과 데이터** — `summary.json` 의 핵심 (prompt_A의 logits_sha256, summary stats)
6. **환경 간 비교** — vastai vs local 의 sha256, atol 비교 결과
7. **알려진 한계** — fp16/bf16 같은 lower precision은 deterministic 더 어려움 (Phase 6 주의)
8. **다음 step** — Step 1 (Our layerwise forward = HF 표준 forward) 진행 권장

## 다음 step 게이트

Step 0 완료 → Step 1 진입 조건:

- [ ] `results/step_00/vastai/summary.json` 의 `all_invariants_passed: true`
- [ ] (선택) `results/step_00/local_a100/summary.json` 의 `all_invariants_passed: true`
- [ ] `python scripts/compare_results.py --step 0` 결과: 양쪽 있으면 Tier 1/2/3 중 하나 PASS, vastai 단독이면 invariant PASS
- [ ] 사용자 리뷰 승인

사용자에게 다음 요청:
> "Step 0 완료. 결정론 보장 확인됨.
> vastai: invariant 0.1/0.2 모두 PASS, SHA-256: ___
> (선택)local_a100: ___
> docs/reports/step_00_determinism_report.html 확인 부탁드립니다. Step 1 진행해도 될까요?"

## 솔직성 노트

- PyTorch 2.10이 비교적 새 버전이라 일부 op의 deterministic 알고리즘이 변동 가능. 만약 invariant 0.1이 실패하면:
  - 어떤 layer에서 비결정성이 들어오는지 추적 (layer hook 추가하여 hash 비교)
  - `torch.use_deterministic_algorithms(True, warn_only=True)` 로 완화 시도
  - 실패 사유 보고서에 정확히 기록 — 결정론 없이는 후속 step의 모든 invariant가 무의미해짐
- vast.ai ↔ 로컬의 SHA-256 일치는 보장 못 함. 환경이 매우 비슷해도 cuDNN 버전 차이 등으로 갈릴 수 있음. 그 경우 atol 1e-5 fallback.
- 이 step에서 KV cache를 사용하지 않는다 (just plain forward). Step 2에서 DynamicCache 도입.
