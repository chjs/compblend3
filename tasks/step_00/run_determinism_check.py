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
