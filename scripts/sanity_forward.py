#!/usr/bin/env python3
"""Phase 0 sanity check: Mistral-7B-Instruct-v0.2 모델을 fp32 eager로 로드하여
짧은 input에 대해 forward 1회 수행.

GPU에 모델이 로드되는지, 합리적인 next-token 예측이 나오는지 확인한다.
결과: results/phase_00/{env}/sanity_forward.json
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dotenv():
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and v.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()


def get_env_tag() -> str:
    tag = os.environ.get("COMPBLEND_ENV_TAG", "")
    if tag in ("vastai", "local"):
        return tag
    hostname = os.uname().nodename.lower()
    if any(k in hostname for k in ("vast", "runpod", "lambda")):
        return "vastai"
    return "local"


def main():
    load_dotenv()

    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA 사용 불가. nvidia-smi 확인.")
        sys.exit(1)

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("[WARN] HF_TOKEN 미설정. gated 모델 다운로드 실패 가능.")

    print(f"[1/3] 모델 로드: {model_id}")
    print(f"      torch_dtype=float32, attn_implementation=eager")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        token=token,
    ).to("cuda").eval()

    # 메모리 확인
    mem_alloc_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"      GPU memory allocated: {mem_alloc_gb:.2f} GB")

    print()
    print(f"[2/3] Sample forward")
    prompt = "The capital of France is"
    ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"      prompt: {prompt!r}")
    print(f"      input_ids shape: {tuple(ids.input_ids.shape)}")

    with torch.no_grad():
        out = model(**ids)

    logits = out.logits  # (B, T, V)
    last = logits[0, -1]  # (V,)
    top5_vals, top5_idx = last.topk(5)
    top1_token_id = int(top5_idx[0].item())
    top1_token_str = tokenizer.decode([top1_token_id])

    print(f"      logits shape: {tuple(logits.shape)}")
    print(f"      top-1 next token: {top1_token_id} -> {top1_token_str!r}")
    print(f"      top-5 token ids: {top5_idx.tolist()}")
    print(f"      top-5 values: {[round(v, 3) for v in top5_vals.tolist()]}")

    # SHA-256 (재현성 확인용)
    arr = logits.detach().cpu().to(torch.float32).numpy()
    logits_sha = hashlib.sha256(arr.tobytes()).hexdigest()
    print(f"      logits sha256: {logits_sha[:16]}...")

    # 결과 저장
    print()
    print(f"[3/3] 결과 저장")
    env_tag = get_env_tag()
    out_dir = Path("results/phase_00") / env_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sanity_forward.json"

    summary = {
        "env_tag": env_tag,
        "model_id": model_id,
        "torch_dtype": "float32",
        "attention_implementation": "eager",
        "seed": SEED,
        "prompt": prompt,
        "logits_shape": list(logits.shape),
        "logits_last_top5_values": top5_vals.tolist(),
        "logits_last_top5_indices": top5_idx.tolist(),
        "top1_token_id": top1_token_id,
        "top1_token_str": top1_token_str,
        "logits_sha256": logits_sha,
        "gpu_memory_allocated_gb": mem_alloc_gb,
    }
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"      저장: {out_file}")

    print()
    print("==> Sanity forward 완료 ✅")
    print(f"    Top-1 token이 합리적인지 확인: {top1_token_str!r} (예상: ' Paris')")


if __name__ == "__main__":
    main()
