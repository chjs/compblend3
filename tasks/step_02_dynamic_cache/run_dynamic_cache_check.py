"""Step 2: HF DynamicCache forward = no-cache forward 검증 (bitwise 동등).

invariant 2.1 (logits SHA-256 동일), 2.2 (layer hidden state SHA-256 동일),
2.3A (split forward = single forward, bitwise 1차 → atol 1e-6 fallback).

fork 코드(src/compblend/modeling/)는 무수정. DynamicCache는 외부 객체로 호출.
모든 계측은 이 스크립트 안에서만 (Step 1과 동일 원칙).
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
import transformers
from transformers import AutoTokenizer

from compblend.modeling import MistralForCausalLM

SEED = 42
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"
ATOL_FALLBACK = 1e-6  # 2.3A: bitwise 실패 시 fallback 임계값


def set_all_seeds(seed: int = SEED) -> None:
    """결정론을 위한 모든 seed 설정 (Step 0/1과 동일 패턴)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic() -> None:
    """PyTorch deterministic mode."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_sha256(t: torch.Tensor) -> str:
    """텐서를 cpu fp32 numpy bytes로 변환 후 SHA-256 (Step 0/1과 동일 기준)."""
    arr = t.detach().cpu().to(torch.float32).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 출력 디렉토리 (예: results/step_02/vastai)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정. 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' 후 재실행.")
        sys.exit(1)

    setup_deterministic()
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    # ids: BatchEncoding {input_ids: (1, T_prompt), attention_mask: (1, T_prompt)}
    # prompt = "The capital of France is" → T_prompt = 6 (BOS 포함)
    ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    print(f"[1/5] 모델 로드: {args.model}")
    model = MistralForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        token=token,
    ).to("cuda").eval()

    # (a) no-cache forward — 2.1·2.2의 reference (use_cache=False)
    print("[2/5] no-cache forward (use_cache=False, output_hidden_states=True)")
    set_all_seeds(SEED)
    with torch.no_grad():
        out_nocache = model(**ids, use_cache=False, output_hidden_states=True)
    # logits: (1, T_prompt, vocab), hidden_states: 33 × (1, T_prompt, H)
    nocache_logits = out_nocache.logits.detach().to("cpu", torch.float32)
    nocache_hiddens = [h.detach().to("cpu", torch.float32) for h in out_nocache.hidden_states]
    del out_nocache

    # (b) cache forward — 2.1·2.2의 변형 path (use_cache=True, DynamicCache 자동 생성)
    print("[3/5] cache forward (use_cache=True, output_hidden_states=True)")
    set_all_seeds(SEED)
    with torch.no_grad():
        out_cache = model(**ids, use_cache=True, output_hidden_states=True)
    # out_cache.past_key_values: DynamicCache 인스턴스 (자동 생성)
    cache_logits = out_cache.logits.detach().to("cpu", torch.float32)
    cache_hiddens = [h.detach().to("cpu", torch.float32) for h in out_cache.hidden_states]
    del out_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # (c) split path (2.3A): prefill 후 decode
    # 주의: set_all_seeds는 split path 시작 직전 1회만. prefill·decode 사이에
    # 재호출 ❌ — split path는 "연속된 두 forward call"이라 single path와 동등
    # 비교가 목적이고, 중간 재seed는 single path와 어긋남.
    print("[4/5] split path: prefill + decode (seed 1회, prefill→decode 사이 재seed ❌)")
    set_all_seeds(SEED)
    with torch.no_grad():
        out_prefill = model(**ids, use_cache=True)
        # next_token_id: (1, 1) — greedy argmax of last-token logits
        next_token_id = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # decode 단계: attention_mask 미전달 → HF가 cache_position 기반으로 추론.
        # past_key_values는 prefill의 DynamicCache 인스턴스 그대로 사용.
        out_decode = model(
            input_ids=next_token_id,
            past_key_values=out_prefill.past_key_values,
            use_cache=True,
        )
    # split_logits: (1, vocab) — decode 단계의 last-token logits (decode 출력은 1 token)
    split_logits = out_decode.logits[:, -1, :].detach().to("cpu", torch.float32)
    next_id_int = int(next_token_id.item())
    next_decoded = tokenizer.decode([next_id_int])
    print(f"  decode token: id={next_id_int} -> {next_decoded!r}")
    del out_prefill, out_decode
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # (d) single path (2.3A): ids + next_token concat 후 단일 forward
    # ids_full_input: (1, T_prompt + 1)
    ids_full_input = torch.cat([ids["input_ids"], next_token_id], dim=1)
    ids_full = {
        "input_ids": ids_full_input,
        "attention_mask": torch.ones_like(ids_full_input),
    }
    print(f"[5/5] single path: forward over {ids_full_input.shape[1]} tokens (use_cache=False)")
    set_all_seeds(SEED)
    with torch.no_grad():
        out_single = model(**ids_full, use_cache=False)
    # single_logits: (1, vocab) — last-token logits, split_logits와 비교 대상
    single_logits = out_single.logits[:, -1, :].detach().to("cpu", torch.float32)
    del out_single

    # --- invariant 2.1: cache vs no-cache logits ---
    cache_sha = tensor_sha256(cache_logits)
    nocache_sha = tensor_sha256(nocache_logits)
    inv_2_1 = cache_sha == nocache_sha

    # --- invariant 2.2: cache vs no-cache hidden states (layer별) ---
    mismatched_layers: list = []
    if len(cache_hiddens) != len(nocache_hiddens):
        mismatched_layers.append("LENGTH_MISMATCH")
    for i in range(min(len(cache_hiddens), len(nocache_hiddens))):
        if tensor_sha256(cache_hiddens[i]) != tensor_sha256(nocache_hiddens[i]):
            mismatched_layers.append(i)
    inv_2_2 = len(mismatched_layers) == 0

    # --- invariant 2.3A: split vs single (3-tier: bitwise → atol → FAIL) ---
    split_sha = tensor_sha256(split_logits)
    single_sha = tensor_sha256(single_logits)
    max_abs = float((split_logits - single_logits).abs().max())
    if split_sha == single_sha:
        tier = "bitwise"
        inv_2_3 = True
        fallback_used = False
    elif max_abs <= ATOL_FALLBACK:
        tier = "atol_1e-6_fallback"
        inv_2_3 = True
        fallback_used = True
    else:
        tier = "FAIL"
        inv_2_3 = False
        fallback_used = False

    all_passed = inv_2_1 and inv_2_2 and inv_2_3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "step": 2,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "torch_dtype": "float32",
        "attention_implementation": "eager",
        "transformers_version": transformers.__version__,
        "prompt": PROMPT,
        "decode_token_id": next_id_int,
        "decode_token_decoded": next_decoded,
        "invariants": {
            "2.1_dynamic_cache_logits_equiv": {
                "passed": inv_2_1,
                "cache_logits_sha256": cache_sha,
                "nocache_logits_sha256": nocache_sha,
            },
            "2.2_per_layer_hidden_equiv": {
                "passed": inv_2_2,
                "n_hidden_states": len(cache_hiddens),
                "mismatched_layers": mismatched_layers,
            },
            "2.3A_split_vs_single_forward": {
                "passed": inv_2_3,
                "comparison_tier": tier,
                "split_logits_sha256": split_sha,
                "single_logits_sha256": single_sha,
                "max_abs_diff": max_abs,
                "atol_threshold": ATOL_FALLBACK,
                "fallback_used": fallback_used,
            },
        },
        "all_invariants_passed": all_passed,
    }
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_file}")

    print()
    print(f"  invariant 2.1 (cache=no-cache logits):  {'✅' if inv_2_1 else '❌'}")
    print(f"  invariant 2.2 (layer hidden state):     {'✅' if inv_2_2 else '❌'}")
    print(f"  invariant 2.3A (split=single, tier={tier}, max_abs={max_abs:.2e}): {'✅' if inv_2_3 else '❌'}")
    if all_passed:
        print("==> 모든 invariant 통과 ✅")
    else:
        print("==> Invariant 실패 ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
