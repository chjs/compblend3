"""Step 2: HF DynamicCache forward 검증 (옵션 B, 2026-05-16 확정).

invariants:
  2.1  cache vs no-cache forward의 logits SHA-256 동일 (same-shape M=6)
  2.2  cache vs no-cache forward의 layer hidden state SHA-256 동일 (same-shape M=6)
  2.3A padded forward vs single forward의 DynamicCache K/V[:6] bitwise (same-shape M=7,
       use_cache=True 양쪽, torch.equal 게이트, atol fallback ❌)
  2.3B 운영 split forward (prefill 6 + M=1 decode)의 drift 측정 (gate ❌, 측정만)

옵션 B 채택 경위: C-3~C-7 진단으로 cuBLAS shape-dependent dispatch 확정.
이전 2.3A 정의(split prefill+decode = single full forward)는 cross-shape mechanism으로
bitwise 불가능. 대신 same-shape 구조(padded vs single, 둘 다 단일 forward + use_cache=True)
로 K/V cache 직접 비교.

fork 코드(src/compblend/modeling/)는 무수정. forward hook ❌, output_attentions ❌.
DynamicCache의 past_key_values[i].key_cache/value_cache를 직접 접근.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer

from compblend.modeling import MistralForCausalLM

SEED = 42
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"
PREFIX_LEN = 6                       # prompt token 수 (BOS 포함)
DRIFT_BUDGET = 1e-4                  # 2.3B optional warning threshold (gate ❌)


def set_all_seeds(seed: int = SEED) -> None:
    """결정론을 위한 모든 seed 설정 (Step 0/1과 동일 패턴)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic() -> None:
    """PyTorch deterministic mode + TF32 OFF (C-6 hygiene 확장)."""
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


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
    enc = tokenizer(PROMPT, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")              # (1, 6)
    attn = enc["attention_mask"].to("cuda")        # (1, 6)
    assert ids.shape[1] == PREFIX_LEN

    print(f"[1/7] 모델 로드: {args.model}")
    model = MistralForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        token=token,
    ).to("cuda").eval()
    n_layers = model.config.num_hidden_layers

    # ============================================================
    # (a) no-cache forward — 2.1·2.2 reference (use_cache=False)
    # ============================================================
    print("[2/7] (a) no-cache forward (use_cache=False, output_hidden_states=True)")
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_nocache = model(input_ids=ids, attention_mask=attn,
                             use_cache=False, output_hidden_states=True)
    # logits: (1, 6, vocab), hidden_states: 33 × (1, 6, H)
    nocache_logits = out_nocache.logits.detach().to("cpu", torch.float32)
    nocache_hiddens = [h.detach().to("cpu", torch.float32) for h in out_nocache.hidden_states]
    del out_nocache

    # ============================================================
    # (b) cache forward — 2.1·2.2 variant (use_cache=True)
    # ============================================================
    print("[3/7] (b) cache forward (use_cache=True, output_hidden_states=True)")
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_cache = model(input_ids=ids, attention_mask=attn,
                           use_cache=True, output_hidden_states=True)
    cache_logits = out_cache.logits.detach().to("cpu", torch.float32)
    cache_hiddens = [h.detach().to("cpu", torch.float32) for h in out_cache.hidden_states]
    # next_token_id: greedy argmax of (b) prompt last-token logits
    next_token_id = out_cache.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
    next_id_int = int(next_token_id.item())
    del out_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ids_full = [prompt(6) + next_token(1)] — padded·single 공통 input
    ids_full = torch.cat([ids, next_token_id], dim=1)                            # (1, 7)
    attn_padded = torch.cat([attn, torch.zeros_like(next_token_id)], dim=1)      # [1,1,1,1,1,1,0]
    attn_single = torch.ones_like(ids_full)                                      # [1]*7

    # ============================================================
    # (c) padded path — 2.3A (use_cache=True, attn[6]=0)
    # ============================================================
    print("[4/7] (c) padded forward (length 7, mask=[1*6,0], use_cache=True)")
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_padded = model(input_ids=ids_full, attention_mask=attn_padded,
                            use_cache=True)
    # padded_kv: list of (K (1, 8, 7, 128), V (1, 8, 7, 128)) per layer
    padded_k_list = [out_padded.past_key_values.key_cache[i].detach().clone()
                     for i in range(n_layers)]
    padded_v_list = [out_padded.past_key_values.value_cache[i].detach().clone()
                     for i in range(n_layers)]
    del out_padded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # (d) single path — 2.3A reference + 2.3B reference (use_cache=True)
    # ============================================================
    print("[5/7] (d) single forward (length 7, mask=[1]*7, use_cache=True)")
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_single = model(input_ids=ids_full, attention_mask=attn_single,
                            use_cache=True)
    single_k_list = [out_single.past_key_values.key_cache[i].detach().clone()
                     for i in range(n_layers)]
    single_v_list = [out_single.past_key_values.value_cache[i].detach().clone()
                     for i in range(n_layers)]
    # single_logits[:, 6, :] — 2.3B reference
    single_logits_pos6 = out_single.logits[:, PREFIX_LEN, :].detach().to("cpu", torch.float32)
    del out_single
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ============================================================
    # (e) operational split — 2.3B (prefill 6 + M=1 decode)
    # ============================================================
    print("[6/7] (e) operational split (prefill 6 + M=1 decode, use_cache=True)")
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_prefill = model(input_ids=ids, attention_mask=attn, use_cache=True)
        decode_attn_mask = torch.ones((1, PREFIX_LEN + 1), dtype=torch.long, device="cuda")
        decode_cache_position = torch.arange(PREFIX_LEN, PREFIX_LEN + 1, device="cuda")
        out_decode = model(input_ids=next_token_id,
                           past_key_values=out_prefill.past_key_values,
                           attention_mask=decode_attn_mask,
                           cache_position=decode_cache_position,
                           use_cache=True)
    # operational_logits: (1, vocab) — decode last-token logits
    operational_logits = out_decode.logits[:, -1, :].detach().to("cpu", torch.float32)
    del out_prefill, out_decode

    # ============================================================
    # 비교 — 2.1 (logits)
    # ============================================================
    print("[7/7] 비교 + summary 작성")
    cache_sha = tensor_sha256(cache_logits)
    nocache_sha = tensor_sha256(nocache_logits)
    inv_2_1 = cache_sha == nocache_sha

    # 2.2 (layer hidden state)
    mismatched_layers_22: list = []
    if len(cache_hiddens) != len(nocache_hiddens):
        mismatched_layers_22.append("LENGTH_MISMATCH")
    for i in range(min(len(cache_hiddens), len(nocache_hiddens))):
        if tensor_sha256(cache_hiddens[i]) != tensor_sha256(nocache_hiddens[i]):
            mismatched_layers_22.append(i)
    inv_2_2 = len(mismatched_layers_22) == 0

    # 2.3A (per-layer K/V [:PREFIX_LEN] torch.equal)
    per_layer_23a: list[dict[str, Any]] = []
    mismatched_layers_23a: list[int] = []
    for i in range(n_layers):
        pk6 = padded_k_list[i][:, :, :PREFIX_LEN, :]
        sk6 = single_k_list[i][:, :, :PREFIX_LEN, :]
        pv6 = padded_v_list[i][:, :, :PREFIX_LEN, :]
        sv6 = single_v_list[i][:, :, :PREFIX_LEN, :]
        k_match = bool(torch.equal(pk6, sk6))
        v_match = bool(torch.equal(pv6, sv6))
        per_layer_23a.append({
            "layer": i,
            "k_match": k_match,
            "v_match": v_match,
            "k_sha_padded": tensor_sha256(pk6),
            "k_sha_single": tensor_sha256(sk6),
            "v_sha_padded": tensor_sha256(pv6),
            "v_sha_single": tensor_sha256(sv6),
            "k_max_abs_diff": float((pk6.cpu().float() - sk6.cpu().float()).abs().max()),
            "k_mean_abs_diff": float((pk6.cpu().float() - sk6.cpu().float()).abs().mean()),
            "v_max_abs_diff": float((pv6.cpu().float() - sv6.cpu().float()).abs().max()),
            "v_mean_abs_diff": float((pv6.cpu().float() - sv6.cpu().float()).abs().mean()),
        })
        if not (k_match and v_match):
            mismatched_layers_23a.append(i)
    inv_2_3a = len(mismatched_layers_23a) == 0

    # 2.3B (operational drift, gate ❌)
    drift = (operational_logits - single_logits_pos6).abs()
    drift_max = float(drift.max())
    drift_mean = float(drift.mean())
    op_argmax = int(operational_logits.argmax(dim=-1).item())
    si_argmax = int(single_logits_pos6.argmax(dim=-1).item())
    argmax_match = op_argmax == si_argmax
    op_top5 = set(operational_logits.topk(5, dim=-1).indices[0].tolist())
    si_top5 = set(single_logits_pos6.topk(5, dim=-1).indices[0].tolist())
    topk_overlap_k5 = len(op_top5 & si_top5)
    drift_budget_exceeded = drift_max > DRIFT_BUDGET

    # 게이트 (2.3B 제외)
    all_passed = inv_2_1 and inv_2_2 and inv_2_3a

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "step": 2,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "torch_dtype": "float32",
        "attention_implementation": "eager",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prompt": PROMPT,
        "decode_token_id": next_id_int,
        "decode_token_decoded": tokenizer.decode([next_id_int]),
        "invariants": {
            "2.1_dynamic_cache_logits_equiv": {
                "passed": inv_2_1,
                "cache_logits_sha256": cache_sha,
                "nocache_logits_sha256": nocache_sha,
            },
            "2.2_per_layer_hidden_equiv": {
                "passed": inv_2_2,
                "n_hidden_states": len(cache_hiddens),
                "mismatched_layers": mismatched_layers_22,
            },
            "2.3A_padded_cache_kv_equiv": {
                "passed": inv_2_3a,
                "gate": "torch.equal",
                "per_layer": per_layer_23a,
                "mismatched_layers": mismatched_layers_23a,
            },
            "2.3B_operational_split_drift": {
                "measured": True,
                "max_abs_diff": drift_max,
                "mean_abs_diff": drift_mean,
                "argmax_match": argmax_match,
                "operational_argmax": op_argmax,
                "single_argmax": si_argmax,
                "topk_overlap_k5": topk_overlap_k5,
                "drift_budget_exceeded": drift_budget_exceeded,
                "drift_budget_threshold": DRIFT_BUDGET,
                "note": "measurement only, no pass/fail gate. cuBLAS shape-dependent dispatch (C-3/C-4/C-6/C-7).",
            },
        },
        "all_invariants_passed": all_passed,
    }
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_file}")

    # 콘솔 요약
    print()
    print(f"  decode token: id={next_id_int} -> {tokenizer.decode([next_id_int])!r}")
    print(f"  invariant 2.1 (cache=no-cache logits):  {'✅' if inv_2_1 else '❌'}")
    print(f"  invariant 2.2 (layer hidden state):     {'✅' if inv_2_2 else '❌'}"
          f" (mismatched_layers={mismatched_layers_22})")
    print(f"  invariant 2.3A (padded K/V[:6] bitwise): {'✅' if inv_2_3a else '❌'}"
          f" (mismatched_layers={mismatched_layers_23a})")
    print(f"  2.3B drift max_abs={drift_max:.3e} mean={drift_mean:.3e}"
          f" argmax_match={argmax_match} top5_overlap={topk_overlap_k5}/5"
          f" drift_budget_exceeded={drift_budget_exceeded}")
    if all_passed:
        print("==> 2.1·2.2·2.3A 모두 PASS ✅ (2.3B는 측정값만)")
    else:
        print("==> 게이트 invariant 실패 ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
