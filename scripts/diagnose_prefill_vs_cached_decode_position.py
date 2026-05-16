#!/usr/bin/env python3
"""C-3 진단: prefill vs cached decode의 정확한 divergence 지점 localization.

목표: A case (cached decode) vs B case (full prefill)의 마지막 logits가
`max_abs_diff ≈ 6.20e-06` 다른 이유를 atol 완화 없이 추적.

A case:
    prefix = "The capital of France is"
    prefix_in.input_ids → forward(use_cache=True) → cache
    next_ids = full_in.input_ids[:, prefix_len:]   (절대 tokenizer("Paris") ❌)
    cache + next_ids → forward → last logits

B case:
    full = "The capital of France is Paris"
    full_in.input_ids → forward(use_cache=True) → last logits

비교:
    1. tokenizer special token / add_special_tokens policy
    2. RoPE forward monkey-patch로 actual position_ids 캡처
    3. 3 decode variants (auto / explicit_correct / explicit_wrong)
    4. prefix KV: prefix-only vs full-prefill[:, :, :prefix_len, :]
    5. per-layer hidden_states earliest divergence

출력:
    artifacts/c3_diagnosis/summary.json
    artifacts/c3_diagnosis/position_report.md
    artifacts/c3_diagnosis/earliest_divergence.csv

결론 case (A: position bug / B: tokenization / C: prefix KV drift /
         D: current q/k/v / E: attention output / F: lm_head) — 휴리스틱 추론
"""
from __future__ import annotations

import argparse
import csv
import inspect
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
PREFIX = "The capital of France is"
FULL = "The capital of France is Paris"


def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 비활성 — fp32 정밀도 최대 유지
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict:
    """두 텐서의 max/mean abs diff + argmax 위치. CPU fp32 변환 후 계산."""
    a = a.detach().to("cpu", torch.float32)
    b = b.detach().to("cpu", torch.float32)
    if a.shape != b.shape:
        return {
            "shape_mismatch": True,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "max_abs": None,
            "mean_abs": None,
            "argmax_idx": None,
        }
    d = (a - b).abs()
    return {
        "shape": list(a.shape),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "argmax_idx": [int(x) for x in np.unravel_index(int(d.argmax().item()), a.shape)],
    }


def install_rope_hook(model, store: dict):
    """`model.model.rotary_emb.forward` monkey-patch. position_ids 캡처. 반환: restore()."""
    rope = model.model.rotary_emb
    orig = rope.forward
    sig = inspect.signature(orig)
    store.setdefault("rope_signature", str(sig))
    store.setdefault("rope_calls", [])

    def hook(*args, **kwargs):
        # Mistral 4.51.3: rotary_emb(hidden_states, position_ids) — positional
        pos = kwargs.get("position_ids")
        if pos is None and len(args) >= 2:
            # args[1]은 보통 position_ids (long tensor (B, T))
            cand = args[1]
            if torch.is_tensor(cand) and cand.dtype in (torch.long, torch.int64, torch.int32):
                pos = cand
        if pos is not None:
            store["rope_calls"].append(pos.detach().cpu().clone())
        return orig(*args, **kwargs)

    rope.forward = hook

    def restore():
        rope.forward = orig

    return restore


def tokenizer_policy_probe(tokenizer) -> dict:
    """3가지 add_special_tokens policy 모두로 prefix/full split invariant 테스트."""
    results = {}
    for policy_name, ast in [
        ("default", "unset"),
        ("add_special_True", True),
        ("add_special_False", False),
    ]:
        kw = {"return_tensors": "pt"}
        if ast != "unset":
            kw["add_special_tokens"] = ast
        pi = tokenizer(PREFIX, **kw)
        fi = tokenizer(FULL, **kw)
        p_ids = pi["input_ids"]
        f_ids = fi["input_ids"]
        p_len = int(p_ids.shape[1])
        try:
            split_ok = bool(torch.equal(p_ids, f_ids[:, :p_len]))
        except Exception:
            split_ok = False
        n_ids = f_ids[:, p_len:]
        results[policy_name] = {
            "prefix_len": p_len,
            "full_len": int(f_ids.shape[1]),
            "prefix_ids": p_ids[0].tolist(),
            "full_ids": f_ids[0].tolist(),
            "next_ids": n_ids[0].tolist(),
            "split_invariant": split_ok,
            "next_len_is_1": int(n_ids.shape[1]) == 1,
            "prefix_tokens": tokenizer.convert_ids_to_tokens(p_ids[0].tolist()),
            "full_tokens": tokenizer.convert_ids_to_tokens(f_ids[0].tolist()),
            "next_tokens": tokenizer.convert_ids_to_tokens(n_ids[0].tolist()),
        }
    return results


def fresh_prefix_cache(model, prefix_in: dict, device: str):
    """매 variant마다 fresh prefix cache (DynamicCache 재사용 금지)."""
    set_all_seeds(SEED)
    with torch.inference_mode():
        out = model(**prefix_in, use_cache=True)
    return out.past_key_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/c3_diagnosis")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정.")
        sys.exit(1)

    setup_deterministic()
    token = os.environ.get("HF_TOKEN")
    device = "cuda"

    # === A. Tokenizer info ===
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token)
    tok_info = {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "add_bos_token": getattr(tokenizer, "add_bos_token", None),
        "add_eos_token": getattr(tokenizer, "add_eos_token", None),
    }
    print("=== Tokenizer info ===")
    for k, v in tok_info.items():
        print(f"  {k}: {v}")

    # === B. 3 policy probe ===
    print("\n=== add_special_tokens policy probe ===")
    policy_results = tokenizer_policy_probe(tokenizer)
    for pn, pr in policy_results.items():
        print(f"  {pn}: prefix_len={pr['prefix_len']}, full_len={pr['full_len']}, "
              f"split_ok={pr['split_invariant']}, next_len_1={pr['next_len_is_1']}, "
              f"next_ids={pr['next_ids']}")

    # 본 실험은 default policy 사용. invariant 미충족 시 즉시 중단.
    default = policy_results["default"]
    if not default["split_invariant"]:
        sys.exit("[ERROR] default policy의 split invariant 실패")
    if not default["next_len_is_1"]:
        sys.exit(f"[ERROR] next_ids 길이 {len(default['next_ids'])} != 1")

    prefix_in = {k: v.to(device) for k, v in tokenizer(PREFIX, return_tensors="pt").items()}
    full_in = {k: v.to(device) for k, v in tokenizer(FULL, return_tensors="pt").items()}
    prefix_ids = prefix_in["input_ids"]
    full_ids = full_in["input_ids"]
    prefix_len = int(prefix_ids.shape[1])
    next_ids = full_ids[:, prefix_len:]
    batch_size = prefix_ids.shape[0]
    new_len = int(next_ids.shape[1])

    # 추가 assert (사용자 요구사항)
    assert torch.equal(prefix_ids, full_ids[:, :prefix_len]), "prefix split invariant 실패"
    assert torch.equal(next_ids, full_ids[:, prefix_len:]), "next_ids split invariant 실패"
    assert new_len == 1, f"next_len={new_len} != 1"
    print(f"\n  used default policy: prefix_len={prefix_len}, full_len={full_ids.shape[1]}, "
          f"next_ids={next_ids[0].tolist()}, batch_size={batch_size}")

    # === 모델 로드 ===
    print("\n=== Model load ===")
    model = MistralForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager", token=token,
    ).to(device).eval()
    print(f"  n_layers: {len(model.model.layers)}")

    # === B case: full prefill ===
    print("\n=== B case: full prefill ===")
    cap_full: dict = {}
    restore_f = install_rope_hook(model, cap_full)
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_full = model(**full_in, use_cache=True, output_hidden_states=True)
    restore_f()
    full_logits = out_full.logits.detach().to("cpu", torch.float32)
    full_hiddens = [h.detach().to("cpu", torch.float32) for h in out_full.hidden_states]
    full_cache = out_full.past_key_values
    full_last_logits = full_logits[:, -1, :]
    full_last_hiddens = [h[:, -1, :] for h in full_hiddens]  # 33 × (1, H)
    rope_full_pos = cap_full["rope_calls"][0][0].tolist() if cap_full.get("rope_calls") else None
    print(f"  logits {tuple(full_logits.shape)}, cache seq_len={full_cache.get_seq_length()}")
    print(f"  rope position_ids (full): {rope_full_pos}")
    print(f"  rope signature: {cap_full.get('rope_signature')}")

    # === A case prefill: prefix forward (reference cache) ===
    print("\n=== A case prefill: prefix forward ===")
    cap_pref: dict = {}
    restore_p = install_rope_hook(model, cap_pref)
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_prefix = model(**prefix_in, use_cache=True, output_hidden_states=True)
    restore_p()
    prefix_logits = out_prefix.logits.detach().to("cpu", torch.float32)
    prefix_cache = out_prefix.past_key_values
    rope_pref_pos = cap_pref["rope_calls"][0][0].tolist() if cap_pref.get("rope_calls") else None
    print(f"  logits {tuple(prefix_logits.shape)}, cache seq_len={prefix_cache.get_seq_length()}")
    print(f"  rope position_ids (prefix): {rope_pref_pos}")
    cache_seq_len = prefix_cache.get_seq_length()
    assert cache_seq_len == prefix_len, (
        f"cache.get_seq_length()={cache_seq_len} != prefix_len={prefix_len}"
    )

    # === 3. Prefix KV diff (prefix-only vs full[:, :, :prefix_len, :]) ===
    print("\n=== Prefix KV diff: prefix-only vs full-prefill prefix-slice ===")
    prefix_legacy = prefix_cache.to_legacy_cache()  # tuple of (K, V) per layer
    full_legacy = full_cache.to_legacy_cache()
    n_layers = len(prefix_legacy)
    prefix_kv_diffs = []
    for i in range(n_layers):
        k_p, v_p = prefix_legacy[i]
        k_f, v_f = full_legacy[i]
        k_f_s = k_f[:, :, :prefix_len, :]
        v_f_s = v_f[:, :, :prefix_len, :]
        kd = tensor_diff(k_p, k_f_s)
        vd = tensor_diff(v_p, v_f_s)
        prefix_kv_diffs.append({
            "layer": i,
            "shape": kd.get("shape"),
            "K_max_abs": kd["max_abs"], "K_mean_abs": kd["mean_abs"],
            "V_max_abs": vd["max_abs"], "V_mean_abs": vd["mean_abs"],
        })
    first_kv_div = next(
        (d["layer"] for d in prefix_kv_diffs
         if (d["K_max_abs"] or 0) > 0 or (d["V_max_abs"] or 0) > 0),
        None,
    )
    print(f"  first layer with K/V diff != 0: {first_kv_div}")
    print(f"  layer 0:  K max={prefix_kv_diffs[0]['K_max_abs']:.3e}  V max={prefix_kv_diffs[0]['V_max_abs']:.3e}")
    print(f"  layer {n_layers-1}: K max={prefix_kv_diffs[-1]['K_max_abs']:.3e}  V max={prefix_kv_diffs[-1]['V_max_abs']:.3e}")

    # === E. 3 decode variants ===
    print("\n=== 3 decode variants (auto / explicit_correct / explicit_wrong) ===")
    past_len = cache_seq_len
    attn_mask_decode = torch.ones((batch_size, past_len + new_len), dtype=torch.long, device=device)
    pos_correct = torch.arange(past_len, past_len + new_len, device=device).unsqueeze(0)
    cache_pos_correct = torch.arange(past_len, past_len + new_len, device=device)
    pos_wrong = torch.zeros(batch_size, new_len, dtype=torch.long, device=device)
    cache_pos_wrong = torch.zeros(new_len, dtype=torch.long, device=device)

    variants = []
    for v_name, pos_ids, cache_pos in [
        ("auto", None, None),
        ("explicit_correct", pos_correct, cache_pos_correct),
        ("explicit_wrong", pos_wrong, cache_pos_wrong),
    ]:
        cap_v: dict = {}
        restore_v = install_rope_hook(model, cap_v)
        cache = fresh_prefix_cache(model, prefix_in, device)  # 매 variant fresh
        kwargs: dict[str, Any] = dict(
            input_ids=next_ids,
            past_key_values=cache,
            use_cache=True,
            attention_mask=attn_mask_decode,
        )
        if pos_ids is not None:
            kwargs["position_ids"] = pos_ids
        if cache_pos is not None:
            kwargs["cache_position"] = cache_pos
        set_all_seeds(SEED)
        with torch.inference_mode():
            out_v = model(**kwargs)
        restore_v()
        decode_last_logits = out_v.logits[:, -1, :].detach().to("cpu", torch.float32)
        captured_pos = cap_v["rope_calls"][0][0].tolist() if cap_v.get("rope_calls") else None
        diff = tensor_diff(decode_last_logits, full_last_logits)
        argmax_id = int(decode_last_logits.argmax(dim=-1).item())
        argmax_dec = tokenizer.decode([argmax_id])
        variants.append({
            "name": v_name,
            "captured_position_ids": captured_pos,
            "diff_vs_full_last_logits": diff,
            "argmax_id": argmax_id,
            "argmax_decoded": argmax_dec,
        })
        print(f"  {v_name}: captured_pos={captured_pos}  max_abs(vs B)={diff['max_abs']:.3e}  "
              f"argmax={argmax_id}={argmax_dec!r}")
        del out_v, cache

    by_name = {v["name"]: v for v in variants}

    # === 4. Per-layer hidden_states earliest divergence ===
    print("\n=== Per-layer hidden_states earliest divergence (explicit_correct decode vs B full last) ===")
    cap_eh: dict = {}
    restore_eh = install_rope_hook(model, cap_eh)
    cache = fresh_prefix_cache(model, prefix_in, device)
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_eh = model(
            input_ids=next_ids,
            past_key_values=cache,
            use_cache=True,
            attention_mask=attn_mask_decode,
            position_ids=pos_correct,
            cache_position=cache_pos_correct,
            output_hidden_states=True,
        )
    restore_eh()
    decode_hiddens = [h.detach().to("cpu", torch.float32) for h in out_eh.hidden_states]
    decode_last_hiddens = [h[:, -1, :] for h in decode_hiddens]
    decode_last_logits_eh = out_eh.logits[:, -1, :].detach().to("cpu", torch.float32)

    earliest_div = []
    for i in range(len(decode_last_hiddens)):
        d = tensor_diff(decode_last_hiddens[i], full_last_hiddens[i])
        earliest_div.append({
            "layer": i, "tensor_name": "hidden_state",
            "max_abs_diff": d["max_abs"], "mean_abs_diff": d["mean_abs"],
            "argmax_idx": d.get("argmax_idx"),
        })
    # logits 비교 (final)
    d_log = tensor_diff(decode_last_logits_eh, full_last_logits)
    earliest_div.append({
        "layer": -1, "tensor_name": "logits",
        "max_abs_diff": d_log["max_abs"], "mean_abs_diff": d_log["mean_abs"],
        "argmax_idx": d_log.get("argmax_idx"),
    })

    THRESH = 1e-7  # earliest divergence detection threshold
    first_hidden_div = next(
        (d["layer"] for d in earliest_div
         if d["tensor_name"] == "hidden_state" and (d["max_abs_diff"] or 0) > THRESH),
        None,
    )
    print(f"  earliest hidden_state div (>{THRESH}): layer {first_hidden_div}")
    print(f"  layer 0: max={earliest_div[0]['max_abs_diff']:.3e}")
    print(f"  layer {len(decode_last_hiddens)-1} (final hidden): max={earliest_div[-2]['max_abs_diff']:.3e}")
    print(f"  logits: max={earliest_div[-1]['max_abs_diff']:.3e}")

    # === 8. Case 추론 (휴리스틱) ===
    auto_pos = by_name["auto"]["captured_position_ids"]
    correct_pos = by_name["explicit_correct"]["captured_position_ids"]
    wrong_pos = by_name["explicit_wrong"]["captured_position_ids"]
    auto_max = by_name["auto"]["diff_vs_full_last_logits"]["max_abs"]
    correct_max = by_name["explicit_correct"]["diff_vs_full_last_logits"]["max_abs"]
    wrong_max = by_name["explicit_wrong"]["diff_vs_full_last_logits"]["max_abs"]

    auto_eq_correct_pos = (auto_pos == correct_pos)
    auto_eq_correct_diff = abs(auto_max - correct_max) < 1e-10

    case = "unknown"
    reason = []
    if first_kv_div is not None and prefix_kv_diffs[first_kv_div]["K_max_abs"] > THRESH:
        case = "C"
        reason.append(f"prefix KV already differs at layer {first_kv_div} "
                      f"(K max_abs > {THRESH}) — strongly supported by prefill "
                      f"sequence-length-dependent numerical drift hypothesis")
    elif not auto_eq_correct_pos or (correct_max < auto_max * 0.5):
        case = "A"
        reason.append("auto position differs from explicit_correct OR explicit_correct "
                      "noticeably improves diff — position handling implicated")
    elif first_hidden_div == 0:
        case = "D (current q/k/v projection of decode input)"
        reason.append("hidden_state divergence appears at layer 0 — consistent with "
                      "decode-shape vs full-shape linear projection GEMM order difference")
    elif first_hidden_div is not None:
        case = f"D/E (divergence appears at layer {first_hidden_div})"
        reason.append("hidden_state divergence appears mid-network — consistent with "
                      "attention kernel / reduction-order drift compounding across layers")
    else:
        case = "F (logits-only)"
        reason.append("all hidden_states match within threshold; only logits differ — "
                      "consistent with lm_head GEMM shape/reduction difference")

    # === artifacts 저장 ===
    print("\n=== Saving artifacts ===")
    summary = {
        "model": MODEL,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device,
        "dtype": "float32",
        "attention_implementation": "eager",
        "tokenizer_info": tok_info,
        "policy_results": policy_results,
        "prefix": PREFIX,
        "full": FULL,
        "prefix_len": prefix_len,
        "full_len": int(full_ids.shape[1]),
        "next_ids": next_ids[0].tolist(),
        "next_tokens": tokenizer.convert_ids_to_tokens(next_ids[0].tolist()),
        "past_len": int(past_len),
        "rope_position_full": rope_full_pos,
        "rope_position_prefix": rope_pref_pos,
        "rope_signature": cap_full.get("rope_signature"),
        "decode_variants": variants,
        "prefix_kv_diffs": prefix_kv_diffs,
        "first_kv_diff_layer": first_kv_div,
        "earliest_divergence_hidden": earliest_div,
        "first_hidden_div_layer": first_hidden_div,
        "max_abs_diff_explicit_correct_vs_full": correct_max,
        "case_inference": {"case": case, "reason": " | ".join(reason)},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(f"  → {out_dir/'summary.json'}")

    # CSV (earliest divergence)
    csv_path = out_dir / "earliest_divergence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "tensor_name", "max_abs_diff", "mean_abs_diff", "argmax_idx", "earliest_nonzero"])
        # hidden + logits
        for d in earliest_div:
            mad = d["max_abs_diff"] or 0
            w.writerow([
                d["layer"], d["tensor_name"], f"{mad:.6e}",
                f"{d['mean_abs_diff']:.6e}" if d["mean_abs_diff"] else "0",
                d.get("argmax_idx", ""),
                "yes" if mad > THRESH else "no",
            ])
        # prefix KV per layer
        for d in prefix_kv_diffs:
            kmax = d["K_max_abs"] or 0
            vmax = d["V_max_abs"] or 0
            w.writerow([d["layer"], "prefix_K_vs_full_slice", f"{kmax:.6e}",
                       f"{d['K_mean_abs']:.6e}" if d["K_mean_abs"] else "0", "",
                       "yes" if kmax > THRESH else "no"])
            w.writerow([d["layer"], "prefix_V_vs_full_slice", f"{vmax:.6e}",
                       f"{d['V_mean_abs']:.6e}" if d["V_mean_abs"] else "0", "",
                       "yes" if vmax > THRESH else "no"])
    print(f"  → {csv_path}")

    # position_report.md
    pos_md = out_dir / "position_report.md"
    pos_md.write_text(f"""# C-3 Position Report

## Tokenizer special tokens
- bos: `{tok_info['bos_token']}` (id={tok_info['bos_token_id']})
- eos: `{tok_info['eos_token']}` (id={tok_info['eos_token_id']})
- add_bos_token: {tok_info['add_bos_token']}
- add_eos_token: {tok_info['add_eos_token']}

## Tokenization (default policy)
- prefix_ids: {policy_results['default']['prefix_ids']}
- prefix_tokens: {policy_results['default']['prefix_tokens']}
- full_ids: {policy_results['default']['full_ids']}
- full_tokens: {policy_results['default']['full_tokens']}
- next_ids: {policy_results['default']['next_ids']}
- next_tokens: {policy_results['default']['next_tokens']}
- prefix_len: {policy_results['default']['prefix_len']}
- full_len: {policy_results['default']['full_len']}
- split_invariant (prefix_ids == full_ids[:, :prefix_len]): {policy_results['default']['split_invariant']}

## All 3 policies — split invariant
| policy | prefix_len | full_len | split_ok | next_len_1 |
|---|---|---|---|---|
| default | {policy_results['default']['prefix_len']} | {policy_results['default']['full_len']} | {policy_results['default']['split_invariant']} | {policy_results['default']['next_len_is_1']} |
| add_special_True | {policy_results['add_special_True']['prefix_len']} | {policy_results['add_special_True']['full_len']} | {policy_results['add_special_True']['split_invariant']} | {policy_results['add_special_True']['next_len_is_1']} |
| add_special_False | {policy_results['add_special_False']['prefix_len']} | {policy_results['add_special_False']['full_len']} | {policy_results['add_special_False']['split_invariant']} | {policy_results['add_special_False']['next_len_is_1']} |

## Actual position_ids captured (via RoPE forward monkey-patch)
- rotary_emb.forward signature: `{cap_full.get('rope_signature')}`
- full prefill: {rope_full_pos}
- prefix prefill: {rope_pref_pos}
- decode auto: {by_name['auto']['captured_position_ids']}
- decode explicit_correct: {by_name['explicit_correct']['captured_position_ids']}
- decode explicit_wrong: {by_name['explicit_wrong']['captured_position_ids']}

## Variant max_abs_diff (decode last logits vs B full last logits)
- auto: {auto_max:.3e}
- explicit_correct: {correct_max:.3e}
- explicit_wrong: {wrong_max:.3e}

## Position-related conclusions
- auto position == explicit_correct position: **{auto_eq_correct_pos}**
- auto diff ≈ explicit_correct diff (within 1e-10): **{auto_eq_correct_diff}**
- explicit_wrong (position=0) produces noticeably different result: **{wrong_max > correct_max * 2}** (diff factor ≈ {wrong_max/correct_max if correct_max > 0 else float('inf'):.1f}x)

## Earliest divergence
- first prefix KV diff layer (prefix-only vs full-slice): **{first_kv_div}**
- first hidden_state diff layer (decode vs B full last-token): **{first_hidden_div}** (threshold 1e-7)

## Case inference (heuristic)
- case: **{case}**
- reason: {' | '.join(reason)}

(자세한 layer-level diff는 `earliest_divergence.csv` 참조)

## 표현 원칙
이 보고는 다음 표현을 사용한다 (강한 단정 회피):
- "strongly supported by diagnostic evidence"
- "confirmed within the tested hypotheses"
- "consistent with fp32 GEMM/reduction-order numerical drift"
- "not attributable to position_ids under the tested configuration"
""")
    print(f"  → {pos_md}")

    print(f"\n=== 결론: case={case} ===")
    for r in reason:
        print(f"  - {r}")


if __name__ == "__main__":
    main()
