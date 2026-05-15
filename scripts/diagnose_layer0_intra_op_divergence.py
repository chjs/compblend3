#!/usr/bin/env python3
"""C-4 진단: Layer 0 intra-op divergence localization.

C-3 결과:
- Layer 0 K/V (prefix-only vs full-prefix-slice): 완전 동일 (max_abs = 0)
- Layer 1 K/V: 발산 시작 (K max ~1.9e-06)
→ 첫 divergence는 Layer 0 내부의 K/V projection 이후 ~ Layer 1 K/V projection 사이 어딘가.

C-4 목표: Layer 0의 각 sub-op (norm/q/k/v/RoPE/attn/o_proj/residual/norm2/MLP)를 step-by-step
        비교해 max_abs_diff가 처음 0이 아니게 되는 정확한 지점 식별. fp32 drift 결론 보류.

비교: A = prefix-only prefill / B = full-prefill (prefix slice).
tokenizer policy = C-3 default (BOS 포함, prefix_len=6). next_ids는 full[:, prefix_len:]에서.

8 sections (사용자 §1-§8):
1. Mask verification (causal / sliding_window, prefix mask vs full-prefix-slice)
2. RoPE cos/sin verification (prefix vs full-prefix-slice)
3. Manual Layer 0 step-by-step (22 intermediates)
4. Manual vs HF Layer 0 sanity (manual reproduces HF)
5. Attention alternatives (3 variants — visible-key vs masked-length)
6. use_cache=True/False effect at layer 0
7. sliding_window disable effect (if applicable)
8. Output artifacts → artifacts/c4_layer0_intra_op/

표현 원칙: "strongly supported", "consistent with", "not attributable" — 단정 회피.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
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
# fork에서 직접 import (apply_rotary_pos_emb, repeat_kv가 fork 본문에 정의됨)
from compblend.modeling.modeling_mistral import apply_rotary_pos_emb, repeat_kv

SEED = 42
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PREFIX = "The capital of France is"
FULL = "The capital of France is Paris"


def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    random.seed(seed); np.random.seed(seed)


def setup_deterministic() -> None:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def tdiff(a: torch.Tensor, b: torch.Tensor, atols=(0.0, 1e-8, 1e-7, 1e-6)) -> dict:
    """두 텐서의 비교 — shape 동일하면 max/mean abs diff + 다중 atol allclose."""
    a = a.detach().to("cpu", torch.float32)
    b = b.detach().to("cpu", torch.float32)
    out = {"shape_a": list(a.shape), "shape_b": list(b.shape)}
    if a.shape != b.shape:
        out["shape_mismatch"] = True
        out["max_abs"] = None; out["mean_abs"] = None
        return out
    d = (a - b).abs()
    out["max_abs"] = float(d.max())
    out["mean_abs"] = float(d.mean())
    out["argmax_idx"] = [int(x) for x in np.unravel_index(int(d.argmax().item()), a.shape)]
    for at in atols:
        out[f"allclose_atol_{at}"] = bool(torch.allclose(a, b, atol=at, rtol=0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/c4_layer0_intra_op")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정.")
        sys.exit(1)

    setup_deterministic()
    token = os.environ.get("HF_TOKEN")
    device = "cuda"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=token)
    prefix_in = {k: v.to(device) for k, v in tokenizer(PREFIX, return_tensors="pt").items()}
    full_in = {k: v.to(device) for k, v in tokenizer(FULL, return_tensors="pt").items()}
    prefix_ids = prefix_in["input_ids"]
    full_ids = full_in["input_ids"]
    prefix_len = int(prefix_ids.shape[1])
    next_ids = full_ids[:, prefix_len:]
    full_len = int(full_ids.shape[1])
    assert torch.equal(prefix_ids, full_ids[:, :prefix_len])
    assert next_ids.shape[1] == 1
    print(f"prefix_len={prefix_len}, full_len={full_len}, next_ids={next_ids[0].tolist()}")

    print("\n=== Model load ===")
    model = MistralForCausalLM.from_pretrained(
        MODEL, torch_dtype=dtype, attn_implementation="eager", token=token,
    ).to(device).eval()
    cfg = model.config
    H_q = cfg.num_attention_heads      # 32
    H_kv = cfg.num_key_value_heads     # 8
    D = cfg.head_dim                   # 128
    H = cfg.hidden_size                # 4096
    n_rep = H_q // H_kv                # 4 (GQA)
    scaling = D ** -0.5
    sliding_window = cfg.sliding_window  # may be None or int
    print(f"  H={H}, H_q={H_q}, H_kv={H_kv}, D={D}, n_rep={n_rep}, scaling={scaling}")
    print(f"  config.sliding_window: {sliding_window!r}")

    base_model = model.model
    layer0 = base_model.layers[0]
    rope = base_model.rotary_emb

    # ============================================================
    # 공통 준비: embed + position_ids + cache_position + RoPE
    # ============================================================
    print("\n=== Common setup: embeddings, position_ids, RoPE ===")
    with torch.inference_mode():
        embed_p = base_model.embed_tokens(prefix_ids)  # (1, prefix_len, H)
        embed_f = base_model.embed_tokens(full_ids)    # (1, full_len, H)

    cache_pos_p = torch.arange(prefix_len, device=device)
    cache_pos_f = torch.arange(full_len, device=device)
    pos_ids_p = cache_pos_p.unsqueeze(0)
    pos_ids_f = cache_pos_f.unsqueeze(0)

    # RoPE
    with torch.inference_mode():
        cos_p, sin_p = rope(embed_p, pos_ids_p)  # shape (1, T, D)
        cos_f, sin_f = rope(embed_f, pos_ids_f)
    print(f"  cos/sin shapes: prefix {tuple(cos_p.shape)} / full {tuple(cos_f.shape)}")

    # Embeddings 비교 (prefix slice)
    embed_cmp = tdiff(embed_p, embed_f[:, :prefix_len, :])
    print(f"  embed prefix vs full[:prefix_len]: max_abs={embed_cmp['max_abs']:.3e}")

    # ============================================================
    # §2. RoPE cos/sin verification
    # ============================================================
    print("\n=== §2. RoPE cos/sin (prefix vs full[:prefix_len]) ===")
    cos_cmp = tdiff(cos_p, cos_f[:, :prefix_len, :])
    sin_cmp = tdiff(sin_p, sin_f[:, :prefix_len, :])
    print(f"  cos: max_abs={cos_cmp['max_abs']:.3e}, allclose_atol_0={cos_cmp['allclose_atol_0']}")
    print(f"  sin: max_abs={sin_cmp['max_abs']:.3e}, allclose_atol_0={sin_cmp['allclose_atol_0']}")

    # ============================================================
    # §1. Mask verification
    # ============================================================
    print("\n=== §1. Mask verification ===")
    # MistralModel._update_causal_mask signature
    mask_method = base_model._update_causal_mask
    mask_sig = str(inspect.signature(mask_method))
    print(f"  _update_causal_mask signature: {mask_sig}")

    # Call with keyword args robustly
    def call_mask(attn_mask_in, input_tensor, cache_pos):
        return mask_method(
            attn_mask_in, input_tensor, cache_pos, None, False
        )

    mask_p = call_mask(prefix_in["attention_mask"], embed_p, cache_pos_p)
    mask_f = call_mask(full_in["attention_mask"], embed_f, cache_pos_f)

    mask_p_info = {
        "shape": list(mask_p.shape) if mask_p is not None else None,
        "dtype": str(mask_p.dtype) if mask_p is not None else None,
        "min": float(mask_p.min()) if mask_p is not None else None,
        "max": float(mask_p.max()) if mask_p is not None else None,
    }
    mask_f_info = {
        "shape": list(mask_f.shape) if mask_f is not None else None,
        "dtype": str(mask_f.dtype) if mask_f is not None else None,
        "min": float(mask_f.min()) if mask_f is not None else None,
        "max": float(mask_f.max()) if mask_f is not None else None,
    }
    print(f"  mask_prefix: {mask_p_info}")
    print(f"  mask_full: {mask_f_info}")

    mask_prefix_slice_cmp = None
    mask_future_col_check = None
    if mask_p is not None and mask_f is not None and mask_p.dim() == 4 and mask_f.dim() == 4:
        # mask shape: (B, 1, T_q, T_k). prefix: (1,1,prefix_len,prefix_len). full: (1,1,full_len,full_len).
        mask_f_slice = mask_f[:, :, :prefix_len, :prefix_len]
        mask_prefix_slice_cmp = tdiff(mask_p, mask_f_slice)
        print(f"  mask_prefix vs mask_full[:, :, :prefix_len, :prefix_len]: "
              f"max_abs={mask_prefix_slice_cmp['max_abs']}, allclose_0={mask_prefix_slice_cmp['allclose_atol_0']}")
        # Future column check: mask_full[:, :, :prefix_len, prefix_len:] should be masked (-inf or very negative)
        future_col = mask_f[:, :, :prefix_len, prefix_len:]  # (1,1,prefix_len, full_len-prefix_len)
        future_max = float(future_col.max())
        # "fully masked" — all values are -inf or very negative
        fully_masked = future_max < -1e30
        mask_future_col_check = {
            "future_col_shape": list(future_col.shape),
            "future_col_max": future_max,
            "fully_masked_future": fully_masked,
        }
        print(f"  mask_full future column (key={prefix_len}): max={future_max:.3e}, fully_masked={fully_masked}")
    else:
        print("  (mask 비교 생략 — mask가 None이거나 4D가 아님)")

    # ============================================================
    # §3. Manual Layer 0 step-by-step
    # ============================================================
    print("\n=== §3. Manual Layer 0 step-by-step ===")

    def manual_layer0(h_in, cos, sin, mask, label: str) -> dict:
        """Manual layer 0 forward. 모든 intermediate를 dict에 저장."""
        B, T, _ = h_in.shape
        with torch.inference_mode():
            i = {}
            i["00_hidden_in"] = h_in
            i["01_input_layernorm"] = layer0.input_layernorm(h_in)
            sa = layer0.self_attn
            i["02_q_proj"] = sa.q_proj(i["01_input_layernorm"])  # (B, T, H_q*D)
            i["03_k_proj"] = sa.k_proj(i["01_input_layernorm"])  # (B, T, H_kv*D)
            i["04_v_proj"] = sa.v_proj(i["01_input_layernorm"])  # (B, T, H_kv*D)
            # reshape to (B, H, T, D)
            q = i["02_q_proj"].view(B, T, H_q, D).transpose(1, 2)
            k = i["03_k_proj"].view(B, T, H_kv, D).transpose(1, 2)
            v = i["04_v_proj"].view(B, T, H_kv, D).transpose(1, 2)
            i["05_q_reshape"] = q
            i["06_k_reshape"] = k
            i["07_v_reshape"] = v
            # RoPE
            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
            i["08_q_rot"] = q_rot
            i["09_k_rot"] = k_rot
            # repeat_kv
            k_rep = repeat_kv(k_rot, n_rep)  # (B, H_q, T, D)
            v_rep = repeat_kv(v, n_rep)
            i["10_k_rep"] = k_rep
            i["11_v_rep"] = v_rep
            # raw attention scores
            attn_raw = torch.matmul(q_rot, k_rep.transpose(2, 3)) * scaling  # (B, H_q, T, T)
            i["12_attn_raw"] = attn_raw
            # apply mask
            if mask is not None:
                # mask: (B, 1, T_q, T_k); HF eager uses mask[:, :, :, :k.shape[-2]]
                cm = mask[:, :, :, :k_rep.shape[-2]]
                attn_masked = attn_raw + cm
            else:
                attn_masked = attn_raw
            i["13_attn_masked"] = attn_masked
            # softmax (HF eager: in fp32 then cast back)
            attn_probs = torch.softmax(attn_masked, dim=-1, dtype=torch.float32).to(q_rot.dtype)
            i["14_attn_probs"] = attn_probs
            # attn output
            attn_out = torch.matmul(attn_probs, v_rep)  # (B, H_q, T, D)
            attn_out = attn_out.transpose(1, 2).contiguous()  # (B, T, H_q, D)
            attn_out_flat = attn_out.view(B, T, H_q * D)
            i["15_attn_output_pre_oproj"] = attn_out_flat
            i["16_o_proj"] = sa.o_proj(attn_out_flat)
            # residual after attention
            i["17_post_attn_residual"] = h_in + i["16_o_proj"]
            # post-attention layernorm
            i["18_post_attn_layernorm"] = layer0.post_attention_layernorm(i["17_post_attn_residual"])
            # MLP
            mlp = layer0.mlp
            i["19_gate_proj"] = mlp.gate_proj(i["18_post_attn_layernorm"])
            i["20_up_proj"] = mlp.up_proj(i["18_post_attn_layernorm"])
            i["21_act_silu"] = mlp.act_fn(i["19_gate_proj"]) * i["20_up_proj"]
            i["22_down_proj"] = mlp.down_proj(i["21_act_silu"])
            # final residual = layer 0 output
            i["23_layer0_output"] = i["17_post_attn_residual"] + i["22_down_proj"]
        print(f"  manual {label}: {len(i)} intermediates captured")
        return i

    manual_p = manual_layer0(embed_p, cos_p, sin_p, mask_p, "prefix")
    manual_f = manual_layer0(embed_f, cos_f, sin_f, mask_f, "full")

    # ============================================================
    # §4. Manual vs HF Layer 0 sanity check
    # ============================================================
    print("\n=== §4. Manual vs HF Layer 0 (sanity) ===")
    # HF layer 0 forward directly
    layer0_sig = str(inspect.signature(layer0.forward))
    print(f"  layer0.forward signature: {layer0_sig}")

    # MistralDecoderLayer.forward: (hidden_states, attention_mask, position_ids, past_key_value,
    #                               output_attentions, use_cache, cache_position, position_embeddings, **kwargs)
    with torch.inference_mode():
        hf_p_out = layer0(
            hidden_states=embed_p,
            attention_mask=mask_p,
            position_ids=pos_ids_p,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_pos_p,
            position_embeddings=(cos_p, sin_p),
        )
        hf_f_out = layer0(
            hidden_states=embed_f,
            attention_mask=mask_f,
            position_ids=pos_ids_f,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_pos_f,
            position_embeddings=(cos_f, sin_f),
        )
    # HF DecoderLayer.forward returns tuple (hidden_states, [self_attn_weights], [present_key_value])
    hf_p = hf_p_out[0] if isinstance(hf_p_out, tuple) else hf_p_out
    hf_f = hf_f_out[0] if isinstance(hf_f_out, tuple) else hf_f_out

    sanity_p = tdiff(manual_p["23_layer0_output"], hf_p)
    sanity_f = tdiff(manual_f["23_layer0_output"], hf_f)
    print(f"  sanity prefix (manual vs HF): max_abs={sanity_p['max_abs']:.3e}, allclose_0={sanity_p['allclose_atol_0']}")
    print(f"  sanity full   (manual vs HF): max_abs={sanity_f['max_abs']:.3e}, allclose_0={sanity_f['allclose_atol_0']}")

    # ============================================================
    # §3 비교: 각 intermediate prefix vs full[:prefix_len]
    # ============================================================
    print("\n=== §3 intermediate diff: prefix vs full[:prefix_len] ===")
    intra_diffs = []
    first_div_idx = None
    THRESH = 1e-10
    for k_name in sorted(manual_p.keys()):
        tp = manual_p[k_name]
        tf = manual_f[k_name]
        # slice full to prefix_len in T dimension
        # T dim 위치: hidden tensors (B,T,H) or (B,T,*), attn tensors (B,H,T,D), attn_raw/probs (B,H_q,T,T)
        # Generic: find T dim by matching size to prefix_len/full_len
        sliced = slice_full_tensor(tf, full_len, prefix_len)
        d = tdiff(tp, sliced)
        intra_diffs.append({"name": k_name, **d})
        max_a = d.get("max_abs")
        if first_div_idx is None and max_a is not None and max_a > THRESH:
            first_div_idx = k_name
        print(f"  {k_name}: max_abs={max_a if max_a is None else f'{max_a:.3e}'}, "
              f"allclose_0={d.get('allclose_atol_0')}")

    print(f"\n  >>> first intermediate with max_abs > {THRESH}: {first_div_idx} <<<")

    # ============================================================
    # §5. Attention alternatives (visible-key vs masked-length)
    # ============================================================
    print("\n=== §5. Attention alternatives ===")
    # Using q_rot, k_rot, v from manual_p and manual_f
    # A_prefix: HF eager attention using prefix tensors (= manual_p["14_attn_probs"], manual_p["15_attn_output_pre_oproj"])
    # B_prefix: reference visible-key attention — same as A_prefix since prefix only has 6 keys
    # A_full_slice: full attention output, slice query rows 0..5
    # B_full_slice: reference attention using full's K/V[0..5] only (drop K[6])
    # C_full: full attention with key 6 included but masked → = A_full_slice (already computed)

    q_rot_p = manual_p["08_q_rot"]    # (1, H_q, prefix_len, D)
    k_rep_p = manual_p["10_k_rep"]    # (1, H_q, prefix_len, D)
    v_rep_p = manual_p["11_v_rep"]
    q_rot_f = manual_f["08_q_rot"]    # (1, H_q, full_len, D)
    k_rep_f = manual_f["10_k_rep"]
    v_rep_f = manual_f["11_v_rep"]
    attn_output_p = manual_p["15_attn_output_pre_oproj"]  # (1, prefix_len, H)
    attn_output_f = manual_f["15_attn_output_pre_oproj"]  # (1, full_len, H)

    # Reference: visible-key attention computed manually with prefix-length K
    def ref_attention(q, k, v, mask=None):
        """Reference eager attention (HF formula). q,k,v: (B, H, T_q, D)/(B, H, T_k, D)."""
        with torch.inference_mode():
            attn = torch.matmul(q, k.transpose(2, 3)) * scaling
            if mask is not None:
                cm = mask[:, :, :, :k.shape[-2]]
                attn = attn + cm
            probs = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
            out = torch.matmul(probs, v)
            return out.transpose(1, 2).contiguous().view(q.shape[0], q.shape[2], H_q * D)

    # B_prefix: 같은 q/k/v로 prefix 위치만 (= A_prefix와 동일)
    B_prefix_out = ref_attention(q_rot_p, k_rep_p, v_rep_p, mask_p)
    # A_prefix는 manual_p["15_..."]
    A_vs_B_prefix = tdiff(attn_output_p, B_prefix_out)

    # B_full_slice: full의 K/V를 prefix_len까지만 잘라서 (drop K[6]) 사용. q는 full의 prefix 부분만.
    q_rot_f_pref = q_rot_f[:, :, :prefix_len, :]
    k_rep_f_pref = k_rep_f[:, :, :prefix_len, :]
    v_rep_f_pref = v_rep_f[:, :, :prefix_len, :]
    # mask: prefix-shape mask (B,1,prefix_len,prefix_len) 사용 (length=6)
    B_full_slice_out = ref_attention(q_rot_f_pref, k_rep_f_pref, v_rep_f_pref, mask_p)

    # A_full_slice: full attention의 prefix rows
    A_full_slice_out = attn_output_f[:, :prefix_len, :]

    # 비교
    cmp_A_pref_vs_A_full_slice = tdiff(attn_output_p, A_full_slice_out)
    cmp_A_pref_vs_B_full_slice = tdiff(attn_output_p, B_full_slice_out)
    cmp_A_full_slice_vs_B_full_slice = tdiff(A_full_slice_out, B_full_slice_out)

    print(f"  A_prefix vs A_full_slice (=현 측정 divergence): max_abs={cmp_A_pref_vs_A_full_slice['max_abs']:.3e}")
    print(f"  A_prefix vs B_full_slice (full q/k/v 잘라 length=6 attention): max_abs={cmp_A_pref_vs_B_full_slice['max_abs']:.3e}")
    print(f"  A_full_slice vs B_full_slice (같은 q/k/v, mask=length 7 vs 6 softmax): max_abs={cmp_A_full_slice_vs_B_full_slice['max_abs']:.3e}")
    print(f"  A_prefix vs B_prefix (sanity, 동일 path): max_abs={A_vs_B_prefix['max_abs']:.3e}")

    # ============================================================
    # §6. use_cache=True/False at layer 0 output
    # ============================================================
    print("\n=== §6. use_cache effect (layer 0 output) ===")
    with torch.inference_mode():
        out_p_nc = base_model(prefix_in["input_ids"], use_cache=False, output_hidden_states=True)
        out_p_wc = base_model(prefix_in["input_ids"], use_cache=True, output_hidden_states=True)
        out_f_nc = base_model(full_in["input_ids"], use_cache=False, output_hidden_states=True)
        out_f_wc = base_model(full_in["input_ids"], use_cache=True, output_hidden_states=True)
    # hidden_states[1] = output of layer 0 (hidden_states[0] is embedding)
    L0_p_nc = out_p_nc.hidden_states[1]
    L0_p_wc = out_p_wc.hidden_states[1]
    L0_f_nc = out_f_nc.hidden_states[1]
    L0_f_wc = out_f_wc.hidden_states[1]
    use_cache_diffs = {
        "prefix_nc_vs_wc": tdiff(L0_p_nc, L0_p_wc),
        "full_nc_vs_wc": tdiff(L0_f_nc, L0_f_wc),
        "L0_prefix_nc_vs_full_nc_slice": tdiff(L0_p_nc, L0_f_nc[:, :prefix_len, :]),
        "L0_prefix_wc_vs_full_wc_slice": tdiff(L0_p_wc, L0_f_wc[:, :prefix_len, :]),
    }
    for k, v in use_cache_diffs.items():
        print(f"  {k}: max_abs={v['max_abs']:.3e}, allclose_0={v.get('allclose_atol_0')}")

    # ============================================================
    # §7. sliding_window disable
    # ============================================================
    print("\n=== §7. sliding_window disable test ===")
    sliding_window_test = {"original_sliding_window": sliding_window}
    if sliding_window is not None:
        print(f"  config.sliding_window = {sliding_window} → 임시 None으로 설정 후 재실행")
        cfg.sliding_window = None
        try:
            with torch.inference_mode():
                out_p_sw = base_model(prefix_in["input_ids"], use_cache=False, output_hidden_states=True)
                out_f_sw = base_model(full_in["input_ids"], use_cache=False, output_hidden_states=True)
            L0_p_sw = out_p_sw.hidden_states[1]
            L0_f_sw = out_f_sw.hidden_states[1]
            sw_diff = tdiff(L0_p_sw, L0_f_sw[:, :prefix_len, :])
            sliding_window_test["L0_prefix_vs_full_slice_no_sw"] = sw_diff
            print(f"  sliding_window=None: L0 prefix vs full slice max_abs={sw_diff['max_abs']:.3e}")
        finally:
            cfg.sliding_window = sliding_window  # 복구
            print(f"  config.sliding_window 복구 → {cfg.sliding_window}")
    else:
        sliding_window_test["note"] = "config.sliding_window=None — 별도 disable 불필요"
        print("  config.sliding_window=None — 비교 생략")

    # ============================================================
    # §8. artifacts 저장
    # ============================================================
    print("\n=== §8. Saving artifacts ===")
    summary = {
        "model": MODEL,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": device, "dtype": str(dtype),
        "config": {
            "H": H, "H_q": H_q, "H_kv": H_kv, "D": D, "n_rep": n_rep,
            "scaling": scaling, "sliding_window": sliding_window,
        },
        "prefix": PREFIX, "full": FULL,
        "prefix_len": prefix_len, "full_len": full_len,
        "next_ids": next_ids[0].tolist(),
        "rope_signature": str(inspect.signature(rope.forward)),
        "mask_signature": mask_sig,
        # §1
        "mask_prefix_info": mask_p_info,
        "mask_full_info": mask_f_info,
        "mask_prefix_slice_cmp": mask_prefix_slice_cmp,
        "mask_future_col_check": mask_future_col_check,
        # §2
        "embed_prefix_vs_full_slice": embed_cmp,
        "rope_cos_prefix_vs_full_slice": cos_cmp,
        "rope_sin_prefix_vs_full_slice": sin_cmp,
        # §3 intra-op diffs
        "intra_op_diffs": intra_diffs,
        "first_intra_op_divergence": first_div_idx,
        # §4 sanity
        "manual_vs_hf_prefix": sanity_p,
        "manual_vs_hf_full": sanity_f,
        # §5 attention alternatives
        "attn_A_prefix_vs_A_full_slice": cmp_A_pref_vs_A_full_slice,
        "attn_A_prefix_vs_B_full_slice": cmp_A_pref_vs_B_full_slice,
        "attn_A_full_slice_vs_B_full_slice": cmp_A_full_slice_vs_B_full_slice,
        "attn_A_prefix_vs_B_prefix_sanity": A_vs_B_prefix,
        # §6 use_cache effect
        "use_cache_diffs": use_cache_diffs,
        # §7 sliding window
        "sliding_window_test": sliding_window_test,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(f"  → {out_dir/'summary.json'}")

    # CSV
    csv_path = out_dir / "intra_op_divergence.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["intermediate", "shape", "max_abs_diff", "mean_abs_diff",
                    "allclose_atol_0", "allclose_atol_1e-8", "allclose_atol_1e-7", "allclose_atol_1e-6",
                    "argmax_idx"])
        for d in intra_diffs:
            w.writerow([d["name"], d.get("shape_a"), d.get("max_abs"), d.get("mean_abs"),
                        d.get("allclose_atol_0.0"), d.get("allclose_atol_1e-08"),
                        d.get("allclose_atol_1e-07"), d.get("allclose_atol_1e-06"),
                        d.get("argmax_idx")])
    print(f"  → {csv_path}")

    # report.md — 복잡한 nested f-string 회피 위해 사전 문자열 생성
    intra_bullets = []
    for d in intra_diffs[:24]:
        ma = d.get("max_abs")
        ma_str = "None" if ma is None else f"{ma:.3e}"
        ac0 = d.get("allclose_atol_0.0")
        intra_bullets.append(f"- {d['name']}: max_abs={ma_str}, allclose_0={ac0}")
    intra_bullets_str = "\n".join(intra_bullets)

    sw_note_line = sliding_window_test.get("note")
    if sw_note_line is None:
        sw_d = sliding_window_test.get("L0_prefix_vs_full_slice_no_sw", {})
        sw_max = sw_d.get("max_abs", "N/A")
        sw_max_str = f"{sw_max:.3e}" if isinstance(sw_max, float) else str(sw_max)
        sw_note_line = f"with sliding_window=None: L0 prefix vs full slice max_abs={sw_max_str}"

    mask_slice_max = mask_prefix_slice_cmp.get("max_abs") if mask_prefix_slice_cmp else None
    mask_slice_allclose = mask_prefix_slice_cmp.get("allclose_atol_0.0") if mask_prefix_slice_cmp else None
    mask_future_max = mask_future_col_check.get("future_col_max") if mask_future_col_check else None
    mask_future_full = mask_future_col_check.get("fully_masked_future") if mask_future_col_check else None

    md = out_dir / "c4_report.md"
    md.write_text(f"""# C-4 Layer 0 Intra-op Divergence Report

## Config
- model: {MODEL}, transformers {transformers.__version__}, torch {torch.__version__}
- dtype: {dtype}, attn_implementation: eager, sliding_window: {sliding_window}
- prefix_len: {prefix_len}, full_len: {full_len}, next_ids: {next_ids[0].tolist()}

## §1. Mask
- mask_prefix shape: {mask_p_info['shape']}, min={mask_p_info['min']}, max={mask_p_info['max']}
- mask_full shape: {mask_f_info['shape']}, min={mask_f_info['min']}, max={mask_f_info['max']}
- mask_prefix == mask_full[:, :, :prefix_len, :prefix_len]: max_abs={mask_slice_max}, allclose_0={mask_slice_allclose}
- mask_full future column (query rows 0..{prefix_len-1}, key {prefix_len}): max={mask_future_max}, fully_masked={mask_future_full}

## §2. RoPE cos/sin
- cos prefix vs full[:prefix_len]: max_abs={cos_cmp['max_abs']:.3e}, allclose_0={cos_cmp['allclose_atol_0.0']}
- sin prefix vs full[:prefix_len]: max_abs={sin_cmp['max_abs']:.3e}, allclose_0={sin_cmp['allclose_atol_0.0']}

## §4. Manual Layer 0 vs HF Layer 0 (sanity)
- prefix: max_abs={sanity_p['max_abs']:.3e}, allclose_0={sanity_p['allclose_atol_0.0']}
- full: max_abs={sanity_f['max_abs']:.3e}, allclose_0={sanity_f['allclose_atol_0.0']}

(manual reproduces HF if these are ~ 0)

## §3. Intra-op divergence — **첫 발산 지점**: `{first_div_idx}`

전체 표는 `intra_op_divergence.csv` 참조. 핵심 단계:
{intra_bullets_str}

## §5. Attention alternatives
- A_prefix vs A_full_slice (HF의 측정 divergence): max_abs={cmp_A_pref_vs_A_full_slice['max_abs']:.3e}
- A_prefix vs B_full_slice (full q/k/v를 length=6으로 잘라 attention): max_abs={cmp_A_pref_vs_B_full_slice['max_abs']:.3e}
- A_full_slice vs B_full_slice (같은 q/k/v, mask length=7 vs 6 softmax): max_abs={cmp_A_full_slice_vs_B_full_slice['max_abs']:.3e}
- A_prefix vs B_prefix (sanity): max_abs={A_vs_B_prefix['max_abs']:.3e}

해석:
- A_prefix vs B_full_slice 작으면 → full q/k/v를 잘라서 length=6 softmax하면 prefix와 ~동일 → softmax length가 핵심
- A_full_slice vs B_full_slice가 큰 만큼 softmax length 7 vs 6의 차이가 measurable

## §6. use_cache effect
- prefix nc vs wc: max_abs={use_cache_diffs['prefix_nc_vs_wc']['max_abs']:.3e}
- full nc vs wc: max_abs={use_cache_diffs['full_nc_vs_wc']['max_abs']:.3e}
- L0 prefix_nc vs full_nc[:prefix_len]: max_abs={use_cache_diffs['L0_prefix_nc_vs_full_nc_slice']['max_abs']:.3e}
- L0 prefix_wc vs full_wc[:prefix_len]: max_abs={use_cache_diffs['L0_prefix_wc_vs_full_wc_slice']['max_abs']:.3e}

## §7. sliding_window
- original sliding_window: {sliding_window}
- {sw_note_line}

## 표현 원칙
보고서에서 "100% confirmed", "guaranteed", "impossible"는 회피. 대신:
- "strongly supported by diagnostic evidence"
- "confirmed within the tested hypotheses"
- "consistent with [specific mechanism]"
- "not attributable to [X] under the tested configuration"
""")
    print(f"  → {md}")

    print("\n=== Done. 핵심 결과: ===")
    print(f"  첫 intra-op divergence: {first_div_idx}")
    print(f"  mask prefix slice 일치: {mask_prefix_slice_cmp.get('allclose_atol_0.0') if mask_prefix_slice_cmp else 'N/A'}")
    print(f"  RoPE cos/sin 일치 (bitwise): {cos_cmp['allclose_atol_0.0']} / {sin_cmp['allclose_atol_0.0']}")
    print(f"  manual vs HF sanity (prefix): max_abs={sanity_p['max_abs']:.3e}")


def slice_full_tensor(t: torch.Tensor, full_len: int, prefix_len: int) -> torch.Tensor:
    """T-dim을 찾아 prefix_len까지 slice. 각 layer 0 intermediate shape에 맞춤."""
    # 보편적 휴리스틱: dim 크기가 full_len인 첫 dim을 T로 간주
    for d in range(t.dim()):
        if t.shape[d] == full_len:
            return t.index_select(d, torch.arange(prefix_len, device=t.device))
    # attn_raw/attn_masked/attn_probs shape: (B, H_q, T, T) — 마지막 두 dim 모두 full_len
    # 이 경우 (B, H_q, full_len, full_len) → (B, H_q, prefix_len, prefix_len)
    if t.dim() == 4 and t.shape[-1] == full_len and t.shape[-2] == full_len:
        return t[:, :, :prefix_len, :prefix_len]
    return t  # no slice


if __name__ == "__main__":
    main()
