#!/usr/bin/env python3
"""C-7 진단: GEMM shape 통일 + position 정보 정확성 검증.

C-6에서 4 조건(입력 동일 / eager / fp32 / deterministic) 모두 통과 + q_proj first 6 max_abs=2.384e-06 재현.
유일하게 남은 미검증 조건은 사용자 전제의 "동일 순서". cuBLAS GEMM은 input shape (M)이 바뀌면
같은 input row에 대해서도 다른 reduction 순서를 선택할 수 있음.

검증 전략: split 입력을 length 7로 padding (7번째 토큰은 single의 generated_token,
attention_mask[6]=0)하여 GEMM problem (M=7)을 single과 강제 일치. 그 결과 q_proj first 6 출력이
bitwise 일치하는지 측정.

추가: 사용자 강조 — "정확한 포지션 정보가 입력되어야 한다". HF가 mask·use_cache에 따라
position_ids를 다르게 자동 생성할 수 있으므로 rotary_emb 입력·출력을 hook으로 캡처하여
3-way bitwise 비교.

3 forward (모두 use_cache=False — q_proj 격리, cache 메커니즘 영향 차단):
  - split:        input_ids (1, 6), mask=[1]*6
  - split_padded: input_ids (1, 7) = [prompt + generated], mask=[1]*6 + [0]
  - single:       input_ids (1, 7) = [prompt + generated], mask=[1]*7

비교 분기:
  E1. position_ids first 6: 3-way bitwise (모두 [0..5] 인지)
  E2. RoPE cos/sin first 6: 3-way bitwise
  F1. split vs single q_proj first 6 max_abs (C-6 재현 예상)
  F2. split_padded vs single q_proj first 6 max_abs (핵심 측정)
  F3. split vs split_padded q_proj first 6 max_abs (F1과 동등 발산 예상)

원칙: fork 무수정. forward hook으로만 캡처.
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
PREFIX_LEN = 6


def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic() -> None:
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
    arr = t.detach().cpu().to(torch.float32).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def tdiff_first_n(a: torch.Tensor, b: torch.Tensor, n: int, dim: int = 1) -> dict:
    """첫 n positions만 슬라이스하여 SHA-256·max_abs 비교 (dim=1 = seq 차원)."""
    sa = a.index_select(dim, torch.arange(n, device=a.device))
    sb = b.index_select(dim, torch.arange(n, device=b.device))
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "sliced_shape": list(sa.shape),
        "sha_a": tensor_sha256(sa),
        "sha_b": tensor_sha256(sb),
        "max_abs": float((sa - sb).abs().max()),
        "bitwise": tensor_sha256(sa) == tensor_sha256(sb),
    }


def run_forward(
    model: MistralForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    label: str,
    captures: dict[str, Any],
) -> torch.Tensor:
    """단일 forward 호출 + 5종 hook 부착.

    Hooks:
      - layers[0] forward_pre: layer 0 input
      - layers[0].input_layernorm forward: post-LN
      - layers[0].self_attn.q_proj forward: q_proj 출력
      - layers[0].self_attn.k_proj forward: k_proj 출력
      - layers[0].self_attn.v_proj forward: v_proj 출력
      - model.rotary_emb forward_pre: position_ids
      - model.rotary_emb forward: (cos, sin)
    """
    layer0 = model.model.layers[0]
    self_attn = layer0.self_attn
    rope = model.model.rotary_emb

    def hook_layer0_input(module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        captures[f"{label}_layer0_input"] = hs.detach().clone()

    def hook_post_ln(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captures[f"{label}_post_ln"] = out.detach().clone()

    def hook_q_proj(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captures[f"{label}_q_proj"] = out.detach().clone()

    def hook_k_proj(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captures[f"{label}_k_proj"] = out.detach().clone()

    def hook_v_proj(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captures[f"{label}_v_proj"] = out.detach().clone()

    def hook_rope_input(module, args, kwargs):
        # rotary_emb.forward(x, position_ids) → args=(x, position_ids)
        pos_ids = args[1] if len(args) > 1 else kwargs.get("position_ids")
        captures[f"{label}_position_ids"] = (
            pos_ids.detach().clone() if pos_ids is not None else None
        )

    def hook_rope_output(module, args, output):
        cos, sin = output
        captures[f"{label}_rope_cos"] = cos.detach().clone()
        captures[f"{label}_rope_sin"] = sin.detach().clone()

    handles = [
        layer0.register_forward_pre_hook(hook_layer0_input, with_kwargs=True),
        layer0.input_layernorm.register_forward_hook(hook_post_ln),
        self_attn.q_proj.register_forward_hook(hook_q_proj),
        self_attn.k_proj.register_forward_hook(hook_k_proj),
        self_attn.v_proj.register_forward_hook(hook_v_proj),
        rope.register_forward_pre_hook(hook_rope_input, with_kwargs=True),
        rope.register_forward_hook(hook_rope_output),
    ]
    set_all_seeds(SEED)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    for h in handles:
        h.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/c7_padded_shape_position_info")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_deterministic()
    set_all_seeds(SEED)

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids_prompt = enc["input_ids"].to("cuda")  # (1, 6)
    attn_mask_prompt = enc["attention_mask"].to("cuda")
    assert input_ids_prompt.shape[1] == PREFIX_LEN

    print(f"[1/5] 모델 로드: {args.model}")
    model = MistralForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        token=token,
    ).to("cuda").eval()

    captures: dict[str, Any] = {}

    # --- Forward 1: split (length 6, mask=ones, use_cache=False) ---
    # next_token_id는 split의 last-position logits에서 greedy argmax (run_dynamic_cache_check와 동일 규칙)
    print("[2/5] split forward (length 6, mask=ones)")
    out_split = run_forward(
        model, input_ids_prompt, attn_mask_prompt, "split", captures
    )
    next_token_id = out_split.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_id_int = int(next_token_id.item())
    print(f"  decode token: id={next_id_int} -> {tokenizer.decode([next_id_int])!r}")

    # --- Forward 2: split_padded (length 7 = prompt + generated, mask=[1*6, 0]) ---
    print("[3/5] split_padded forward (length 7, mask=[1*6, 0])")
    input_ids_padded = torch.cat([input_ids_prompt, next_token_id], dim=1)  # (1, 7)
    attn_mask_padded = torch.zeros((1, PREFIX_LEN + 1), dtype=torch.long, device="cuda")
    attn_mask_padded[0, :PREFIX_LEN] = 1
    # 결과: [1, 1, 1, 1, 1, 1, 0]
    out_padded = run_forward(
        model, input_ids_padded, attn_mask_padded, "split_padded", captures
    )

    # --- Forward 3: single (length 7, mask=ones) ---
    print("[4/5] single forward (length 7, mask=ones)")
    attn_mask_single = torch.ones((1, PREFIX_LEN + 1), dtype=torch.long, device="cuda")
    out_single = run_forward(
        model, input_ids_padded, attn_mask_single, "single", captures
    )

    # ============================================================
    # E1: position_ids 3-way bitwise first 6
    # ============================================================
    print("[5/5] 비교: E1 (position_ids), E2 (RoPE), F1/F2/F3 (q_proj)")
    pos_split = captures["split_position_ids"]               # (1, 6)
    pos_padded = captures["split_padded_position_ids"]       # (1, 7)
    pos_single = captures["single_position_ids"]             # (1, 7)
    e1 = {
        "split_first6": pos_split[0, :PREFIX_LEN].cpu().tolist(),
        "split_padded_first6": pos_padded[0, :PREFIX_LEN].cpu().tolist(),
        "single_first6": pos_single[0, :PREFIX_LEN].cpu().tolist(),
        "split_padded_position6": int(pos_padded[0, PREFIX_LEN].item()),
        "single_position6": int(pos_single[0, PREFIX_LEN].item()),
        "split_padded_full": pos_padded[0].cpu().tolist(),
        "single_full": pos_single[0].cpu().tolist(),
    }
    e1["3way_first6_bitwise"] = (
        e1["split_first6"] == e1["split_padded_first6"] == e1["single_first6"]
    )
    e1["padded_vs_single_full_match"] = e1["split_padded_full"] == e1["single_full"]

    # ============================================================
    # E2: RoPE cos/sin 3-way bitwise first 6
    # ============================================================
    cos_split = captures["split_rope_cos"]                   # (1, 6, D)
    cos_padded = captures["split_padded_rope_cos"]           # (1, 7, D)
    cos_single = captures["single_rope_cos"]                 # (1, 7, D)
    sin_split = captures["split_rope_sin"]
    sin_padded = captures["split_padded_rope_sin"]
    sin_single = captures["single_rope_sin"]

    cos_sp_vs_si = tdiff_first_n(cos_split, cos_single, PREFIX_LEN, dim=1)
    cos_pad_vs_si = tdiff_first_n(cos_padded, cos_single, PREFIX_LEN, dim=1)
    cos_sp_vs_pad = tdiff_first_n(cos_split, cos_padded, PREFIX_LEN, dim=1)
    sin_sp_vs_si = tdiff_first_n(sin_split, sin_single, PREFIX_LEN, dim=1)
    sin_pad_vs_si = tdiff_first_n(sin_padded, sin_single, PREFIX_LEN, dim=1)
    sin_sp_vs_pad = tdiff_first_n(sin_split, sin_padded, PREFIX_LEN, dim=1)

    e2 = {
        "cos": {
            "split_vs_single": cos_sp_vs_si,
            "split_padded_vs_single": cos_pad_vs_si,
            "split_vs_split_padded": cos_sp_vs_pad,
        },
        "sin": {
            "split_vs_single": sin_sp_vs_si,
            "split_padded_vs_single": sin_pad_vs_si,
            "split_vs_split_padded": sin_sp_vs_pad,
        },
        "3way_all_bitwise_first6": (
            cos_sp_vs_si["bitwise"]
            and cos_pad_vs_si["bitwise"]
            and cos_sp_vs_pad["bitwise"]
            and sin_sp_vs_si["bitwise"]
            and sin_pad_vs_si["bitwise"]
            and sin_sp_vs_pad["bitwise"]
        ),
    }

    # ============================================================
    # F1/F2/F3: q_proj / k_proj / v_proj 출력 first 6 비교
    # ============================================================
    def proj_compare(proj_name: str) -> dict:
        a = captures[f"split_{proj_name}"]               # (1, 6, dim)
        b = captures[f"split_padded_{proj_name}"]        # (1, 7, dim)
        c = captures[f"single_{proj_name}"]              # (1, 7, dim)
        return {
            "F1_split_vs_single": tdiff_first_n(a, c, PREFIX_LEN, dim=1),
            "F2_split_padded_vs_single": tdiff_first_n(b, c, PREFIX_LEN, dim=1),
            "F3_split_vs_split_padded": tdiff_first_n(a, b, PREFIX_LEN, dim=1),
        }

    f_qproj = proj_compare("q_proj")
    f_kproj = proj_compare("k_proj")
    f_vproj = proj_compare("v_proj")

    # 추가 sanity: 입력단 (post-LN) 3-way first 6 bitwise (A4의 확장)
    a4_compare = {
        "split_vs_single": tdiff_first_n(
            captures["split_post_ln"], captures["single_post_ln"], PREFIX_LEN, dim=1
        ),
        "split_padded_vs_single": tdiff_first_n(
            captures["split_padded_post_ln"], captures["single_post_ln"], PREFIX_LEN, dim=1
        ),
        "split_vs_split_padded": tdiff_first_n(
            captures["split_post_ln"], captures["split_padded_post_ln"], PREFIX_LEN, dim=1
        ),
    }

    # ============================================================
    # JSON 저장 + 요약 출력
    # ============================================================
    summary: dict[str, Any] = {
        "round": "C-7",
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "prompt": PROMPT,
        "prefix_len": PREFIX_LEN,
        "decode_token_id": next_id_int,
        "decode_token_decoded": tokenizer.decode([next_id_int]),
        "forward_setup": {
            "split":        {"input_ids_shape": [1, 6], "attention_mask": [1]*6},
            "split_padded": {"input_ids_shape": [1, 7], "attention_mask": [1]*6 + [0]},
            "single":       {"input_ids_shape": [1, 7], "attention_mask": [1]*7},
            "all_use_cache_false": True,
        },
        "E1_position_ids": e1,
        "E2_rope_cos_sin": e2,
        "A4_post_input_layernorm_first6": a4_compare,
        "F_q_proj": f_qproj,
        "F_k_proj": f_kproj,
        "F_v_proj": f_vproj,
    }

    out_file = out_dir / "diagnose_c7.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  저장: {out_file}")

    # 요약 출력
    print()
    print("=" * 60)
    print("E1: position_ids first 6 3-way bitwise")
    print(f"    split:        {e1['split_first6']}")
    print(f"    split_padded: {e1['split_padded_first6']}")
    print(f"    single:       {e1['single_first6']}")
    print(f"    3way bitwise: {'✅' if e1['3way_first6_bitwise'] else '❌'}")
    print(f"    padded[6]={e1['split_padded_position6']}, single[6]={e1['single_position6']}")
    print()
    print("E2: RoPE cos/sin first 6 3-way bitwise")
    print(f"    cos split vs single:        max_abs={cos_sp_vs_si['max_abs']:.3e} bitwise={cos_sp_vs_si['bitwise']}")
    print(f"    cos split_padded vs single: max_abs={cos_pad_vs_si['max_abs']:.3e} bitwise={cos_pad_vs_si['bitwise']}")
    print(f"    sin split vs single:        max_abs={sin_sp_vs_si['max_abs']:.3e} bitwise={sin_sp_vs_si['bitwise']}")
    print(f"    sin split_padded vs single: max_abs={sin_pad_vs_si['max_abs']:.3e} bitwise={sin_pad_vs_si['bitwise']}")
    print()
    print("A4: post-input_layernorm first 6 (3-way bitwise 기대)")
    for k, v in a4_compare.items():
        print(f"    {k:30s} max_abs={v['max_abs']:.3e} bitwise={v['bitwise']}")
    print()
    print("F: q/k/v_proj first 6 비교")
    for proj_name, f_dict in [("q_proj", f_qproj), ("k_proj", f_kproj), ("v_proj", f_vproj)]:
        print(f"  [{proj_name}]")
        for k, v in f_dict.items():
            print(f"    {k:30s} max_abs={v['max_abs']:.3e} bitwise={v['bitwise']}")

    print()
    print("=" * 60)
    print("분기 판정:")
    f2_qproj_bitwise = f_qproj["F2_split_padded_vs_single"]["bitwise"]
    f1_qproj_bitwise = f_qproj["F1_split_vs_single"]["bitwise"]
    if not e1["3way_first6_bitwise"]:
        print("  ❌ E1 실패: position_ids first 6이 3-way 일치 안 함 → position 정보 root cause")
    elif not e2["3way_all_bitwise_first6"]:
        print("  ❌ E2 실패: RoPE cos/sin first 6이 3-way 일치 안 함 → position 정보 root cause")
    elif f2_qproj_bitwise and not f1_qproj_bitwise:
        print("  ✅ F2 bitwise + F1 발산: GEMM input shape (M)이 '동일 순서' 위반의 원인 확정")
    elif f2_qproj_bitwise and f1_qproj_bitwise:
        print("  ⚠️  F1·F2 모두 bitwise: C-6 환경과 다른 무엇이 변함 → 재현 불가")
    elif not f2_qproj_bitwise:
        print(f"  ⚠️  F2 발산 (max_abs={f_qproj['F2_split_padded_vs_single']['max_abs']:.3e}): 같은 GEMM shape에서도 발산 → 다른 mechanism 탐색 필요")


if __name__ == "__main__":
    main()
