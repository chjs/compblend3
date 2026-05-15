#!/usr/bin/env python3
"""C-6 진단: 사용자 전제 4 조건 명시 검증.

전제: "첫 입력이 완전히 같고, 모든 레이어 계산이 동일한 순서·동일한 정밀도로
       이루어진다면 최종 logits은 같아야 한다. eager 모드 + deterministic 모드 필수."

검증 조건:
  A. 첫 입력 동일성 — split prefill의 layer 0 input vs single의 layer 0 input[:, :6, :]
  B. eager 모드 실제 동작 — MistralAttention class (Sdpa/FA2 ❌), Linear (monkey-patch ❌)
  C. 동일 정밀도 — fp32 일관, TF32 OFF
  D. deterministic 모드 — setup 직후 + 각 forward 직전 3 시점 실측

추가 측정 (보너스): C-4의 q_proj 출력 max_abs (split vs single first 6 position)를
    C-6의 검증된 4 조건 하에서 재측정 — TF32 등 hygiene 추가로 변화 있는지 확인.

원칙: fork 무수정. forward hook으로 layer 0 input·input_layernorm·q_proj 출력 캡처.
"""
from __future__ import annotations

import argparse
import hashlib
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
PROMPT = "The capital of France is"
PREFIX_LEN = 6  # tokenizer("The capital of France is", BOS 포함) -> 6 tokens


def set_all_seeds(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic() -> None:
    """4 조건 D를 적극적으로 강제 (C-4와 동일 hygiene)."""
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


def snapshot_deterministic_state(label: str) -> dict:
    """D 조건 실측 — 시점 라벨 포함. RNG 상태는 변경하지 않음 (read-only)."""
    # python random / numpy state는 직접 hash (str repr이 크지만 결정적)
    py_rand_state = random.getstate()
    np_rand_state = np.random.get_state()
    return {
        "label": label,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "torch_initial_seed": int(torch.initial_seed()),
        "torch_cuda_initial_seed": int(torch.cuda.initial_seed()),
        "python_random_state_hash": hashlib.sha256(
            repr(py_rand_state).encode()
        ).hexdigest()[:16],
        "numpy_random_state_hash": hashlib.sha256(
            np_rand_state[1].tobytes()
        ).hexdigest()[:16],
    }


def gpu_determinism_quick_check() -> dict:
    """matmul 동일 입력 2회 호출 → bitwise 일치 sanity. RNG 상태 변경됨 (호출 후 reseed 필요)."""
    torch.manual_seed(0)
    a = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    out1 = torch.matmul(a, b)
    out2 = torch.matmul(a, b)
    return {
        "matmul_same_input_bitwise": bool(torch.equal(out1, out2)),
        "max_abs_diff": float((out1 - out2).abs().max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/c6_input_eager_precision_deterministic")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정. 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' 후 재실행.")
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_deterministic()
    set_all_seeds(SEED)

    # ----- D 시점 1: setup·seed 직후 -----
    d_state_post_setup = snapshot_deterministic_state("post_setup")

    # GPU determinism quick check (RNG 소모하므로 직후 reseed)
    matmul_check = gpu_determinism_quick_check()
    set_all_seeds(SEED)

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)

    # split·single 공통 입력 prefix (BOS 포함 6 tokens)
    enc = tokenizer(PROMPT, return_tensors="pt")
    input_ids_prefix = enc["input_ids"].to("cuda")            # (1, 6)
    attn_mask_prefix = enc["attention_mask"].to("cuda")       # (1, 6)
    assert input_ids_prefix.shape[1] == PREFIX_LEN, (
        f"prefix len mismatch: {input_ids_prefix.shape[1]} vs {PREFIX_LEN}"
    )

    print(f"[1/4] 모델 로드: {args.model}")
    model = MistralForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        token=token,
    ).to("cuda").eval()

    # ============================================================
    # B 조건: eager 모드 실제 동작 확인
    # ============================================================
    attn_impl = getattr(model.config, "_attn_implementation", None)
    self_attn_module = model.model.layers[0].self_attn
    q_proj_module = self_attn_module.q_proj
    self_attn_cls = type(self_attn_module).__name__
    q_proj_cls = type(q_proj_module).__name__
    q_proj_is_linear = isinstance(q_proj_module, torch.nn.Linear)
    # forward 메서드가 어디 정의되어 있는지 (monkey-patch 검출)
    self_attn_forward_source = inspect.getsourcefile(type(self_attn_module).forward) or ""
    q_proj_forward_source = inspect.getsourcefile(type(q_proj_module).forward) or ""
    eager_mode_state = {
        "attn_implementation_config": attn_impl,
        "self_attn_class_name": self_attn_cls,
        "self_attn_expected": "MistralAttention",
        "self_attn_match": self_attn_cls == "MistralAttention",
        "self_attn_forward_source": self_attn_forward_source,
        "q_proj_class_name": q_proj_cls,
        "q_proj_is_torch_nn_Linear": q_proj_is_linear,
        "q_proj_forward_source": q_proj_forward_source,
        # Sdpa/FlashAttention2 class name이 아닌지 명시 음성 체크
        "self_attn_is_not_sdpa": "Sdpa" not in self_attn_cls,
        "self_attn_is_not_flash": "Flash" not in self_attn_cls,
    }

    # ============================================================
    # C 조건: 정밀도 / weight 동일성
    # ============================================================
    first_param = next(model.parameters())
    precision_state = {
        "model_dtype": str(model.dtype),
        "first_param_dtype": str(first_param.dtype),
        "embed_weight_dtype": str(model.model.embed_tokens.weight.dtype),
        "layer0_qproj_weight_dtype": str(q_proj_module.weight.dtype),
        "layer0_qproj_weight_data_ptr": q_proj_module.weight.data_ptr(),
        "layer0_qproj_weight_sha256": tensor_sha256(q_proj_module.weight),
        "layer0_kproj_weight_sha256": tensor_sha256(self_attn_module.k_proj.weight),
        "layer0_vproj_weight_sha256": tensor_sha256(self_attn_module.v_proj.weight),
    }

    # ============================================================
    # A 조건: forward hook으로 layer 0 input·input_layernorm·q_proj 출력 캡처
    # ============================================================
    captures: dict[str, torch.Tensor] = {}

    def make_input_hook(key: str):
        """forward_pre_hook — module 입력의 args[0] (= hidden_states) 캡처."""
        def hook(module, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            captures[key] = hs.detach().clone()
        return hook

    def make_output_hook(key: str):
        """forward_hook — module 출력 캡처."""
        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            captures[key] = out.detach().clone()
        return hook

    # ----- split prefill 캡처 -----
    print("[2/4] split prefill forward (hook attached: layer0 input / input_layernorm / q_proj)")
    set_all_seeds(SEED)
    d_state_pre_split_prefill = snapshot_deterministic_state("pre_split_prefill")
    handles = [
        model.model.layers[0].register_forward_pre_hook(
            make_input_hook("split_prefill_layer0_input"), with_kwargs=True
        ),
        model.model.layers[0].input_layernorm.register_forward_hook(
            make_output_hook("split_prefill_layer0_post_ln")
        ),
        q_proj_module.register_forward_hook(
            make_output_hook("split_prefill_layer0_q_proj_out")
        ),
        self_attn_module.k_proj.register_forward_hook(
            make_output_hook("split_prefill_layer0_k_proj_out")
        ),
        self_attn_module.v_proj.register_forward_hook(
            make_output_hook("split_prefill_layer0_v_proj_out")
        ),
    ]
    with torch.inference_mode():
        out_prefill = model(
            input_ids=input_ids_prefix,
            attention_mask=attn_mask_prefix,
            use_cache=True,
        )
        next_token_id = out_prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for h in handles:
        h.remove()
    next_id_int = int(next_token_id.item())

    # ----- single forward 캡처 (7 토큰 = prefix + decode 결과) -----
    print("[3/4] single forward over 7 tokens (hook attached: 동일 5 hooks)")
    set_all_seeds(SEED)
    d_state_pre_single = snapshot_deterministic_state("pre_single")
    ids_full_input = torch.cat([input_ids_prefix, next_token_id], dim=1)  # (1, 7)
    ids_full_attn = torch.ones_like(ids_full_input)
    handles = [
        model.model.layers[0].register_forward_pre_hook(
            make_input_hook("single_layer0_input"), with_kwargs=True
        ),
        model.model.layers[0].input_layernorm.register_forward_hook(
            make_output_hook("single_layer0_post_ln")
        ),
        q_proj_module.register_forward_hook(
            make_output_hook("single_layer0_q_proj_out")
        ),
        self_attn_module.k_proj.register_forward_hook(
            make_output_hook("single_layer0_k_proj_out")
        ),
        self_attn_module.v_proj.register_forward_hook(
            make_output_hook("single_layer0_v_proj_out")
        ),
    ]
    with torch.inference_mode():
        out_single = model(
            input_ids=ids_full_input,
            attention_mask=ids_full_attn,
            use_cache=False,
        )
    for h in handles:
        h.remove()

    # ============================================================
    # A 조건 비교
    # ============================================================
    print("[4/4] A 조건 비교 + 보너스 q_proj 측정")

    # A1: input_ids first 6 tokens
    a1_split_ids = input_ids_prefix[0, :PREFIX_LEN].cpu().tolist()
    a1_single_ids = ids_full_input[0, :PREFIX_LEN].cpu().tolist()
    a1_match = a1_split_ids == a1_single_ids

    # A2: embed_tokens output first 6 positions
    with torch.inference_mode():
        embed_split = model.model.embed_tokens(input_ids_prefix)            # (1, 6, H)
        embed_single = model.model.embed_tokens(ids_full_input)             # (1, 7, H)
    a2_split_sha = tensor_sha256(embed_split[:, :PREFIX_LEN, :])
    a2_single_sha = tensor_sha256(embed_single[:, :PREFIX_LEN, :])
    a2_max_abs = float(
        (embed_split[:, :PREFIX_LEN, :] - embed_single[:, :PREFIX_LEN, :]).abs().max()
    )
    a2_match = a2_split_sha == a2_single_sha

    # A3: layer 0 진입 직전 hidden_states first 6 positions
    a3_split = captures["split_prefill_layer0_input"][:, :PREFIX_LEN, :]
    a3_single = captures["single_layer0_input"][:, :PREFIX_LEN, :]
    a3_split_sha = tensor_sha256(a3_split)
    a3_single_sha = tensor_sha256(a3_single)
    a3_max_abs = float((a3_split - a3_single).abs().max())
    a3_match = a3_split_sha == a3_single_sha

    # A4: input_layernorm 직후 hidden_states first 6 positions
    a4_split = captures["split_prefill_layer0_post_ln"][:, :PREFIX_LEN, :]
    a4_single = captures["single_layer0_post_ln"][:, :PREFIX_LEN, :]
    a4_split_sha = tensor_sha256(a4_split)
    a4_single_sha = tensor_sha256(a4_single)
    a4_max_abs = float((a4_split - a4_single).abs().max())
    a4_match = a4_split_sha == a4_single_sha

    # ----- 보너스: q_proj / k_proj / v_proj 출력 발산 측정 (C-4 재현) -----
    qp_split = captures["split_prefill_layer0_q_proj_out"][:, :PREFIX_LEN, :]
    qp_single = captures["single_layer0_q_proj_out"][:, :PREFIX_LEN, :]
    qp_match = tensor_sha256(qp_split) == tensor_sha256(qp_single)
    qp_max_abs = float((qp_split - qp_single).abs().max())

    kp_split = captures["split_prefill_layer0_k_proj_out"][:, :PREFIX_LEN, :]
    kp_single = captures["single_layer0_k_proj_out"][:, :PREFIX_LEN, :]
    kp_match = tensor_sha256(kp_split) == tensor_sha256(kp_single)
    kp_max_abs = float((kp_split - kp_single).abs().max())

    vp_split = captures["split_prefill_layer0_v_proj_out"][:, :PREFIX_LEN, :]
    vp_single = captures["single_layer0_v_proj_out"][:, :PREFIX_LEN, :]
    vp_match = tensor_sha256(vp_split) == tensor_sha256(vp_single)
    vp_max_abs = float((vp_split - vp_single).abs().max())

    # 입력 shape 기록 (q_proj 호출 시점의 GEMM 문제 크기 추적용)
    qproj_input_shape_split = list(a4_split.shape)
    qproj_input_shape_split_full = list(captures["split_prefill_layer0_post_ln"].shape)
    qproj_input_shape_single = list(a4_single.shape)
    qproj_input_shape_single_full = list(captures["single_layer0_post_ln"].shape)

    # ============================================================
    # JSON 저장 + 요약 출력
    # ============================================================
    summary: dict[str, Any] = {
        "round": "C-6",
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "prompt": PROMPT,
        "prefix_len": PREFIX_LEN,
        "decode_token_id": next_id_int,
        "decode_token_decoded": tokenizer.decode([next_id_int]),

        "condition_A_first_input_identity": {
            "A1_input_ids_first6": {
                "split": a1_split_ids,
                "single": a1_single_ids,
                "match": a1_match,
            },
            "A2_embed_first6": {
                "split_sha256": a2_split_sha,
                "single_sha256": a2_single_sha,
                "max_abs": a2_max_abs,
                "match": a2_match,
            },
            "A3_layer0_input_first6": {
                "split_sha256": a3_split_sha,
                "single_sha256": a3_single_sha,
                "max_abs": a3_max_abs,
                "match": a3_match,
            },
            "A4_post_input_layernorm_first6": {
                "split_sha256": a4_split_sha,
                "single_sha256": a4_single_sha,
                "max_abs": a4_max_abs,
                "match": a4_match,
            },
            "all_pass": a1_match and a2_match and a3_match and a4_match,
        },

        "condition_B_eager_mode": eager_mode_state,
        "condition_C_precision": precision_state,
        "condition_D_deterministic_mode": {
            "post_setup": d_state_post_setup,
            "pre_split_prefill": d_state_pre_split_prefill,
            "pre_single": d_state_pre_single,
            "gpu_determinism_quick_check": matmul_check,
        },

        "bonus_qkv_proj_divergence": {
            "note": "조건 A·B·C·D 검증 하에서 C-4의 q_proj 발산 재측정",
            "qproj_input_shape_split_first6": qproj_input_shape_split,
            "qproj_input_shape_split_full": qproj_input_shape_split_full,
            "qproj_input_shape_single_first6": qproj_input_shape_single,
            "qproj_input_shape_single_full": qproj_input_shape_single_full,
            "q_proj_out_first6_match_bitwise": qp_match,
            "q_proj_out_first6_max_abs": qp_max_abs,
            "k_proj_out_first6_match_bitwise": kp_match,
            "k_proj_out_first6_max_abs": kp_max_abs,
            "v_proj_out_first6_match_bitwise": vp_match,
            "v_proj_out_first6_max_abs": vp_max_abs,
        },
    }

    out_file = out_dir / "diagnose_c6.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  저장: {out_file}")

    # 요약 판정
    a_all = summary["condition_A_first_input_identity"]["all_pass"]
    b_all = (
        eager_mode_state["self_attn_match"]
        and eager_mode_state["q_proj_is_torch_nn_Linear"]
        and attn_impl == "eager"
        and eager_mode_state["self_attn_is_not_sdpa"]
        and eager_mode_state["self_attn_is_not_flash"]
    )
    c_all = (
        precision_state["model_dtype"] == "torch.float32"
        and precision_state["first_param_dtype"] == "torch.float32"
        and precision_state["layer0_qproj_weight_dtype"] == "torch.float32"
    )
    d_all = (
        d_state_post_setup["use_deterministic_algorithms"]
        and not d_state_post_setup["deterministic_warn_only"]
        and d_state_post_setup["cudnn_deterministic"]
        and not d_state_post_setup["cudnn_benchmark"]
        and not d_state_post_setup["cuda_matmul_allow_tf32"]
        and not d_state_post_setup["cudnn_allow_tf32"]
        and d_state_post_setup["float32_matmul_precision"] == "highest"
        and matmul_check["matmul_same_input_bitwise"]
    )

    print()
    print(f"  조건 A (첫 입력 동일):        {'✅' if a_all else '❌'}")
    print(f"    A1 input_ids first 6:       {'✅' if a1_match else '❌'}")
    print(f"    A2 embed first 6:           {'✅' if a2_match else '❌'} (max_abs={a2_max_abs:.3e})")
    print(f"    A3 layer0 input first 6:    {'✅' if a3_match else '❌'} (max_abs={a3_max_abs:.3e})")
    print(f"    A4 post-LN first 6:         {'✅' if a4_match else '❌'} (max_abs={a4_max_abs:.3e})")
    print(f"  조건 B (eager 모드):          {'✅' if b_all else '❌'}")
    print(f"    self_attn class:            {self_attn_cls}")
    print(f"    q_proj is nn.Linear:        {q_proj_is_linear}")
    print(f"  조건 C (fp32 동일 정밀도):    {'✅' if c_all else '❌'}")
    print(f"  조건 D (deterministic 모드):  {'✅' if d_all else '❌'}")
    print(f"    matmul bitwise self-check:  {matmul_check['matmul_same_input_bitwise']}")
    print()
    print(f"  보너스 q_proj first 6 max_abs:  {qp_max_abs:.3e}  (bitwise={qp_match})")
    print(f"  보너스 k_proj first 6 max_abs:  {kp_max_abs:.3e}  (bitwise={kp_match})")
    print(f"  보너스 v_proj first 6 max_abs:  {vp_max_abs:.3e}  (bitwise={vp_match})")

    if a_all and b_all and c_all and d_all:
        print()
        print("  ==> 4 조건 모두 통과. q_proj 발산 측정값을 다음 round 분석 기준으로.")
    else:
        print()
        print("  ==> 4 조건 중 일부 실패. 실패 항목이 cause 후보.")


if __name__ == "__main__":
    main()
