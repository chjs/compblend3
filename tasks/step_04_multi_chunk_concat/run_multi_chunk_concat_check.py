"""Step 4: N chunks 따로 prefill → concat (with RoPE re-rotation) = vanilla full prefill.

invariants:
  4.1 RoPE re-rotation self-consistency (model-less, bitwise)
  4.2 ChunkedKVStore reordering storage (model-less, bitwise)
  4.3 Multi-chunk vanilla equivalence (model-backed vast.ai, drift measurement)

Gate:
  local_smoke_gate_passed = 4.1 AND 4.2
  step_04_final_gate_passed = 4.1 AND 4.2 (4.3는 measurement, gate ❌)
  all_invariants_passed = step_04_final_gate_passed
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from compblend.cache import ChunkMeta, ChunkedKVStore
from compblend.rope_rotation import (
    _compute_rope_freqs, _rotate_half,
    re_rotate_k, re_rotate_chunked_store_k_inplace,
)

SEED = 42
H_KV = 8
D = 128
NUM_LAYERS = 32
B = 1
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"
DRIFT_BUDGET = 1e-2  # Step 4 loose threshold (100% reuse w/o recompute 가정)


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


def apply_rope_to_k(k_raw: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """K_raw에 R(positions) 적용. K shape (H_kv, T, D)."""
    cos, sin = _compute_rope_freqs(k_raw.shape[-1], positions,
                                     device=k_raw.device, dtype=k_raw.dtype)
    # cos/sin (T, D) → broadcast over H_kv
    cos_b = cos.unsqueeze(0)
    sin_b = sin.unsqueeze(0)
    return k_raw * cos_b + _rotate_half(k_raw) * sin_b


def check_4_1_rope_self_consistency() -> dict:
    """4.1: K_old = R(old)·K_raw → re_rotate(K_old, old, new) ≈ R(new)·K_raw.

    Gate: atol 1e-6 (fp32 RoPE composition noise floor).
    근거: fp32에서 R(a+b) direct 계산 vs R(a)·R(b) composition은 cos/sin
    trig identity의 fp32 누적 오차로 bitwise 보장 안 됨. 실측 max_abs ~5e-7
    수준 (RoPE 단계 noise floor).
    """
    set_all_seeds(SEED)
    T = 4
    atol = 1e-6
    shift_cases = [(0, 0), (0, 2), (0, 5), (3, 7), (2, 6)]
    per_case = []
    all_pass = True
    for old_start, new_start in shift_cases:
        k_raw = torch.randn(H_KV, T, D, dtype=torch.float32)
        old_pos = torch.arange(old_start, old_start + T)
        new_pos = torch.arange(new_start, new_start + T)
        k_old = apply_rope_to_k(k_raw, old_pos)
        k_new_actual = re_rotate_k(k_old, old_pos, new_pos)
        k_new_target = apply_rope_to_k(k_raw, new_pos)
        bitwise = bool(torch.equal(k_new_actual, k_new_target))
        max_abs = float((k_new_actual - k_new_target).abs().max())
        within_atol = max_abs <= atol
        per_case.append({
            "old_start": old_start, "new_start": new_start, "T": T,
            "bitwise": bitwise, "within_atol_1e-6": within_atol,
            "max_abs": max_abs,
        })
        if not within_atol:
            all_pass = False
    return {"passed": all_pass, "gate": f"atol {atol} (fp32 RoPE composition noise floor)",
             "atol_threshold": atol, "per_case": per_case}


def check_4_2_reorder_storage() -> dict:
    """4.2: chunk_spec의 new_offset != original_offset 인 경우 to_dynamic_cache 가
    new_offset 순으로 concat 함을 검증.

    setup: 3 chunks, original_offset=[0,2,4]. new_offset reorder=[4,0,2]
    예상: to_dynamic_cache 결과 = concat(chunk@new=0 → chunk@new=2 → chunk@new=4)
                              = concat(original[1], original[2], original[0])
    """
    set_all_seeds(SEED)
    # 3 chunks, each T=2
    dc_src = DynamicCache()
    chunk_T = 2
    n_chunks = 3
    T_total = chunk_T * n_chunks
    # 각 chunk 별 K/V를 미리 구분 가능한 값으로
    for layer in range(NUM_LAYERS):
        k = torch.randn(B, H_KV, T_total, D, dtype=torch.float32)
        v = torch.randn(B, H_KV, T_total, D, dtype=torch.float32)
        dc_src.update(k, v, layer)

    # chunk_spec: original_offset = [0, 2, 4], new_offset reorder = [4, 0, 2]
    new_offsets = [4, 0, 2]
    chunk_spec = []
    for i in range(n_chunks):
        chunk_spec.append(ChunkMeta(
            chunk_id=f"c{i}",
            token_ids=list(range(100*i, 100*i + chunk_T)),
            original_offset=i * chunk_T,
            new_offset=new_offsets[i],
            original_length=chunk_T,
            is_cacheable=True,
            is_permanent_hit=(i == 0),
        ))
    store = ChunkedKVStore.from_dynamic_cache(dc_src, chunk_spec)
    dc_round = store.to_dynamic_cache()

    # 기대: new_offset 순 = [0, 2, 4] → chunks_sorted = [c1, c2, c0] (original index)
    ordered_by_new = sorted(chunk_spec, key=lambda cm: (cm.new_offset, cm.chunk_id))
    expected_k_per_layer = []
    expected_v_per_layer = []
    for layer in range(NUM_LAYERS):
        ks, vs = [], []
        for cm in ordered_by_new:
            start = cm.original_offset
            end = start + cm.original_length
            ks.append(dc_src.key_cache[layer][0, :, start:end, :])
            vs.append(dc_src.value_cache[layer][0, :, start:end, :])
        expected_k_per_layer.append(torch.cat(ks, dim=-2).unsqueeze(0))
        expected_v_per_layer.append(torch.cat(vs, dim=-2).unsqueeze(0))

    per_layer = []
    mismatched = []
    for i in range(NUM_LAYERS):
        k_match = bool(torch.equal(dc_round.key_cache[i], expected_k_per_layer[i]))
        v_match = bool(torch.equal(dc_round.value_cache[i], expected_v_per_layer[i]))
        per_layer.append({"layer": i, "k_match": k_match, "v_match": v_match})
        if not (k_match and v_match):
            mismatched.append(i)
    return {
        "passed": len(mismatched) == 0,
        "gate": "torch.equal vs expected new_offset-ordered concat",
        "n_chunks": n_chunks, "chunk_T": chunk_T, "new_offsets_input": new_offsets,
        "expected_order": [cm.chunk_id for cm in ordered_by_new],
        "n_layers_compared": NUM_LAYERS,
        "mismatched_layers": mismatched,
    }


def _deepcopy_dynamic_cache(dc: DynamicCache) -> DynamicCache:
    """DynamicCache deepcopy (cache는 list of tensors)."""
    new = DynamicCache()
    for i in range(len(dc.key_cache)):
        new.update(dc.key_cache[i].clone(), dc.value_cache[i].clone(), i)
    return new


def check_4_3_vanilla_equivalence(model_id: str) -> dict:
    """4.3: multi-chunk independent prefill + RoPE re-rotation + concat vs vanilla.

    절차:
      1. vanilla prefill: model(prompt(6), use_cache=True) → D_vanilla
      2. chunk i 독립 prefill (input_ids=chunk_i tokens, length 2, position 0~1)
         → K/V_i (RoPE at positions 0..1)
      3. ChunkedKVStore에 chunk i 저장 (original_offset=0, new_offset=2*i)
      4. re_rotate_chunked_store_k_inplace → 각 chunk K가 새 position으로 회전
      5. to_dynamic_cache() → D_blended
      6. decode_vanilla = model(next_token, past_key_values=copy(D_vanilla), ...)
      7. decode_blended = model(next_token, past_key_values=D_blended, ...)
      8. drift = (decode_blended - decode_vanilla).abs()
    """
    if not torch.cuda.is_available():
        return {"passed": None, "skipped": True,
                "reason": "CUDA unavailable (model check requires GPU)"}

    import transformers as _tx
    from transformers import AutoTokenizer
    from compblend.modeling import MistralForCausalLM

    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    enc = tokenizer(PROMPT, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")               # (1, T_prompt=6)
    attn = enc["attention_mask"].to("cuda")
    T = ids.shape[1]
    chunk_T = 2
    n_chunks = T // chunk_T
    assert T % chunk_T == 0, f"T={T} not divisible by chunk_T={chunk_T}"

    model = MistralForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        attn_implementation="eager", token=token,
    ).to("cuda").eval()

    # (1) vanilla prefill
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_v = model(input_ids=ids, attention_mask=attn, use_cache=True)
    d_vanilla = out_v.past_key_values
    next_token_id = out_v.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_id_int = int(next_token_id.item())

    # (2) 각 chunk 독립 prefill at position 0
    chunk_caches: list[DynamicCache] = []
    chunk_spec: list[ChunkMeta] = []
    for i in range(n_chunks):
        chunk_ids = ids[:, i*chunk_T:(i+1)*chunk_T]
        chunk_attn = attn[:, i*chunk_T:(i+1)*chunk_T]
        set_all_seeds(SEED)
        with torch.inference_mode():
            out_c = model(input_ids=chunk_ids, attention_mask=chunk_attn, use_cache=True)
        chunk_caches.append(out_c.past_key_values)
        chunk_spec.append(ChunkMeta(
            chunk_id=f"c{i}",
            token_ids=chunk_ids[0].tolist(),
            original_offset=0,           # 독립 prefill 시 position 0
            new_offset=i * chunk_T,      # blend 시 위치
            original_length=chunk_T,
            is_cacheable=True,
            is_permanent_hit=(i == 0),
        ))

    # (3) ChunkedKVStore에 각 chunk 저장
    #    ChunkedKVStore.from_dynamic_cache는 단일 DynamicCache 받음 →
    #    각 chunk별 별도 store 만들고 합치는 식으로
    #    간단히 내부 dict 직접 구성
    chunks_dict: dict[str, ChunkMeta] = {}
    kv_dict: dict[str, list] = {}
    for i, cm in enumerate(chunk_spec):
        chunks_dict[cm.chunk_id] = cm
        chunk_kv = []
        dc_i = chunk_caches[i]
        for layer in range(NUM_LAYERS):
            k = dc_i.key_cache[layer][0, :, :, :].detach().clone()    # (H_kv, T_chunk, D)
            v = dc_i.value_cache[layer][0, :, :, :].detach().clone()
            chunk_kv.append((k, v))
        kv_dict[cm.chunk_id] = chunk_kv
    store = ChunkedKVStore(chunks=chunks_dict, kv=kv_dict, num_layers=NUM_LAYERS)

    # (4) re-rotate
    re_rotate_chunked_store_k_inplace(store)

    # (5) materialize to DynamicCache
    d_blended = store.to_dynamic_cache()

    # (보너스) per-layer K bitwise — k_proj out_dim=1024이라 same-shape K_raw bitwise 기대
    # chunk[i] re-rotated K layer 0 vs vanilla K[pos 2i..2i+1] layer 0
    per_chunk_layer0 = []
    for i, cm in enumerate(chunk_spec):
        # store의 chunk i layer 0 K (re-rotated)
        k_chunk = store.kv[cm.chunk_id][0][0]                          # (H_kv, T_chunk, D)
        k_vanilla = d_vanilla.key_cache[0][0, :, cm.new_offset:cm.new_offset+chunk_T, :]
        bw = bool(torch.equal(k_chunk, k_vanilla))
        ma = float((k_chunk - k_vanilla).abs().max())
        per_chunk_layer0.append({
            "chunk_id": cm.chunk_id, "new_offset": cm.new_offset,
            "layer0_k_bitwise": bw, "layer0_k_max_abs": ma,
        })

    # (6)/(7) decode with copies of caches (forward mutates cache)
    decode_attn = torch.ones((1, T + 1), dtype=torch.long, device="cuda")
    decode_cp = torch.arange(T, T + 1, device="cuda")

    d_vanilla_copy = _deepcopy_dynamic_cache(d_vanilla)
    with torch.inference_mode():
        dec_v = model(input_ids=next_token_id, past_key_values=d_vanilla_copy,
                       attention_mask=decode_attn, cache_position=decode_cp,
                       use_cache=True)
        dec_b = model(input_ids=next_token_id, past_key_values=d_blended,
                       attention_mask=decode_attn, cache_position=decode_cp,
                       use_cache=True)
    logits_v = dec_v.logits.detach().to("cpu", torch.float32)
    logits_b = dec_b.logits.detach().to("cpu", torch.float32)
    logits_v_last = logits_v[:, -1, :]
    logits_b_last = logits_b[:, -1, :]

    drift = (logits_v - logits_b).abs()
    max_abs = float(drift.max())
    mean_abs = float(drift.mean())
    v_argmax = int(logits_v_last.argmax(dim=-1).item())
    b_argmax = int(logits_b_last.argmax(dim=-1).item())
    argmax_match = v_argmax == b_argmax
    v_top5 = set(logits_v_last.topk(5).indices[0].tolist())
    b_top5 = set(logits_b_last.topk(5).indices[0].tolist())
    top5_overlap = len(v_top5 & b_top5)
    drift_budget_exceeded = max_abs > DRIFT_BUDGET

    return {
        "passed": None,  # measurement only, gate ❌
        "skipped": False,
        "gate": "measurement only (drift_budget_exceeded = max_abs > 1e-2)",
        "n_chunks": n_chunks, "chunk_T": chunk_T, "prefix_length": T,
        "decode_token_id_vanilla": next_id_int,
        "decode_token_decoded_vanilla": tokenizer.decode([next_id_int]),
        "vanilla_argmax": v_argmax, "blended_argmax": b_argmax,
        "argmax_match": argmax_match,
        "top5_overlap_k5": top5_overlap,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "drift_budget_threshold": DRIFT_BUDGET,
        "drift_budget_exceeded": drift_budget_exceeded,
        "per_chunk_layer0_k_check": per_chunk_layer0,
        "logits_vanilla_sha256_last": tensor_sha256(logits_v_last),
        "logits_blended_sha256_last": tensor_sha256(logits_b_last),
        "env": {
            "torch_version": torch.__version__,
            "transformers_version": _tx.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "model_dtype": str(model.dtype),
            "attention_implementation": getattr(model.config, "_attn_implementation", None),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-model-check", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    setup_deterministic()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] 4.1 RoPE re-rotation self-consistency (model-less)")
    r_4_1 = check_4_1_rope_self_consistency()

    print("[2/4] 4.2 ChunkedKVStore reordering storage (model-less)")
    r_4_2 = check_4_2_reorder_storage()

    print(f"[3/4] 4.3 model-backed — {'활성' if args.enable_model_check else '생략'}")
    if args.enable_model_check and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 필요"); sys.exit(1)
    if args.enable_model_check:
        r_4_3 = check_4_3_vanilla_equivalence(args.model)
    else:
        r_4_3 = {"passed": None, "skipped": True,
                  "reason": "model check 미활성 (--enable-model-check 필요)"}

    print("[4/4] summary 작성")
    local_smoke_gate_passed = bool(r_4_1["passed"] and r_4_2["passed"])
    step_04_final_gate_passed = local_smoke_gate_passed
    summary = {
        "step": 4,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "model_check_enabled": args.enable_model_check,
        "shapes": {"B": B, "H_kv": H_KV, "D": D, "num_layers": NUM_LAYERS},
        "invariants": {
            "4.1_rope_re_rotation_self_consistency": r_4_1,
            "4.2_chunked_store_reorder_concat":      r_4_2,
            "4.3_multi_chunk_vanilla_drift":         r_4_3,
        },
        "local_smoke_gate_passed": local_smoke_gate_passed,
        "step_04_final_gate_passed": step_04_final_gate_passed,
        "all_invariants_passed": step_04_final_gate_passed,
    }
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_file}")

    print()
    print(f"  4.1 RoPE re-rotation:        {'✅' if r_4_1['passed'] else '❌'}")
    print(f"  4.2 reorder concat:          {'✅' if r_4_2['passed'] else '❌'}"
          f" (mismatched={r_4_2['mismatched_layers']})")
    if r_4_3.get("skipped"):
        print(f"  4.3 vanilla equivalence:     ⏭️  생략 ({r_4_3.get('reason')})")
    else:
        print(f"  4.3 drift max_abs={r_4_3['max_abs_diff']:.3e} mean={r_4_3['mean_abs_diff']:.3e}"
              f" argmax_match={r_4_3['argmax_match']} top5={r_4_3['top5_overlap_k5']}/5"
              f" budget_exceeded={r_4_3['drift_budget_exceeded']}")

    if args.enable_model_check:
        if step_04_final_gate_passed:
            print("==> Step 4 final gate PASS (4.3는 measurement)")
        else:
            print("==> Step 4 final gate FAIL"); sys.exit(1)
    else:
        if local_smoke_gate_passed:
            print("==> local smoke gate PASS; final gate pending (4.3 vast.ai)")
        else:
            print("==> local smoke gate FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
