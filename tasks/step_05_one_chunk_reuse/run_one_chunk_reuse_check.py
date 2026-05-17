"""Step 5: 1 chunk reuse = vanilla forward.

invariants:
  5.1 one full chunk K/V roundtrip (bitwise, model-less)
  5.2 one chunk reuse decode equivalence (bitwise, model-backed)
  5.3 post-update cache equivalence (bitwise, model-backed) — 신규
  5.4 ChunkMeta 7-field equality (bitwise, model-less)
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from compblend.cache import ChunkMeta, ChunkedKVStore

SEED = 42
H_KV = 8
D = 128
NUM_LAYERS = 32
B = 1
T_TOTAL = 6
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"


def set_all_seeds(s=SEED):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); random.seed(s); np.random.seed(s)


def setup_deterministic():
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try: torch.set_float32_matmul_precision("highest")
    except: pass


def tensor_sha256(t):
    return hashlib.sha256(t.detach().cpu().to(torch.float32).numpy().tobytes()).hexdigest()


def _deepcopy_dc(dc):
    new = DynamicCache()
    for i in range(len(dc.key_cache)):
        new.update(dc.key_cache[i].clone(), dc.value_cache[i].clone(), i)
    return new


def build_single_chunk_spec(T):
    return [ChunkMeta(chunk_id="full_chunk", token_ids=list(range(T)),
                       original_offset=0, new_offset=0, original_length=T,
                       is_cacheable=True, is_permanent_hit=True)]


def make_dummy_dc(num_layers, b, h_kv, t, d, device="cpu"):
    set_all_seeds(SEED)
    dc = DynamicCache()
    for i in range(num_layers):
        k = torch.randn(b, h_kv, t, d, device=device, dtype=torch.float32)
        v = torch.randn(b, h_kv, t, d, device=device, dtype=torch.float32)
        dc.update(k, v, i)
    return dc


def check_5_1_roundtrip(dc0, dc1, n_layers):
    mismatched = []
    for i in range(n_layers):
        if not (torch.equal(dc0.key_cache[i], dc1.key_cache[i]) and
                torch.equal(dc0.value_cache[i], dc1.value_cache[i])):
            mismatched.append(i)
    return {"passed": len(mismatched) == 0, "gate": "torch.equal",
             "n_layers": n_layers, "mismatched_layers": mismatched,
             "seq_orig": dc0.get_seq_length(), "seq_round": dc1.get_seq_length()}


def check_5_4_meta(spec, store):
    mm = []
    for cm in spec:
        s = store.chunks.get(cm.chunk_id)
        if not (s is not None and asdict(s) == asdict(cm)):
            mm.append(cm.chunk_id)
    return {"passed": len(mm) == 0, "mismatched": mm,
             "n_chunks_input": len(spec), "n_chunks_stored": len(store.chunks)}


def check_5_2_5_3_model(model_id):
    if not torch.cuda.is_available():
        return ({"passed": None, "skipped": True, "reason": "CUDA unavailable"},
                {"passed": None, "skipped": True, "reason": "CUDA unavailable"})
    import transformers as _tx
    from transformers import AutoTokenizer
    from compblend.modeling import MistralForCausalLM
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    enc = tok(PROMPT, return_tensors="pt")
    ids = enc["input_ids"].to("cuda"); attn = enc["attention_mask"].to("cuda")
    T = ids.shape[1]
    model = MistralForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32,
                                                  attn_implementation="eager",
                                                  token=os.environ.get("HF_TOKEN")).to("cuda").eval()
    # prefill A
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_a = model(input_ids=ids, attention_mask=attn, use_cache=True)
    d_a = out_a.past_key_values
    next_id = out_a.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_int = int(next_id.item())
    # prefill B → roundtrip
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_b = model(input_ids=ids, attention_mask=attn, use_cache=True)
    d_orig = out_b.past_key_values
    spec = build_single_chunk_spec(T)
    d_round = ChunkedKVStore.from_dynamic_cache(d_orig, spec).to_dynamic_cache()
    n_layers = len(d_round.key_cache)
    # 5.2 decode logits equivalence
    decode_attn = torch.ones((1, T+1), dtype=torch.long, device="cuda")
    decode_cp = torch.arange(T, T+1, device="cuda")
    d_a_for_5_2 = _deepcopy_dc(d_a)
    d_round_for_5_2 = _deepcopy_dc(d_round)
    with torch.inference_mode():
        dec_a = model(input_ids=next_id, past_key_values=d_a_for_5_2,
                       attention_mask=decode_attn, cache_position=decode_cp, use_cache=True)
        dec_b = model(input_ids=next_id, past_key_values=d_round_for_5_2,
                       attention_mask=decode_attn, cache_position=decode_cp, use_cache=True)
    logits_a = dec_a.logits[:, -1, :].detach().to("cpu", torch.float32)
    logits_b = dec_b.logits[:, -1, :].detach().to("cpu", torch.float32)
    sha_a, sha_b = tensor_sha256(logits_a), tensor_sha256(logits_b)
    max_abs = float((logits_a - logits_b).abs().max())
    r_5_2 = {
        "passed": sha_a == sha_b, "gate": "sha256 logits equality",
        "logits_a_sha256": sha_a, "logits_b_sha256": sha_b,
        "max_abs_diff": max_abs, "decode_token_id": next_int,
        "decode_token_decoded": tok.decode([next_int]),
        "prefix_length": T, "n_chunks": 1,
        "env": {"torch": torch.__version__, "transformers": _tx.__version__,
                 "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    # 5.3 post-update cache equivalence — d_a_for_5_2와 d_round_for_5_2 mutate된 K/V 비교
    mm_53 = []
    per_layer_53 = []
    for i in range(n_layers):
        ka, kb = d_a_for_5_2.key_cache[i], d_round_for_5_2.key_cache[i]
        va, vb = d_a_for_5_2.value_cache[i], d_round_for_5_2.value_cache[i]
        sliced_a_k = ka[:, :, :T+1, :]; sliced_b_k = kb[:, :, :T+1, :]
        sliced_a_v = va[:, :, :T+1, :]; sliced_b_v = vb[:, :, :T+1, :]
        k_match = bool(torch.equal(sliced_a_k, sliced_b_k))
        v_match = bool(torch.equal(sliced_a_v, sliced_b_v))
        per_layer_53.append({"layer": i, "k_match": k_match, "v_match": v_match,
                              "k_max_abs": float((sliced_a_k - sliced_b_k).abs().max()),
                              "v_max_abs": float((sliced_a_v - sliced_b_v).abs().max())})
        if not (k_match and v_match):
            mm_53.append(i)
    r_5_3 = {
        "passed": len(mm_53) == 0, "gate": "torch.equal K/V[:T+1] per layer",
        "n_layers": n_layers, "T_after_decode": T+1,
        "mismatched_layers": mm_53, "per_layer": per_layer_53[:4],  # 첫 4 layer만 (용량)
    }
    return r_5_2, r_5_3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-model-check", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    setup_deterministic()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] model-less 5.1 + 5.4")
    dc_orig = make_dummy_dc(NUM_LAYERS, B, H_KV, T_TOTAL, D)
    spec = build_single_chunk_spec(T_TOTAL)
    store = ChunkedKVStore.from_dynamic_cache(dc_orig, spec)
    dc_round = store.to_dynamic_cache()
    r_5_1 = check_5_1_roundtrip(dc_orig, dc_round, NUM_LAYERS)
    r_5_4 = check_5_4_meta(spec, store)

    print(f"[2/5] model-backed 5.2 + 5.3 — {'활성' if args.enable_model_check else '생략'}")
    if args.enable_model_check and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 필요"); sys.exit(1)
    if args.enable_model_check:
        r_5_2, r_5_3 = check_5_2_5_3_model(args.model)
    else:
        r_5_2 = {"passed": None, "skipped": True, "reason": "model check 미활성"}
        r_5_3 = {"passed": None, "skipped": True, "reason": "model check 미활성"}

    local_smoke = bool(r_5_1["passed"] and r_5_4["passed"])
    final = bool(local_smoke and (r_5_2.get("passed") is True) and (r_5_3.get("passed") is True))
    summary = {
        "step": 5,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "model_check_enabled": args.enable_model_check,
        "shapes": {"B": B, "H_kv": H_KV, "D": D, "num_layers": NUM_LAYERS, "T_total": T_TOTAL},
        "invariants": {
            "5.1_one_chunk_roundtrip": r_5_1,
            "5.2_decode_logits_equiv": r_5_2,
            "5.3_post_update_cache_equiv": r_5_3,
            "5.4_chunk_meta_equality": r_5_4,
        },
        "local_smoke_gate_passed": local_smoke,
        "step_05_final_gate_passed": final,
        "all_invariants_passed": final,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_dir / 'summary.json'}")
    print()
    print(f"  5.1 roundtrip:         {'✅' if r_5_1['passed'] else '❌'}")
    print(f"  5.2 decode logits:     {'✅' if r_5_2.get('passed') else ('⏭️' if r_5_2.get('skipped') else '❌')}")
    print(f"  5.3 post-update cache: {'✅' if r_5_3.get('passed') else ('⏭️' if r_5_3.get('skipped') else '❌')}")
    print(f"  5.4 ChunkMeta equal:   {'✅' if r_5_4['passed'] else '❌'}")
    if args.enable_model_check:
        if final: print("==> Step 5 final gate PASS")
        else: print("==> Step 5 final gate FAIL"); sys.exit(1)
    else:
        if local_smoke: print("==> local smoke PASS; final pending (5.2·5.3 vast.ai)")
        else: print("==> local smoke FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
