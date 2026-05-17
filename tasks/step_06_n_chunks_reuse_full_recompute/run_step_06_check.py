"""Step 6: N chunks reuse + recompute_ratio=1.0 = vanilla forward.

invariants:
  6.1 100% recompute path logits == vanilla logits (bitwise, model-backed)
  (local smoke: API contract — ratio=0.5 raises NotImplementedError)
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from compblend.blend import cacheblend_forward_full_recompute
from compblend.cache import ChunkMeta, ChunkedKVStore
from compblend.rope_rotation import re_rotate_chunked_store_k_inplace

SEED = 42
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"
NUM_LAYERS = 32


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


def check_api_contract():
    """ratio=0.5 → NotImplementedError 확인 (model-less)."""
    class DummyModel:
        def __call__(self, **kw):
            raise RuntimeError("model.__call__ called — should not happen for ratio != 1.0 contract check")
    dummy = DummyModel()
    ids = torch.zeros(1, 6, dtype=torch.long)
    mask = torch.ones(1, 6, dtype=torch.long)
    try:
        cacheblend_forward_full_recompute(dummy, ids, mask, blended_cache=None,
                                            recompute_ratio=0.5)
        return {"passed": False, "reason": "ratio=0.5 did not raise NotImplementedError"}
    except NotImplementedError as e:
        return {"passed": True, "raised": "NotImplementedError", "msg": str(e)[:80]}
    except Exception as e:
        return {"passed": False, "reason": f"unexpected exception: {type(e).__name__}: {e}"}


def check_6_1(model_id):
    if not torch.cuda.is_available():
        return {"passed": None, "skipped": True, "reason": "CUDA unavailable"}
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
    # blended cache 생성 (Step 4 setup, recompute_ratio=1.0 모드에서는 무시되나
    # contract 상 입력 받음)
    chunk_T = 2
    n_chunks = T // chunk_T
    chunk_caches = []
    chunk_spec = []
    for i in range(n_chunks):
        chunk_ids = ids[:, i*chunk_T:(i+1)*chunk_T]
        chunk_attn = attn[:, i*chunk_T:(i+1)*chunk_T]
        set_all_seeds(SEED)
        with torch.inference_mode():
            out_c = model(input_ids=chunk_ids, attention_mask=chunk_attn, use_cache=True)
        chunk_caches.append(out_c.past_key_values)
        chunk_spec.append(ChunkMeta(
            chunk_id=f"c{i}", token_ids=chunk_ids[0].tolist(),
            original_offset=0, new_offset=i*chunk_T, original_length=chunk_T,
            is_cacheable=True, is_permanent_hit=(i == 0)))
    # ChunkedKVStore 조립
    chunks_dict = {cm.chunk_id: cm for cm in chunk_spec}
    kv_dict = {}
    for i, cm in enumerate(chunk_spec):
        kv = []
        dc_i = chunk_caches[i]
        for layer in range(NUM_LAYERS):
            k = dc_i.key_cache[layer][0, :, :, :].detach().clone()
            v = dc_i.value_cache[layer][0, :, :, :].detach().clone()
            kv.append((k, v))
        kv_dict[cm.chunk_id] = kv
    store = ChunkedKVStore(chunks=chunks_dict, kv=kv_dict, num_layers=NUM_LAYERS)
    re_rotate_chunked_store_k_inplace(store)
    blended_cache = store.to_dynamic_cache()
    # 6.1
    set_all_seeds(SEED)
    with torch.inference_mode():
        logits_blend = cacheblend_forward_full_recompute(
            model, ids, attn, blended_cache=blended_cache, recompute_ratio=1.0
        )
    set_all_seeds(SEED)
    with torch.inference_mode():
        logits_vanilla = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
    logits_blend_cpu = logits_blend.detach().to("cpu", torch.float32)
    logits_vanilla_cpu = logits_vanilla.detach().to("cpu", torch.float32)
    sha_b = tensor_sha256(logits_blend_cpu)
    sha_v = tensor_sha256(logits_vanilla_cpu)
    max_abs = float((logits_blend_cpu - logits_vanilla_cpu).abs().max())
    return {
        "passed": sha_b == sha_v, "gate": "sha256 logits equality",
        "logits_blend_sha256": sha_b, "logits_vanilla_sha256": sha_v,
        "max_abs_diff": max_abs,
        "n_chunks": n_chunks, "chunk_T": chunk_T, "prefix_length": T,
        "recompute_ratio": 1.0,
        "env": {"torch": torch.__version__, "transformers": _tx.__version__,
                 "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--enable-model-check", action="store_true")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    setup_deterministic()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] API contract check (model-less)")
    r_api = check_api_contract()

    print(f"[2/3] 6.1 100% recompute path == vanilla — {'활성' if args.enable_model_check else '생략'}")
    if args.enable_model_check and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 필요"); sys.exit(1)
    if args.enable_model_check:
        r_6_1 = check_6_1(args.model)
    else:
        r_6_1 = {"passed": None, "skipped": True, "reason": "model check 미활성"}

    print("[3/3] summary")
    local_smoke = bool(r_api["passed"])
    final = bool(local_smoke and (r_6_1.get("passed") is True))
    summary = {
        "step": 6,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "model_check_enabled": args.enable_model_check,
        "invariants": {
            "api_contract_ratio_lt_1_raises": r_api,
            "6.1_full_recompute_eq_vanilla":  r_6_1,
        },
        "local_smoke_gate_passed": local_smoke,
        "step_06_final_gate_passed": final,
        "all_invariants_passed": final,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_dir / 'summary.json'}")
    print()
    print(f"  API contract (ratio<1 → NotImplementedError): {'✅' if r_api['passed'] else '❌'}")
    if r_6_1.get("skipped"):
        print(f"  6.1 (model-backed): ⏭️  생략")
    else:
        print(f"  6.1 (full recompute == vanilla, bitwise): {'✅' if r_6_1['passed'] else '❌'}"
              f" max_abs={r_6_1.get('max_abs_diff')}")
    if args.enable_model_check:
        if final: print("==> Step 6 final gate PASS")
        else: print("==> Step 6 final gate FAIL"); sys.exit(1)
    else:
        if local_smoke: print("==> local smoke PASS; final pending (6.1 vast.ai)")
        else: print("==> local smoke FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
