"""Step 3 검증: ChunkedKVStore 자료구조 정확성 + Cache 인터페이스 호환성.

invariants:
  3.1  DynamicCache → ChunkedKVStore → DynamicCache' roundtrip K/V bitwise (model-less)
  3.2  ChunkMeta 7-field equality (model-less)
  3.3A 변환 출력 인터페이스 적합성 (model-less): isinstance(Cache/DynamicCache),
       get_seq_length, get_max_cache_shape, update 동작
  3.3B 모델 forward logits SHA-256 일치 (model-backed, vast.ai 전용,
       --enable-model-check flag)

원칙: src/compblend/modeling/ fork 무수정 (Step 1/2). ChunkedKVStore는
src/compblend/cache.py 신규 모듈 (해석 A — Cache 상속 ❌).
"""
from __future__ import annotations

import argparse
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
from transformers.cache_utils import Cache, DynamicCache

from compblend.cache import ChunkMeta, ChunkedKVStore

SEED = 42

# Mistral 7B v0.2 shape constants (Step 0/1/2와 일치, B=1)
H_KV = 8         # num_key_value_heads (GQA)
D = 128          # head_dim
NUM_LAYERS = 32  # Mistral 7B
B = 1
T_TOTAL = 6      # smoke prompt length

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"


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


def make_dummy_dynamic_cache(num_layers: int, b: int, h_kv: int, t: int, d: int,
                              device: str = "cpu") -> DynamicCache:
    """랜덤 K/V로 채운 DynamicCache (model-less 테스트용)."""
    set_all_seeds(SEED)
    dc = DynamicCache()
    for i in range(num_layers):
        k = torch.randn(b, h_kv, t, d, device=device, dtype=torch.float32)
        v = torch.randn(b, h_kv, t, d, device=device, dtype=torch.float32)
        dc.update(k, v, i)
    return dc


def build_default_chunk_spec(t_total: int = T_TOTAL) -> list[ChunkMeta]:
    """Step 3 smoke chunk_spec — t_total 토큰을 2-token chunks로 분할.

    new_offset == original_offset (Step 3 scope, 재배열 ❌).
    is_permanent_hit: 첫 chunk만 True (system prompt 가정).
    """
    spec: list[ChunkMeta] = []
    for i in range(0, t_total, 2):
        chunk_len = min(2, t_total - i)
        spec.append(ChunkMeta(
            chunk_id=f"chunk_{i//2}",
            token_ids=list(range(100 * (i//2 + 1), 100 * (i//2 + 1) + chunk_len)),
            original_offset=i,
            new_offset=i,
            original_length=chunk_len,
            is_cacheable=True,
            is_permanent_hit=(i == 0),
        ))
    return spec


def check_3_1_roundtrip(dc_orig: DynamicCache, dc_round: DynamicCache,
                          n_layers: int) -> dict:
    """3.1: per-layer K/V `torch.equal` + seq_length 일치."""
    per_layer: list[dict[str, Any]] = []
    mismatched_layers: list[int] = []
    for i in range(n_layers):
        k_match = bool(torch.equal(dc_orig.key_cache[i], dc_round.key_cache[i]))
        v_match = bool(torch.equal(dc_orig.value_cache[i], dc_round.value_cache[i]))
        per_layer.append({
            "layer": i,
            "k_match": k_match,
            "v_match": v_match,
            "k_sha_orig":  tensor_sha256(dc_orig.key_cache[i]),
            "k_sha_round": tensor_sha256(dc_round.key_cache[i]),
            "v_sha_orig":  tensor_sha256(dc_orig.value_cache[i]),
            "v_sha_round": tensor_sha256(dc_round.value_cache[i]),
        })
        if not (k_match and v_match):
            mismatched_layers.append(i)
    seq_orig = dc_orig.get_seq_length()
    seq_round = dc_round.get_seq_length()
    seq_match = seq_orig == seq_round
    passed = (len(mismatched_layers) == 0) and seq_match
    return {
        "passed": passed,
        "gate": "torch.equal",
        "n_layers": n_layers,
        "mismatched_layers": mismatched_layers,
        "seq_len_orig": seq_orig,
        "seq_len_round": seq_round,
        "seq_len_match": seq_match,
        "per_layer": per_layer,
    }


def check_3_2_meta(chunk_spec: list[ChunkMeta], store: ChunkedKVStore) -> dict:
    """3.2: ChunkMeta 7 필드 dataclass equality."""
    per_chunk: list[dict[str, Any]] = []
    mismatched: list[str] = []
    for cm in chunk_spec:
        stored = store.chunks.get(cm.chunk_id)
        match = stored is not None and asdict(stored) == asdict(cm)
        per_chunk.append({
            "chunk_id": cm.chunk_id,
            "match": match,
            "input":  asdict(cm),
            "stored": asdict(stored) if stored else None,
        })
        if not match:
            mismatched.append(cm.chunk_id)
    return {
        "passed": len(mismatched) == 0,
        "n_chunks_input": len(chunk_spec),
        "n_chunks_stored": len(store.chunks),
        "mismatched_chunk_ids": mismatched,
        "per_chunk": per_chunk,
    }


def check_3_3A_interface(dc_orig: DynamicCache, dc_round: DynamicCache,
                           h_kv: int, d: int) -> dict:
    """3.3A: HF Cache 인터페이스 surface 검증.

    검증 항목:
      - isinstance(dc_round, DynamicCache) / Cache
      - get_seq_length(), get_max_cache_shape() 가 dc_orig와 일치
      - update() 호출 시 shape 확장 정상

    주의: 본 함수는 dc_round.key_cache[0]에 dummy K/V를 update하므로
    호출 후 dc_round의 layer 0 길이는 +1 됨. 3.1/3.2 검증 이후에 호출할 것.
    """
    checks: dict[str, Any] = {}
    checks["isinstance_DynamicCache"] = isinstance(dc_round, DynamicCache)
    checks["isinstance_Cache"] = isinstance(dc_round, Cache)
    checks["seq_length_orig"]  = dc_orig.get_seq_length()
    checks["seq_length_round"] = dc_round.get_seq_length()
    checks["seq_length_match"] = checks["seq_length_orig"] == checks["seq_length_round"]
    checks["max_cache_shape_orig"]  = dc_orig.get_max_cache_shape()
    checks["max_cache_shape_round"] = dc_round.get_max_cache_shape()
    checks["max_cache_shape_match"] = (
        checks["max_cache_shape_orig"] == checks["max_cache_shape_round"]
    )

    # update() 동작 — layer 0에 +1 K/V append 시 길이 +1 확인
    seq_before = dc_round.get_seq_length()
    new_k = torch.zeros(1, h_kv, 1, d,
                          device=dc_round.key_cache[0].device,
                          dtype=dc_round.key_cache[0].dtype)
    new_v = torch.zeros(1, h_kv, 1, d,
                          device=dc_round.value_cache[0].device,
                          dtype=dc_round.value_cache[0].dtype)
    returned_k, returned_v = dc_round.update(new_k, new_v, layer_idx=0)
    seq_after = dc_round.get_seq_length()
    checks["update_returned_k_shape"] = list(returned_k.shape)
    checks["update_returned_v_shape"] = list(returned_v.shape)
    checks["update_seq_before"] = seq_before
    checks["update_seq_after"]  = seq_after
    checks["update_seq_grew_by_1"] = seq_after == seq_before + 1

    passed = (
        checks["isinstance_DynamicCache"]
        and checks["isinstance_Cache"]
        and checks["seq_length_match"]
        and checks["max_cache_shape_match"]
        and checks["update_seq_grew_by_1"]
    )
    return {"passed": passed, "checks": checks}


def _compare_dc_kv(d1: DynamicCache, d2: DynamicCache, n_layers: int) -> dict:
    """두 DynamicCache의 K/V layer별 bitwise + seq_len 비교 (read-only)."""
    mismatched: list[int] = []
    for i in range(n_layers):
        k_eq = bool(torch.equal(d1.key_cache[i], d2.key_cache[i]))
        v_eq = bool(torch.equal(d1.value_cache[i], d2.value_cache[i]))
        if not (k_eq and v_eq):
            mismatched.append(i)
    return {
        "bitwise": len(mismatched) == 0,
        "mismatched_layers": mismatched,
        "seq_len_a": d1.get_seq_length(),
        "seq_len_b": d2.get_seq_length(),
        "seq_len_match": d1.get_seq_length() == d2.get_seq_length(),
    }


def _classify_3_3b_failure(diag: dict) -> str:
    """3.3B failure case 자동 판정 (PASS면 빈 문자열)."""
    if diag.get("logits_sha_match"):
        return ""
    pao = diag.get("prefill_a_vs_orig", {})
    orw = diag.get("orig_vs_round", {})
    if not pao.get("bitwise", True):
        return "Case 1: prefill A·orig 결정론 실패 (ChunkedKVStore 무관)"
    if not orw.get("bitwise", True):
        return "Case 2: ChunkedKVStore 저장/복원 손실"
    return "Case 3: K/V는 bitwise이나 decode logits 불일치 (forward 인자/인터페이스)"


def check_3_3B_model_forward(model_id: str) -> dict:
    """3.3B: 모델 forward 시 logits SHA-256 일치 (vast.ai GPU 전용).

    절차:
      1) prefill A: model(prompt, use_cache=True) → D_A (path A의 decode 대상)
      2) prefill B: model(prompt, use_cache=True) → D_orig (별도 forward, 동일 결과 기대)
      3) D_round = ChunkedKVStore.from_dynamic_cache(D_orig, spec).to_dynamic_cache()
      4) decode A: model(next_token, past_key_values=D_A) → logits_A
      5) decode B: model(next_token, past_key_values=D_round) → logits_B
      6) sha256(logits_A) == sha256(logits_B)

    Diagnostic fields (PASS·FAIL 모두 채움):
      - prefill_a_vs_orig: D_A vs D_orig bitwise (결정론 sanity)
      - orig_vs_round:    D_orig vs D_round bitwise (ChunkedKVStore 저장/복원 sanity)
      - failure_case:     case 1~3 자동 판정 (PASS면 "")
      - env: torch / transformers / cuda / device / dtype / attn impl
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
    ids = enc["input_ids"].to("cuda")             # (1, T_prompt)
    attn = enc["attention_mask"].to("cuda")
    T = ids.shape[1]

    model = MistralForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        attn_implementation="eager", token=token,
    ).to("cuda").eval()

    # (1) prefill A — 그대로 decode A에 사용
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_a = model(input_ids=ids, attention_mask=attn, use_cache=True)
    d_a = out_a.past_key_values
    next_token_id = out_a.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    next_id_int = int(next_token_id.item())

    # (2) prefill B — ChunkedKVStore source. 결정론적이라 d_a와 bitwise 일치 기대
    set_all_seeds(SEED)
    with torch.inference_mode():
        out_b = model(input_ids=ids, attention_mask=attn, use_cache=True)
    d_orig = out_b.past_key_values

    # (3) Roundtrip — T 토큰을 2씩 chunk
    chunk_spec: list[ChunkMeta] = []
    for i in range(0, T, 2):
        chunk_len = min(2, T - i)
        chunk_spec.append(ChunkMeta(
            chunk_id=f"c{i//2}",
            token_ids=ids[0, i:i+chunk_len].tolist(),
            original_offset=i,
            new_offset=i,
            original_length=chunk_len,
            is_cacheable=True,
            is_permanent_hit=(i == 0),
        ))
    n_layers = len(d_orig.key_cache)
    d_round = ChunkedKVStore.from_dynamic_cache(d_orig, chunk_spec).to_dynamic_cache()

    # --- decode 전 sanity (read-only K/V 비교) ---
    pao = _compare_dc_kv(d_a, d_orig, n_layers)        # decode로 d_a/d_orig mutate 전
    orw = _compare_dc_kv(d_orig, d_round, n_layers)
    seq_a_pre, seq_o_pre, seq_r_pre = (
        d_a.get_seq_length(), d_orig.get_seq_length(), d_round.get_seq_length()
    )

    # (4)/(5) decode A·B (option E 패턴 — attention_mask + cache_position 명시)
    decode_attn = torch.ones((1, T + 1), dtype=torch.long, device="cuda")
    decode_cp = torch.arange(T, T + 1, device="cuda")
    with torch.inference_mode():
        dec_a = model(input_ids=next_token_id, past_key_values=d_a,
                       attention_mask=decode_attn, cache_position=decode_cp,
                       use_cache=True)
        dec_b = model(input_ids=next_token_id, past_key_values=d_round,
                       attention_mask=decode_attn, cache_position=decode_cp,
                       use_cache=True)
    logits_a_full = dec_a.logits.detach().to("cpu", torch.float32)
    logits_b_full = dec_b.logits.detach().to("cpu", torch.float32)
    logits_a = logits_a_full[:, -1, :]
    logits_b = logits_b_full[:, -1, :]

    sha_a = tensor_sha256(logits_a)
    sha_b = tensor_sha256(logits_b)
    max_abs = float((logits_a_full - logits_b_full).abs().max())
    mean_abs = float((logits_a_full - logits_b_full).abs().mean())

    diag: dict[str, Any] = {
        "passed": sha_a == sha_b,
        "skipped": False,
        "gate": "sha256(logits_a) == sha256(logits_b)",
        "logits_sha_match": sha_a == sha_b,
        "logits_a_sha256": sha_a,
        "logits_b_sha256": sha_b,
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "decode_token_id": next_id_int,
        "decode_token_decoded": tokenizer.decode([next_id_int]),
        "n_chunks": len(chunk_spec),
        "prefix_length": T,
        # decode 전 sanity (read-only K/V 비교; failure case 1·2 자동 판정용)
        "prefill_a_vs_orig": pao,
        "orig_vs_round": orw,
        "seq_len_a_pre_decode": seq_a_pre,
        "seq_len_orig_pre_decode": seq_o_pre,
        "seq_len_round_pre_decode": seq_r_pre,
        "seq_len_match_all": (seq_a_pre == seq_o_pre == seq_r_pre),
        # decode 호출 인자
        "attention_mask_shape_decode": list(decode_attn.shape),
        "cache_position_decode": decode_cp.cpu().tolist(),
        # env
        "torch_version": torch.__version__,
        "transformers_version": _tx.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": torch.cuda.get_device_name(0),
        "model_dtype": str(model.dtype),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
    }
    diag["failure_case"] = _classify_3_3b_failure(diag)
    return diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 출력 디렉토리 (예: results/step_03/vastai)")
    ap.add_argument("--enable-model-check", action="store_true",
                    help="3.3B 실행 (vast.ai GPU 환경 전용)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    setup_deterministic()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] dummy DynamicCache 생성 (model-less, CPU)")
    dc_orig = make_dummy_dynamic_cache(NUM_LAYERS, B, H_KV, T_TOTAL, D)

    print(f"[2/5] chunk_spec 빌드 — {T_TOTAL} 토큰을 2-token chunks로")
    chunk_spec = build_default_chunk_spec(T_TOTAL)

    print(f"[3/5] ChunkedKVStore roundtrip + 3.1/3.2/3.3A 검증")
    store = ChunkedKVStore.from_dynamic_cache(dc_orig, chunk_spec)
    dc_round = store.to_dynamic_cache()
    r_3_1  = check_3_1_roundtrip(dc_orig, dc_round, NUM_LAYERS)
    r_3_2  = check_3_2_meta(chunk_spec, store)
    r_3_3a = check_3_3A_interface(dc_orig, dc_round, H_KV, D)

    print(f"[4/5] 3.3B (model forward) — {'활성' if args.enable_model_check else '생략 (--enable-model-check 미설정)'}")
    if args.enable_model_check and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] 3.3B 실행 시 CUBLAS_WORKSPACE_CONFIG 필요. export CUBLAS_WORKSPACE_CONFIG=:4096:8")
        sys.exit(1)
    if args.enable_model_check:
        r_3_3b = check_3_3B_model_forward(args.model)
    else:
        r_3_3b = {"passed": None, "skipped": True,
                   "reason": "model check 미활성 (--enable-model-check 필요, vast.ai 전용)"}

    print(f"[5/5] summary 작성")
    # Gate 분리 (hygiene): local smoke (3.1·3.2·3.3A) vs Step 3 final (+ 3.3B PASS).
    local_smoke_gate_passed = bool(
        r_3_1["passed"] and r_3_2["passed"] and r_3_3a["passed"]
    )
    step_03_final_gate_passed = bool(
        local_smoke_gate_passed and (r_3_3b.get("passed") is True)
    )
    # all_invariants_passed = step_03_final_gate_passed (final 의미로 통일)

    summary = {
        "step": 3,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "model_check_enabled": args.enable_model_check,
        "shapes": {
            "B": B, "H_kv": H_KV, "D": D,
            "num_layers": NUM_LAYERS, "T_total": T_TOTAL,
        },
        "chunk_spec_n_chunks": len(chunk_spec),
        "invariants": {
            "3.1_roundtrip_bitwise":            r_3_1,
            "3.2_chunk_meta_equality":          r_3_2,
            "3.3A_cache_interface_compat":      r_3_3a,
            "3.3B_model_forward_logits_equiv":  r_3_3b,
        },
        "local_smoke_gate_passed": local_smoke_gate_passed,
        "step_03_final_gate_passed": step_03_final_gate_passed,
        "all_invariants_passed": step_03_final_gate_passed,
    }
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_file}")

    # 콘솔 요약
    print()
    print(f"  3.1  (roundtrip K/V bitwise):       {'✅' if r_3_1['passed'] else '❌'}"
          f"  (mismatched_layers={r_3_1['mismatched_layers']})")
    print(f"  3.2  (ChunkMeta equality):          {'✅' if r_3_2['passed'] else '❌'}"
          f"  (mismatched_chunks={r_3_2['mismatched_chunk_ids']})")
    print(f"  3.3A (Cache interface compat):      {'✅' if r_3_3a['passed'] else '❌'}")
    if r_3_3b.get("skipped"):
        print(f"  3.3B (model forward logits):        ⏭️  생략 ({r_3_3b.get('reason')})")
    else:
        print(f"  3.3B (model forward logits):        {'✅' if r_3_3b['passed'] else '❌'}"
              f"  max_abs={r_3_3b.get('max_abs_diff')}")
        if r_3_3b.get("failure_case"):
            print(f"        failure_case: {r_3_3b['failure_case']}")

    # 3 분기 콘솔 출력 (사용자 spec)
    if not args.enable_model_check and local_smoke_gate_passed:
        print("==> local smoke gate PASS; Step 3 final gate pending (3.3B skipped)")
    elif args.enable_model_check and step_03_final_gate_passed:
        print("==> Step 3 final gate PASS")
    elif args.enable_model_check and not step_03_final_gate_passed:
        print("==> Step 3 final gate FAIL")
        sys.exit(1)
    else:
        # local smoke 자체 실패 (3.1/3.2/3.3A 중 어느 하나 FAIL)
        print("==> local smoke gate FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
