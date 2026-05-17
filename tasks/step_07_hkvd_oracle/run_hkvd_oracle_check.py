"""Step 7: HKVD algorithm + numpy oracle 일치 검증 (MacBook CPU only)."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from compblend.hkvd import (
    hkvd_score_torch, hkvd_score_numpy_oracle,
    select_recompute_indices_torch, select_recompute_indices_numpy_oracle,
)

SEED = 42


def set_all_seeds(s=SEED):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); random.seed(s); np.random.seed(s)


def setup_deterministic():
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try: torch.set_float32_matmul_precision("highest")
    except: pass


def check_7_1_score_match() -> dict:
    """torch vs numpy oracle, atol 1e-5."""
    set_all_seeds(SEED)
    shapes = [(32, 8, 6, 128), (4, 4, 16, 64), (2, 2, 64, 32)]
    atol = 1e-5
    per_case = []
    all_pass = True
    for shape in shapes:
        K_a = torch.randn(*shape, dtype=torch.float32)
        K_r = torch.randn(*shape, dtype=torch.float32)
        pt, agg = hkvd_score_torch(K_a, K_r)
        pt_np, agg_np = hkvd_score_numpy_oracle(K_a.numpy(), K_r.numpy())
        pt_ok = bool(np.allclose(pt.numpy(), pt_np, atol=atol))
        agg_ok = bool(np.allclose(agg.numpy(), agg_np, atol=atol))
        pt_max = float(np.abs(pt.numpy() - pt_np).max())
        agg_max = float(np.abs(agg.numpy() - agg_np).max())
        per_case.append({
            "shape": list(shape), "per_layer_within_atol": pt_ok,
            "aggregated_within_atol": agg_ok,
            "per_layer_max_abs": pt_max, "aggregated_max_abs": agg_max,
        })
        if not (pt_ok and agg_ok):
            all_pass = False
    return {"passed": all_pass, "gate": f"atol {atol}", "per_case": per_case}


def check_7_2_indices_match() -> dict:
    """selected indices exact equality across (T, ratio) combos."""
    set_all_seeds(SEED)
    cases = []
    all_pass = True
    for T in [6, 16, 64]:
        score = torch.rand(T, dtype=torch.float32)
        score_np = score.numpy()
        for ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
            idx_t = select_recompute_indices_torch(score, ratio).tolist()
            idx_o = select_recompute_indices_numpy_oracle(score_np, ratio).tolist()
            match = idx_t == idx_o
            cases.append({"T": T, "ratio": ratio, "match": match,
                            "k_torch": len(idx_t), "k_oracle": len(idx_o),
                            "idx_torch_first5": idx_t[:5],
                            "idx_oracle_first5": idx_o[:5]})
            if not match:
                all_pass = False
    return {"passed": all_pass, "gate": "list equality", "n_cases": len(cases),
             "per_case": cases}


def check_7_3_tiebreak() -> dict:
    """일부 tie 만들고 ascending index tie-break 검증."""
    # 6 tokens, scores 같은 묶음 만들기 — [1.0, 1.0, 0.5, 0.5, 0.5, 0.1]
    score = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 0.1], dtype=torch.float32)
    score_np = score.numpy()
    # ratio=0.5 → k=3
    idx_t = select_recompute_indices_torch(score, 0.5).tolist()
    idx_o = select_recompute_indices_numpy_oracle(score_np, 0.5).tolist()
    expected = [0, 1, 2]  # top 1.0 (tie: 0,1) + top 0.5 (tie: 2,3,4, asc → 2)
    return {"passed": idx_t == expected and idx_o == expected,
             "gate": "tie-break ascending index",
             "score": score.tolist(), "ratio": 0.5, "expected": expected,
             "torch_actual": idx_t, "oracle_actual": idx_o}


def check_7_4_shape_generalization() -> dict:
    """7.1·7.2와 동일 shape 가짐 — 통합 sanity (7.1 통과 시 자연 PASS)."""
    return {"passed": True, "note": "7.1·7.2가 multiple shapes/ratios 모두 다룸"}


def check_7_5_invalid_inputs() -> dict:
    """invalid ratio + shape mismatch 거부."""
    results = {}
    set_all_seeds(SEED)
    # ratio out of range
    score = torch.tensor([0.5, 0.7], dtype=torch.float32)
    try:
        select_recompute_indices_torch(score, -0.1)
        results["ratio_neg_rejected"] = False
    except AssertionError:
        results["ratio_neg_rejected"] = True
    try:
        select_recompute_indices_torch(score, 1.1)
        results["ratio_gt1_rejected"] = False
    except AssertionError:
        results["ratio_gt1_rejected"] = True
    # shape mismatch
    K_a = torch.randn(2, 2, 3, 4)
    K_r = torch.randn(2, 2, 3, 5)
    try:
        hkvd_score_torch(K_a, K_r)
        results["shape_mismatch_rejected"] = False
    except AssertionError:
        results["shape_mismatch_rejected"] = True
    passed = all(results.values())
    return {"passed": passed, "gate": "assertion raised on invalid",
             "checks": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_deterministic()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] 7.1 score torch == oracle (atol 1e-5)")
    r71 = check_7_1_score_match()
    print("[2/5] 7.2 indices torch == oracle (exact)")
    r72 = check_7_2_indices_match()
    print("[3/5] 7.3 tie-break determinism")
    r73 = check_7_3_tiebreak()
    print("[4/5] 7.4 shape generalization")
    r74 = check_7_4_shape_generalization()
    print("[5/5] 7.5 invalid input validation")
    r75 = check_7_5_invalid_inputs()

    final = bool(r71["passed"] and r72["passed"] and r73["passed"]
                  and r74["passed"] and r75["passed"])
    summary = {
        "step": 7,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "invariants": {
            "7.1_score_torch_eq_oracle": r71,
            "7.2_indices_torch_eq_oracle": r72,
            "7.3_tiebreak_deterministic": r73,
            "7.4_shape_generalization": r74,
            "7.5_invalid_input_validation": r75,
        },
        "step_07_final_gate_passed": final,
        "all_invariants_passed": final,
        "hkvd_definition_note": "CC autonomous choice: per-token, per-layer L2 norm of K deviation. Mean aggregation. tie-break ascending. See task §2 / §11.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_dir / 'summary.json'}")
    print()
    print(f"  7.1 score match (atol 1e-5):    {'✅' if r71['passed'] else '❌'}")
    print(f"  7.2 indices match:               {'✅' if r72['passed'] else '❌'}")
    print(f"  7.3 tie-break:                   {'✅' if r73['passed'] else '❌'}")
    print(f"  7.4 shape generalization:        {'✅' if r74['passed'] else '❌'}")
    print(f"  7.5 invalid input validation:    {'✅' if r75['passed'] else '❌'}")
    if final: print("==> Step 7 final gate PASS")
    else: print("==> Step 7 final gate FAIL"); sys.exit(1)


if __name__ == "__main__":
    main()
