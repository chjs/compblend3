#!/usr/bin/env python3
"""환경별 결과 비교 도구.

사용법:
    python scripts/compare_results.py --step 0
    python scripts/compare_results.py --step 0 --left vastai --right local_a100

기본 모드: vastai vs local_a100.
local_a100 결과가 없으면 vastai 단독 검증으로 판정 (대부분 step은 이 경우).

비교 강도 (자동 fallback):
    1순위: SHA-256 bitwise 일치
    2순위: atol 1e-5
    3순위: top-k logit 일치 (Step 8 같은 task metric)

차이 발생 시 어느 layer/position에서 갈렸는지 자동 리포트.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_summary(step: int, env: str) -> dict | None:
    """summary.json 로드. 없으면 None."""
    step_str = f"step_{step:02d}"
    path = Path("results") / step_str / env / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def vastai_only_check(step: int, summary: dict) -> tuple[bool, list[str]]:
    """vastai 단독 검증: invariant 통과 여부만 본다."""
    issues = []
    all_passed = summary.get("all_invariants_passed", False)
    if not all_passed:
        invariants = summary.get("invariants", {})
        for k, v in invariants.items():
            if isinstance(v, dict) and not v.get("passed", False):
                issues.append(f"  invariant {k}: FAIL — {v.get('description', '')}")
    return all_passed, issues


def compare_sha256(left: dict, right: dict) -> tuple[bool, list[str]]:
    issues = []
    if "logits_sha256" not in left or "logits_sha256" not in right:
        return False, ["logits_sha256 필드 없음"]
    if left["logits_sha256"] == right["logits_sha256"]:
        return True, []
    issues.append(
        f"SHA-256 불일치: left={left['logits_sha256'][:16]}... right={right['logits_sha256'][:16]}..."
    )
    return False, issues


def compare_numerical(left: dict, right: dict, atol: float = 1e-5) -> tuple[bool, list[str]]:
    issues = []
    if "logits_summary" not in left or "logits_summary" not in right:
        return False, ["logits_summary 필드 없음"]
    keys = ["max", "min", "mean", "norm"]
    all_ok = True
    for k in keys:
        l = left["logits_summary"].get(k)
        r = right["logits_summary"].get(k)
        if l is None or r is None:
            continue
        d = abs(l - r)
        if d > atol:
            issues.append(f"logits_summary.{k}: diff={d:.2e} > atol={atol}")
            all_ok = False
    return all_ok, issues


def compare_token_sequence(left: dict, right: dict) -> tuple[bool, list[str]]:
    issues = []
    l_tokens = left.get("generated_token_ids", [])
    r_tokens = right.get("generated_token_ids", [])
    if not l_tokens or not r_tokens:
        return False, ["generated_token_ids 필드 없음"]
    if l_tokens == r_tokens:
        return True, []
    diverge_at = None
    for i, (lt, rt) in enumerate(zip(l_tokens, r_tokens)):
        if lt != rt:
            diverge_at = i
            break
    if diverge_at is not None:
        issues.append(f"token sequence diverges at index {diverge_at}: left={l_tokens[diverge_at]} right={r_tokens[diverge_at]}")
    else:
        issues.append(f"token sequence 길이 다름: left={len(l_tokens)} right={len(r_tokens)}")
    return False, issues


def compare_layer_breakdown(left: dict, right: dict) -> list[str]:
    issues = []
    l_layers = left.get("layer_hashes", {})
    r_layers = right.get("layer_hashes", {})
    if not l_layers or not r_layers:
        return ["layer_hashes 없음"]
    common = set(l_layers.keys()) & set(r_layers.keys())
    for layer_id in sorted(common, key=lambda x: int(x) if x.isdigit() else x):
        if l_layers[layer_id] != r_layers[layer_id]:
            issues.append(f"  layer {layer_id}: 불일치")
    if not issues:
        issues.append("  모든 layer 일치 (다른 곳에서 차이)")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, help="Step number (0~8)")
    ap.add_argument("--left", default="vastai", help="Left env tag (default: vastai)")
    ap.add_argument("--right", default="local_a100", help="Right env tag (default: local_a100)")
    ap.add_argument("--atol", type=float, default=1e-5, help="Numerical tolerance (default: 1e-5)")
    args = ap.parse_args()

    print(f"==> Step {args.step:02d} 결과 비교")
    print()

    left = load_summary(args.step, args.left)
    right = load_summary(args.step, args.right)

    # 양쪽 다 없으면 에러
    if left is None and right is None:
        print(f"[ERROR] 양쪽 모두 결과 없음: {args.left}, {args.right}")
        sys.exit(1)

    # left만 있고 right 없음: 단독 모드 (대부분 step의 정상 케이스)
    if right is None:
        print(f"==> {args.right} 결과 없음 → {args.left} 단독 invariant 검증 모드")
        print()
        ok, issues = vastai_only_check(args.step, left)
        if ok:
            print(f"  ✅ {args.left} 단독으로 모든 invariant 통과")
            print(f"  ({args.right} 검증은 사용자가 결정한 step에서만 진행)")
            print()
            print("==> 단독 검증 PASS")
            sys.exit(0)
        else:
            print(f"  ❌ {args.left} invariant 실패:")
            for i in issues:
                print(i)
            sys.exit(1)

    # right만 있고 left 없음: 비대칭. 일반적이지 않음.
    if left is None:
        print(f"[WARN] {args.left} 결과 없음. {args.right} 단독 검증 진행.")
        ok, issues = vastai_only_check(args.step, right)
        sys.exit(0 if ok else 1)

    # 양쪽 다 있음: 본격 비교
    print(f"  {args.left}: timestamp={left.get('timestamp', 'N/A')}, env_tag={left.get('env_tag')}")
    print(f"  {args.right}: timestamp={right.get('timestamp', 'N/A')}, env_tag={right.get('env_tag')}")
    print()

    # Tier 1: SHA-256
    print("[Tier 1] SHA-256 bitwise 일치")
    ok, issues = compare_sha256(left, right)
    if ok:
        print("  ✅ 통과 — 두 환경 결정론 동일성 확인")
        print()
        print("==> 비교 결과: TIER 1 PASS (가장 강한 검증)")
        sys.exit(0)
    else:
        for i in issues:
            print(f"  - {i}")
        print()

    # Tier 2: atol
    print(f"[Tier 2] Numerical 비교 (atol={args.atol})")
    ok2, issues2 = compare_numerical(left, right, atol=args.atol)
    if ok2:
        print("  ✅ 통과 — numerical 동등")
    else:
        for i in issues2:
            print(f"  - {i}")
    print()

    # Tier 3: token sequence
    print("[Tier 3] Token sequence 일치")
    ok3, issues3 = compare_token_sequence(left, right)
    if ok3:
        print("  ✅ 통과 — 생성 토큰 시퀀스 동일")
    else:
        for i in issues3:
            print(f"  - {i}")
    print()

    # Layer breakdown
    print("[Diagnostic] Layer별 분석")
    for i in compare_layer_breakdown(left, right):
        print(i)
    print()

    if ok2:
        print("==> 비교 결과: TIER 2 PASS (수치 동등)")
        sys.exit(0)
    elif ok3:
        print("==> 비교 결과: TIER 3 PASS (토큰 동등, 수치는 차이)")
        sys.exit(0)
    else:
        print("==> 비교 결과: 모든 tier 실패 ❌")
        print("    docs/setup/troubleshooting.md '결과 비교' 섹션 참조")
        sys.exit(1)


if __name__ == "__main__":
    main()
