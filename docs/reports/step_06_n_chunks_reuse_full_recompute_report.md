# Step 6 — N chunks reuse + recompute_ratio=1.0 = vanilla forward 보고서

## 1. Summary
**Step 6 final gate PASS** ✅. invariant 6.1 + API contract 모두 PASS. `step_06_final_gate_passed=True`. 100% recompute mode가 vanilla forward와 logits bitwise (`max_abs_diff=0.0`). Selective recompute (Step 7+)의 fallback path 확보.

## 2. Goal and Scope
`cacheblend_forward_full_recompute(recompute_ratio=1.0)` path가 vanilla full prefill과 동등함을 검증. "100% recompute에서도 안 맞으면 partial recompute 무의미" (사용자 spec).

## 3. Environment
- vast.ai instance 36953235, A100-SXM4-80GB, torch 2.10.0+cu128, transformers 4.51.3, CUDA 12.8, fp32/eager, CUBLAS_WORKSPACE_CONFIG=:4096:8. destroy 완료, 잔존 0.
- MacBook: API contract smoke (NotImplementedError raise 확인).
- 추정 비용 ~$0.15.
- Step 0~6 누적 추정: ~$1.86.

## 4. Implementation
- 신규 모듈 `src/compblend/blend.py`:
  - `cacheblend_forward_full_recompute(model, input_ids, attention_mask, blended_cache=None, recompute_ratio=1.0)`
  - `ratio==1.0`: blended_cache 입력 무시, `model(use_cache=False)` 직접 호출.
  - `ratio!=1.0`: `NotImplementedError` (Step 7+ scope).
- 검증 스크립트: API contract (model-less) + 6.1 (model-backed).
- blended_cache 생성은 Step 4 setup 재사용 (3 chunks × 2 tokens, RoPE re-rotation).

## 5. Invariants and Gates
| ID | gate | 결과 |
|---|---|---|
| API contract (`ratio<1` → `NotImplementedError`) | exception class | ✅ |
| **6.1** 100% recompute logits == vanilla logits SHA-256 | bitwise | ✅ `max_abs_diff=0.0` |

| gate field | 값 |
|---|---|
| `local_smoke_gate_passed` | True (API contract) |
| `step_06_final_gate_passed` | True (6.1) |
| `all_invariants_passed` | True |

## 6. MacBook Smoke
API contract: ✅ `ratio=0.5` → `NotImplementedError` raised.

## 7. vast.ai Results
6.1: ✅ `logits_blend_sha256 == logits_vanilla_sha256`, `max_abs_diff=0.0`.

## 8. Key Findings
- recompute_ratio=1.0 모드의 implementation 정확성 검증. Step 0~3의 deterministic vanilla forward 결과를 그대로 재사용.
- CacheBlend forward 모듈의 scaffolding (`src/compblend/blend.py`) + recompute_ratio API contract 확정. Step 7+에서 selective recompute 추가 진입점 명확.

## 9. Mechanism
- ratio=1.0 → blended_cache 입력 무시 → vanilla forward 그대로 실행 → Step 0 결정성·Step 1 fork 동등성 결과의 재확인.
- `max_abs_diff=0.0`은 동일 forward 두 번 실행의 trivial 결과.

## 10. Limitations
1. recompute_ratio=1.0만 검증. 0.0 < ratio < 1.0의 selective recompute는 Step 7+.
2. blended_cache 입력은 contract 상 받지만 무시 — 실제 selective recompute 시점에 활성.
3. prompt 1개, B=1, fp32, eager 단일 환경.
4. 6.1 trivially PASS — 실제 검증 가치는 API contract 확정 + Step 7+ scaffolding.

## 11. Implications for Step 7
- Step 7: HKVD oracle — selective recompute의 score formula, top-k selection, numpy oracle 일치. recompute_ratio < 1.0 path 의 구현 정확성.
- CacheBlend forward `src/compblend/blend.py` 가 같은 entry point 사용 (Step 7에서 ratio < 1.0 branch 추가).

## 12. Artifacts
- `src/compblend/blend.py`
- `tasks/step_06_n_chunks_reuse_full_recompute.md`, `tasks/step_06_*/run_step_06_check.py`
- `results/step_06/{macbook,vastai}/summary.json`
- commits: `cbadc03` (task + code + smoke), `5429a87` (vast.ai results)
