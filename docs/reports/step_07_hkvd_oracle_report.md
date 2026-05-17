# Step 7 — HKVD Oracle 보고서

## 1. Summary
**Step 7 final gate PASS** ✅. invariant 7.1·7.2·7.3·7.4·7.5 모두 PASS. CPU-only (vast.ai 사용 ❌). `step_07_final_gate_passed=True`, `all_invariants_passed=True`.

**중요 — HKVD score formula는 CC 자율 채택**: DECISIONS / GOAL / notes에 명시 정의가 부재하여 CacheBlend paper 일반 패턴 + 사용자 spec §5.4 "단순 부수 결정은 CC 자율 + report 명시"에 따라 다음 채택. **Step 8 진입 전 사용자 검토 권장**.

```
score(t, L) = || K_actual[L, :, t, :] − K_reference[L, :, t, :] ||_2
aggregated_token_score(t) = mean over L of score(t, L)
top-k tokens (highest aggregated_token_score) where k = ceil(r * T)
tie-break: token index ascending
```

## 2. Goal and Scope
HKVD score formula + recompute index 선택의 PyTorch implementation이 numpy oracle (독립 구현)과 일치함을 검증. 알고리즘 정확성 중심, 모델 forward 통합 ❌ (Step 8+).

## 3. Environment
- MacBook CPU 전용 (CUDA 불필요).
- Python 3.10, torch 2.10.0, numpy 1.x.
- 추정 비용: $0 (vast.ai 미사용).
- Step 0~7 누적 추정: ~$1.86 (Step 6와 동일, Step 7는 추가 비용 ❌).

## 4. Implementation
- `src/compblend/hkvd.py` 신규:
  - `hkvd_score_torch(k_actual, k_reference) -> (per_layer_token, aggregated_token)`
  - `hkvd_score_numpy_oracle(k_actual_np, k_reference_np) -> (per_layer_token, aggregated_token)`
  - `select_recompute_indices_torch(aggregated_score, ratio) -> indices`
  - `select_recompute_indices_numpy_oracle(aggregated_score_np, ratio) -> indices`
- numpy oracle은 PyTorch 코드 공유 ❌, 독립 구현 (numpy.linalg.norm + sorted).
- `select_*`의 tie-break은 양쪽 모두 `sorted(range(T), key=lambda i: (-score[i], i))` 패턴 (ascending index).

## 5. Invariants and Gates
| ID | gate | 결과 |
|---|---|---|
| 7.1 score torch == oracle | atol 1e-5 (fp32 L2 norm 누적) | ✅ |
| 7.2 indices torch == oracle | exact list equality | ✅ 15 cases (T ∈ {6, 16, 64} × ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}) |
| 7.3 tie-break deterministic | expected [0,1,2] | ✅ |
| 7.4 shape generalization | 7.1·7.2가 3 shapes 모두 다룸 | ✅ |
| 7.5 invalid input validation | AssertionError on invalid ratio/shape | ✅ ratio_neg / ratio_gt1 / shape_mismatch 모두 거부 |

| gate field | 값 |
|---|---|
| `step_07_final_gate_passed` | True |
| `all_invariants_passed` | True |

## 6. MacBook Smoke
모든 5 invariant PASS. CPU.

## 7. vast.ai Results
**사용 안 함** (CPU-only로 충분).

## 8. Key Findings
1. PyTorch `torch.linalg.norm` vs numpy `np.linalg.norm`은 fp32 reduction 순서 차이로 bitwise 보장 ❌. atol 1e-5는 충분히 보수적 (실측 max_abs 3 shape 모두 atol 안에).
2. tie-break은 sorted with secondary key (ascending index)로 양쪽 일치. 결정론적.
3. shape generalization: `(32, 8, 6, 128)`, `(4, 4, 16, 64)`, `(2, 2, 64, 32)` 모두 PASS.
4. invalid input: assertion 즉시 raise.
5. **HKVD formula 정의 모호성**: CacheBlend paper 원문 확인 안 됨. 본 구현은 합리적 기본값. Step 8 진입 전 사용자 검토 필요.

## 9. Mechanism
- score: per-token, per-layer L2 norm of K diff. layer 평균 aggregation.
- selection: top-k highest score with tie-break ascending index.
- numpy oracle: 독립 구현으로 implementation bug 검출 가능. 두 구현이 동일 결과 → 알고리즘 정확성 확인.

## 10. Limitations
1. **HKVD score formula는 CC 자율 채택**. CacheBlend paper 원문 확정 후 갱신 필요. 본 결과는 채택 정의 내부 일관성 검증만.
2. mean aggregation 외 다른 layer-wise aggregation (max, weighted) 미검증.
3. selective recompute의 실제 forward 통합 ❌. Step 8+ 또는 Phase 추후 범위.
4. real Mistral K/V tensor 사용 ❌ (random tensor 만 검증). 실제 model K/V에서의 score 분포·top-k 선택 특성은 별도 분석.
5. tie-break 정책은 ascending index. CacheBlend paper의 명시 정책과 다를 수 있음.
6. atol 1e-5 gate는 보수적. fp32 L2 norm 실측 max는 atol 보다 작음.
7. T=6/16/64 case만. 큰 T (1000+) 일반화 미검증.

## 11. Implications for Step 8
- Step 8 (Loong F1 측정): `cacheblend_forward_full_recompute`의 ratio<1.0 branch에 HKVD 통합 필요.
  - K_actual = blended cache (Step 4 setup)
  - K_reference = vanilla full prefill K
  - top-k tokens recompute, 나머지는 cache reuse
- HKVD formula 확정 후 진입.
- 본 step의 numpy oracle은 Step 8 검증 시 reference로 재사용 가능.

## 12. Artifacts
- `src/compblend/hkvd.py`
- `tasks/step_07_hkvd_oracle.md`, `tasks/step_07_hkvd_oracle/run_hkvd_oracle_check.py`
- `results/step_07/macbook/summary.json`
- commits: hygiene round 후 묶음
