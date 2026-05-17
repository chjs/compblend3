# Step 7 — HKVD 알고리즘 정확성 (numpy oracle 일치)

> 2026-05-18 (overnight): stub → 자체완결 spec 확장.
> HKVD score + recompute index 선택의 PyTorch implementation이 numpy oracle과 일치함을 검증.

---

## 1. 목표

HKVD (High KV Deviation) score formula 및 selective recompute index 선택의 알고리즘 정확성을 numpy oracle과 비교 검증.

## 2. HKVD 정의 (CC 자율 채택, CacheBlend paper 일반 패턴)

DECISIONS / GOAL / 노트에 명시적 score formula 부재. CacheBlend paper의 일반 패턴 + 사용자 spec §5.4의 "단순 부수 결정은 CC 자율 + report 명시"에 따라 다음 채택:

```
score(t, L) = || K_actual[L, :, t, :] − K_reference[L, :, t, :] ||_2   (per-token, per-layer L2 norm)
aggregated_token_score(t) = mean over L of score(t, L)
```

선택 정책:
```
recompute_ratio r ∈ [0, 1]
k = ceil(r * T)
top-k tokens (highest aggregated_token_score) → recompute 대상
tie-break: 같은 score 시 token index ascending (deterministic)
```

> **연구 왜곡 위험 노트**: 본 정의는 CC 자율 합리적 기본값. CacheBlend paper의 원문 정의를 사용자가 확정 후 갱신 가능. Step 8 진입 전 사용자 검토 권장.

## 3. Step 7 원칙

- fork 무수정.
- 신규 모듈 `src/compblend/hkvd.py` — PyTorch + numpy oracle 동시 구현.
- numpy oracle은 PyTorch 코드 공유 ❌ (독립 구현).
- GPU 불필요 → vast.ai ❌, MacBook/CPU.

## 4. Invariants

### 7.1 HKVD score Torch == numpy oracle (atol 1e-5)
```
(per_layer_token, aggregated)        = hkvd_score_torch(K_a, K_r)
(per_layer_token_np, aggregated_np)  = hkvd_score_numpy_oracle(K_a.numpy(), K_r.numpy())
allclose(per_layer_token, per_layer_token_np, atol=1e-5)
allclose(aggregated, aggregated_np, atol=1e-5)
```
Gate: atol 1e-5 (fp32 L2 norm 누적 오차).

### 7.2 selected indices == oracle (exact equality)
```
idx_torch  = select_recompute_indices_torch(score, ratio)
idx_oracle = select_recompute_indices_numpy_oracle(score.numpy(), ratio)
idx_torch.tolist() == idx_oracle.tolist()
```
Gate: list equality. ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, T ∈ {6, 16, 64}.

### 7.3 tie-breaking deterministic
score 일부 tie 만들고 PyTorch·oracle 모두 token index ascending tie-break으로 같은 index 반환.

### 7.4 shape generalization
multiple `(num_layers, H_kv, T, D)`: `(32, 8, 6, 128)`, `(4, 4, 16, 64)`, `(2, 2, 64, 32)`.

### 7.5 invalid input validation
- `recompute_ratio < 0` 또는 `> 1`: assertion 또는 ValueError
- shape mismatch: assertion

## 5. 구현 사양
`src/compblend/hkvd.py`:
- `hkvd_score_torch(k_actual, k_reference) -> (per_layer_token, aggregated_token)`
- `hkvd_score_numpy_oracle(k_actual_np, k_reference_np) -> (per_layer_token, aggregated_token)`
- `select_recompute_indices_torch(aggregated_score, ratio) -> indices`
- `select_recompute_indices_numpy_oracle(aggregated_score_np, ratio) -> indices`

검증 스크립트 `tasks/step_07_hkvd_oracle/run_hkvd_oracle_check.py`: 7.1·7.2·7.3·7.4·7.5 model-less.

### Tensor shape
| 변수 | shape |
|---|---|
| `K_actual / K_reference` | `(num_layers, H_kv, T, D)` |
| `per_layer_token_score` | `(num_layers, T)` |
| `aggregated_token_score` | `(T,)` |
| `selected_indices` | `(ceil(r * T),)` |

## 6. 검증 계측
직접 함수 호출 + assert. forward hook 없음.

## 7. 실행 환경
MacBook CPU 전용.

## 8. 결과 저장
`results/step_07/macbook/summary.json`.

## 9. Gate
| field | 조건 |
|---|---|
| **`step_07_final_gate_passed`** | 7.1 atol AND 7.2·7.3·7.4·7.5 exact |
| `all_invariants_passed` | `= step_07_final_gate_passed` |

## 10. 작업 순서
1. hkvd.py 작성.
2. 검증 스크립트.
3. MacBook smoke.
4. commit + push.
5. report + merge + tag.

## 11. 솔직성 노트
- HKVD score formula는 CC 자율 채택. CacheBlend paper 원문 확정 후 갱신 권장.
- 7.1 atol 1e-5: PyTorch torch.linalg.norm vs numpy.linalg.norm은 reduction order 차이로 fp32 bitwise 보장 ❌.
- selective recompute의 실제 forward 통합 ❌ (Step 8+ 또는 Phase 추후).
- mean aggregation은 합리적 기본값 — sum, weighted mean 등 다른 선택지도 가능.

## 12. 다음 step 예고
Step 8: Loong F1 측정 — recompute_ratio < 1.0 적용 forward + Loong dataset.
