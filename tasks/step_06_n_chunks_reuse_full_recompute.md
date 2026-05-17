# Step 6 — N chunks reuse, recompute_ratio=1.0 = vanilla forward

> 2026-05-18 (overnight): stub → 자체완결 spec 확장.
> CacheBlend forward의 100% recompute path가 vanilla full prefill과 동등함을 검증. selective recompute (HKVD)의 fallback path 확보.

---

## 1. 목표

N개 chunk 별도 prefill → blended cache → `cacheblend_forward_full_recompute(recompute_ratio=1.0)` path가 vanilla full prefill과 logits 동등함을 검증.

> "100% recompute에서도 vanilla와 안 맞으면 partial recompute 무의미" (사용자 spec)

## 2. 사전 확인

| 항목 | 확인 |
|---|---|
| Step 4 | multi-chunk blended cache vs vanilla decode drift max=8.46 (100% reuse w/o recompute) |
| Step 6 mechanism | recompute_ratio=1.0 → blended cache 무시, vanilla full prefill 실행 → 동일 logits 기대 |
| Step 7 향후 | recompute_ratio < 1.0 (HKVD selective) 는 Step 7 |

## 3. Step 6 원칙

- fork 무수정.
- 신규 모듈 `src/compblend/blend.py` — `cacheblend_forward_full_recompute` (minimal scaffold, Step 7+ 확장).
- recompute_ratio=1.0만 지원. 다른 값 `NotImplementedError`.

## 4. Invariants

### 6.1 100% recompute path == vanilla (model-backed, bitwise)
```
logits_blend_full = cacheblend_forward_full_recompute(model, ids, mask, blended_cache, ratio=1.0)
logits_vanilla    = model(ids, attention_mask=mask, use_cache=False).logits
sha256(logits_blend_full) == sha256(logits_vanilla)
```
Gate: bitwise.

### 6.2 (optional) selected token q/k/v at layer L
Step 6 scope에서는 6.1로 충분. final gate에 포함 ❌.

## 5. 구현 사양
- `src/compblend/blend.py`: `cacheblend_forward_full_recompute(model, input_ids, attention_mask, blended_cache=None, recompute_ratio=1.0)`
  - ratio=1.0: blended_cache 무시 + vanilla forward (use_cache=False) 실행.
  - 다른 ratio: NotImplementedError.
- 검증 스크립트 `tasks/step_06_n_chunks_reuse_full_recompute/run_step_06_check.py`:
  - 6.1 vast.ai 전용 (model-backed).
  - blended_cache 생성: Step 4 setup (3 chunks × 2 tokens, RoPE re-rotation).

## 6. 검증 계측
forward hook ❌. logits SHA-256 직접 비교.

## 7. 실행 환경
- vast.ai 전용 (model 필요).
- MacBook은 py_compile + import + API contract smoke (ratio=0.5 → NotImplementedError).

## 8. 결과 저장
`results/step_06/{macbook,vastai}/summary.json`.

## 9. Gate
| field | 조건 |
|---|---|
| `local_smoke_gate_passed` | py_compile + import + ratio=0.5 raises NotImplementedError |
| **`step_06_final_gate_passed`** | 6.1 bitwise (vast.ai) |
| `all_invariants_passed` | `= step_06_final_gate_passed` |

## 10. 작업 순서
1. blend.py 작성.
2. 검증 스크립트.
3. MacBook smoke (api sanity).
4. commit + push.
5. vast.ai 6.1 실행.
6. report + merge + tag.

## 11. 솔직성 노트
- 6.1은 mechanism적으로 trivially PASS 기대 — recompute_ratio=1.0 = vanilla forward 이므로 cache 입력 무시 + vanilla = vanilla. Step 0 결정성 결과의 직접 재확인.
- Step 6의 진짜 가치는 **CacheBlend forward 모듈 scaffolding** + recompute_ratio API contract 확정. selective recompute (Step 7+) 가 같은 entry point 사용.
- recompute_ratio < 1.0가 NotImplementedError raise 명시 검증 (API contract).

## 12. 다음 step 예고
Step 7: HKVD oracle — selective recompute의 score formula + numpy oracle 일치.
