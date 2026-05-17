# Step 5 — 1 chunk reuse = vanilla forward 보고서

## 1. Summary
**Step 5 final gate PASS** ✅. invariant 5.1·5.2·5.3·5.4 모두 PASS. `step_05_final_gate_passed=True`, `all_invariants_passed=True`. 5.1·5.2·5.4는 Step 3 결과(N=1 특수 case)의 자연스러운 재확인. **5.3 (post-update cache equivalence, 신규)** 도 bitwise.

## 2. Goal and Scope
전체 prefix를 1개 chunk로 묶어 ChunkedKVStore에 저장한 후 decode path가 vanilla forward와 동등한지 검증.

## 3. Environment
- vast.ai instance 36952797, A100-SXM4-80GB, torch 2.10.0+cu128, transformers 4.51.3, CUDA 12.8, fp32/eager, CUBLAS_WORKSPACE_CONFIG=:4096:8. destroy 완료, 잔존 0.
- MacBook: 5.1, 5.4 model-less.
- 추정 비용 ~$0.15.
- Step 0~5 누적 추정: ~$1.71.

## 4. Implementation
- 신규 모듈 ❌. Step 3 `ChunkedKVStore` 재사용.
- 검증 스크립트 `tasks/step_05_one_chunk_reuse/run_one_chunk_reuse_check.py`.
- chunk_spec: N=1, full prefix (`original_offset=0, new_offset=0, original_length=T_prompt=6`).
- 5.3는 decode 후 `D_A_after_decode`와 `D_round_after_decode`의 `K/V[:T+1]` per-layer `torch.equal`.

## 5. Invariants and Gates
| ID | gate | 결과 |
|---|---|---|
| 5.1 one chunk roundtrip | bitwise | ✅ |
| 5.2 decode logits SHA-256 | bitwise | ✅ |
| 5.3 post-update cache equiv | bitwise per-layer | ✅ |
| 5.4 ChunkMeta equality | dataclass eq | ✅ |

## 6. MacBook Smoke
5.1 ✅ / 5.4 ✅ / 5.2 ⏭️ / 5.3 ⏭️ → `local_smoke_gate_passed=True`.

## 7. vast.ai Results
모두 PASS. decode token `5465 = "Paris"`. `step_05_final_gate_passed=True`.

## 8. Key Findings
- Step 3.3B의 N=1 case 확장이 mechanism적으로 PASS 기대대로 작동.
- 신규 5.3 (post-update cache): decode forward가 cache를 append (1 K/V) 한 후 양쪽 K/V[:T+1]이 bitwise. 동일 cache 시작 + 동일 forward shape → bitwise (Step 2 옵션 B mechanism 재확인).

## 9. Mechanism
- 5.1·5.4: Step 3 결과의 N=1 special case.
- 5.2: same input, same cache, same forward shape → bitwise (Step 2 옵션 B 패턴).
- 5.3: append-only `DynamicCache.update`가 동일 K/V를 추가 → bitwise.

## 10. Limitations
1. N=1 single chunk만 검증. multi-chunk + reuse는 Step 6+ 범위.
2. prompt 1개만. F1 일반화는 Step 8.
3. decode 1 token만. multi-token decode 미검증.
4. B=1 가정.
5. drift_budget 미적용 (Step 5 모든 invariant bitwise 가능).

## 11. Implications for Step 6
Step 6 (N chunks reuse + recompute_ratio=1.0)이 multi-chunk drift (Step 4.3 max=8.46)를 100% recompute로 회복하는지 검증. bitwise 또는 atol PASS 기대.

## 12. Artifacts
- `tasks/step_05_one_chunk_reuse/run_one_chunk_reuse_check.py`
- `results/step_05/{macbook,vastai}/summary.json`
- commits: `16a41dc` (task + code + smoke), `5b113b6` (vast.ai results)
