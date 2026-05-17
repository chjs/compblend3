# Step 5 — 1 chunk reuse = vanilla forward

> 2026-05-18 (overnight): stub → 자체완결 spec 확장.
> Step 3.3B의 N=1 특수 case 확장. 1 full chunk reuse path 가 vanilla forward와 동등함을 검증.

---

## 1. 목표

전체 prefix를 1개 chunk로 묶어 ChunkedKVStore에 저장한 후 decode path가 vanilla forward (DynamicCache 직접 사용) 와 동등한지 검증.

## 2. 사전 확인

| 항목 | 확인 |
|---|---|
| Step 3.3B | bitwise PASS (max_abs=0.0) at N chunks 분할. N=1 특수도 동일 mechanism |
| Step 4.2 | reorder concat bitwise. N=1 trivially |
| ChunkedKVStore.from_dynamic_cache / to_dynamic_cache | Step 3에서 검증 완료 |
| RoPE re-rotation | new_offset=0 = original_offset 시 shift=0, trivially identity |

## 3. Step 5 원칙

- fork 무수정. Step 3·4 모듈 (`src/compblend/cache.py`, `rope_rotation.py`) 재사용. 추가 모듈 ❌.
- chunk_spec: N=1, `original_offset=0, new_offset=0, original_length=T_prompt`.
- 5.1·5.2·5.4는 Step 3의 N=1 특수 case 재확인. 5.3 (post-update cache equivalence) 가 신규.

## 4. Invariants

### 5.1 one full chunk K/V roundtrip (model-less, bitwise)
Step 3.1의 N=1 특수 case. `torch.equal` per layer × K/V.

### 5.2 one chunk reuse decode equivalence (model-backed, bitwise)
Step 3.3B의 N=1 특수 case. `sha256(logits_a) == sha256(logits_b)`.

### 5.3 post-update cache equivalence (model-backed, bitwise)
**신규**. decode 후 `D_vanilla` vs `D_round`의 layer별 K/V `[:T_prompt+1]` 가 bitwise.

```
prefill_a: model(prompt, use_cache=True) → D_A
prefill_b: model(prompt, use_cache=True) → D_orig
D_round = ChunkedKVStore.from_dynamic_cache(D_orig, [1 chunk]).to_dynamic_cache()
decode A: model(next_token, past_key_values=D_A_copy) → updates cache to length T+1
decode B: model(next_token, past_key_values=D_round_copy) → updates cache to length T+1

∀ layer i: torch.equal(D_A_after_decode.key_cache[i][:, :, :T+1, :],
                       D_round_after_decode.key_cache[i][:, :, :T+1, :])
```

### 5.4 ChunkMeta 7-field equality (model-less, bitwise)
Step 3.2의 N=1 특수 case. dataclass equality.

## 5. 구현 사양

검증 스크립트 `tasks/step_05_one_chunk_reuse/run_one_chunk_reuse_check.py`:
- 5.1·5.4 model-less, MacBook 가능
- 5.2·5.3 `--enable-model-check` (vast.ai 전용)
- summary.json: invariants 4종 + gate fields

### Tensor shape
| 변수 | shape |
|---|---|
| `dc.key_cache[i]` (after prefill) | `(1, 8, 6, 128)` |
| `dc.key_cache[i]` (after decode) | `(1, 8, 7, 128)` |
| 비교 대상 slice | `[:, :, :7, :]` |

## 6. 검증 계측

- forward hook ❌. K/V 직접 접근.
- decode 시 cache mutate되므로 양쪽 copy 후 비교.

## 7. 실행 환경

| 환경 | invariants |
|---|---|
| MacBook | 5.1, 5.4 |
| vast.ai A100 | 5.1, 5.2, 5.3, 5.4 |

## 8. 결과 저장
`results/step_05/{macbook,vastai}/summary.json` — Step 3·4 패턴.

## 9. Gate
| gate field | 조건 |
|---|---|
| `local_smoke_gate_passed` | 5.1 AND 5.4 |
| **`step_05_final_gate_passed`** | 5.1 AND 5.2 AND 5.3 AND 5.4 (모두 bitwise) |
| `all_invariants_passed` | `= step_05_final_gate_passed` |

## 10. 작업 순서
1. Step 5 branch 생성.
2. 스크립트 작성.
3. MacBook smoke.
4. commit + push.
5. vast.ai 실행.
6. 보고서 작성.
7. main merge + tag.

## 11. 솔직성 노트
- Step 3.3B PASS가 5.2를 강하게 시사. 5.1도 Step 3.1 N=1 case라 PASS 기대. 5.3 (post-update)이 새로운 검증.
- 5.3 bitwise: D_A와 D_round의 K/V는 Step 3.1로 시작 시 동일 → forward에서 same input·same cache·same shape → bitwise 기대.

## 12. 다음 step 예고
Step 6: N chunks reuse, recompute_ratio=1.0 = vanilla (100% recompute로 drift 회복 검증).
