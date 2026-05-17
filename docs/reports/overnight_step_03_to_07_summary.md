# Overnight Round: Step 3 finalization → Step 7 (2026-05-17 → 2026-05-18)

> 사용자 standing approval 기반 자율 진행. 내일 아침 리뷰용 요약.

---

## 1. 시작 상태 (round 진입 직전)

- main HEAD: `71f9c03` (step_02_done)
- step/step_03_chunked_kv_store HEAD: `c29ceeb` (Step 3 final gate PASS, 보고서 작성 완료, merge 대기)
- Tags: `step_00_done`, `step_01_done`, `step_02_done` (3개)
- vast.ai 잔존: 0개

## 2. 완료한 step / skip 한 step

| step | 결과 | round 의미 |
|---|---|---|
| Step 3 finalization | ✅ merge + tag | §1 — 직전 round 결과 활용 |
| Step 4 | ✅ merge + tag | §2 — 신규 RoPE re-rotation module |
| Step 5 | ✅ merge + tag | §3 — Step 3 N=1 특수 + 신규 5.3 post-update cache |
| Step 6 | ✅ merge + tag | §4 — blend module scaffold (recompute_ratio=1.0 only) |
| Step 7 | ✅ merge + tag | §5 — HKVD oracle (CC 자율 formula) |
| Step 8 진입 | ❌ skip | 사용자 spec 명시 (Step 7에서 정지) |

## 3. 각 step branch / merge commit / tag / report / summary

| step | branch (삭제됨) | merge commit (main) | tag | report | summary.json |
|---|---|---|---|---|---|
| 3 | step/step_03_chunked_kv_store | `527fb92` | `step_03_done` | `docs/reports/step_03_chunked_kv_store_report.md` | `results/step_03/{macbook,vastai}/summary.json` |
| 4 | step/step_04_multi_chunk_concat | `ad8ef29` | `step_04_done` | `docs/reports/step_04_multi_chunk_concat_report.md` | `results/step_04/{macbook,vastai}/summary.json` |
| 5 | step/step_05_one_chunk_reuse | `6447ffe` | `step_05_done` | `docs/reports/step_05_one_chunk_reuse_report.md` | `results/step_05/{macbook,vastai}/summary.json` |
| 6 | step/step_06_n_chunks_reuse_full_recompute | `3a547cd` | `step_06_done` | `docs/reports/step_06_n_chunks_reuse_full_recompute_report.md` | `results/step_06/{macbook,vastai}/summary.json` |
| 7 | step/step_07_hkvd_oracle | `ab1b70d` | `step_07_done` | `docs/reports/step_07_hkvd_oracle_report.md` | `results/step_07/macbook/summary.json` |

## 4. 각 step invariant 결과 (PASS/FAIL/skipped)

| Step | invariant | gate | 결과 |
|---|---|---|---|
| 3 | 3.1 roundtrip K/V | bitwise | ✅ |
| 3 | 3.2 ChunkMeta equality | dataclass eq | ✅ |
| 3 | 3.3A Cache interface compat | 5 조건 AND | ✅ |
| 3 | 3.3B model forward logits | SHA-256 | ✅ (max_abs=0.0) |
| 4 | 4.1 RoPE re-rotation self-consistency | atol 1e-6 (사전 bitwise 가정 정정) | ✅ |
| 4 | 4.2 ChunkedKVStore reorder concat | bitwise | ✅ |
| 4 | 4.3 multi-chunk vanilla equivalence | measurement only (gate ❌) | drift max=8.46, argmax_match=True, top5=3/5 |
| 5 | 5.1 one chunk roundtrip | bitwise | ✅ |
| 5 | 5.2 decode logits | bitwise | ✅ |
| 5 | 5.3 post-update cache | bitwise per-layer | ✅ |
| 5 | 5.4 ChunkMeta | dataclass eq | ✅ |
| 6 | API contract (ratio<1 → NotImplementedError) | exception | ✅ |
| 6 | 6.1 full recompute == vanilla | bitwise SHA-256 | ✅ (max_abs=0.0) |
| 7 | 7.1 score torch == oracle | atol 1e-5 | ✅ |
| 7 | 7.2 indices torch == oracle | exact list eq | ✅ (15 cases) |
| 7 | 7.3 tie-break deterministic | expected list | ✅ |
| 7 | 7.4 shape generalization | 3 shapes | ✅ |
| 7 | 7.5 invalid input validation | assertion | ✅ |

## 5. vast.ai 사용 이력 (overnight round)

| step | instance ID | running 시간 | dph_total (show) | 추정 비용 | destroy |
|---|---|---|---|---|---|
| 3 (직전 round) | 36936503 | 322s | $1.205/h | ~$0.15 | ✅ |
| 4 (첫 시도, stuck) | 36951804 | — (loading) | — | ~$0.00 | ✅ (destroy 후 재할당) |
| 4 (재할당) | 36952360 | ~10분 | ~$1.07/h | ~$0.20 | ✅ |
| 5 | 36952797 | ~7분 | ~$1.21/h | ~$0.15 | ✅ |
| 6 | 36953235 | ~7분 | ~$1.21/h | ~$0.15 | ✅ |
| 7 | — | — | — | $0 (MacBook CPU) | — |
| **overnight 합계 (3 이후)** | | | | **~$0.65** | |

추정치, 정확 비용은 vast.ai 콘솔 기준 확인 필요. 잔존 인스턴스 0개.

| Step 0~7 누적 추정 비용 | ~$2.36 |
|---|---|
| Step 0 / 1 / 2 / 3 / 4 / 5 / 6 / 7 | ~$0.05 / ~$0.16 / ~$1.0 / ~$0.15 / ~$0.20 / ~$0.15 / ~$0.15 / $0 |

## 6. 실패 / 재검토 필요 항목

| 항목 | 종류 | 우선순위 | 비고 |
|---|---|---|---|
| **Step 7 HKVD formula 정의** | 임의 채택 | 🔴 HIGH | DECISIONS / GOAL에 명시 부재. CC 자율로 per-token L2 norm + mean aggregation + ascending tie-break 채택. **Step 8 진입 전 사용자 확정 필요** (CacheBlend paper 원문 vs 채택 정의 일치 확인). report §11에 상세. |
| Step 4 4.3 drift max=8.46 | 측정만 (gate ❌) | 🟡 MEDIUM | mechanism적 기대치 (100% reuse w/o recompute). Step 7 selective recompute로 회복 예정. argmax 유지·top5=3/5 |
| Step 4 첫 instance 36951804 stuck | 인프라 | 🟢 LOW | offer 운 — destroy 후 재할당으로 해결. vast_helper.py 의 600s timeout 정상 동작 |
| Step 5/6 trivially PASS | 검증 가치 | 🟢 LOW | 5.1·5.2·5.4 = Step 3 N=1 case, 6.1 = vanilla 두 번 forward. 의미는 API contract + scaffolding |
| 4.1 사전 bitwise 가정 정정 | 명세 정확성 | 🟢 LOW | fp32 RoPE composition은 bitwise 불가 (cos(a+b) ≠ cos a cos b − sin a sin b). atol 1e-6으로 정정 후 task spec·script·report 일관 |
| 다른 진단 스크립트 RoPE hook 첫 call defect | 기술 부채 | 🟢 LOW | Step 2 C-3 진단 인프라. 향후 hook 사용 step에서 fix |

## 7. 사용자 내일 아침 리뷰 우선순위

1. **🔴 HKVD formula 정의 (Step 7)** — `src/compblend/hkvd.py` + `tasks/step_07_hkvd_oracle.md` §2 + `docs/reports/step_07_hkvd_oracle_report.md` §1·§11. CacheBlend paper 원문 정의와 비교 후 확정.
2. **Step 4 보고서** — RoPE re-rotation module + 4.3 drift measurement 해석.
3. **Step 6 blend module scaffold** — `src/compblend/blend.py`. recompute_ratio API contract 검토 (Step 7+에서 확장 예정).
4. Step 3·5의 trivial PASS는 빠르게 읽고 통과 가능.

## 8. 다시 수행할 가능성 있는 step

| step | 재수행 시나리오 |
|---|---|
| Step 7 | HKVD formula 정정 시 → `hkvd.py` + 검증 코드 + report 갱신 후 재실행 |
| Step 4 | drift threshold 변경 또는 chunk_T·n_chunks 일반화 추가 시 |
| Step 8 진입 시 | Step 7 formula 확정 후, Step 6 `blend.py`에 recompute_ratio<1.0 branch 추가 + Loong dataset 통합 필요 |

## 9. Step 8 진입 전 필요한 결정 (사용자)

1. **HKVD score formula 확정** — 최우선. CacheBlend paper 원문 확인 + CC 채택 정의 (per-token L2 + mean) 비교.
2. **HKVD aggregation 방식** — mean vs sum vs weighted vs max. 본 채택은 mean.
3. **tie-break 정책** — ascending index vs random vs paper 원문 방식.
4. **selective recompute의 forward 통합 설계** — `cacheblend_forward(model, ids, mask, blended_cache, recompute_ratio=r<1.0)`의 algorithmic detail. 어느 layer에서 HKVD 적용? 모든 layer? 또는 first-layer-only?
5. **Loong dataset 통합** — `data/loong/` 구성 (DECISIONS §3.6). Phase 0-B Loong manifest는 아직 미수행 (PROGRESS §Phase 0-B.5).
6. **Step 8 F1 측정 환경** — vast.ai A100 80GB로 충분한지, multi-GPU 필요한지.

## 10. 최종 사용자 보고 (round 종료 상태)

| 항목 | 값 |
|---|---|
| 현재 main HEAD | `ab1b70d3...` (Step 7 merge) |
| 누적 tags | `step_00_done` ~ `step_07_done` (8개) |
| Step 4~7 완료 | ✅ 4 step 모두 final gate PASS, merge + tag 완료 |
| 실패 stop point | 없음 — 모든 step PASS |
| 남은 branch | 없음 (모두 main merge 후 삭제) |
| vast.ai 잔존 | 0개 (모든 인스턴스 destroy 확인) |

내일 아침 리뷰 순서:
1. 본 overnight summary 먼저.
2. **Step 7 보고서 + `src/compblend/hkvd.py`** (HKVD formula 검토 — 최우선).
3. Step 4 보고서 (RoPE re-rotation, 4.3 drift 해석).
4. Step 6 보고서 (blend module scaffold).
5. Step 3·5 보고서 (trivial PASS 확인).
6. PROGRESS.md (Step 0~7 모두 ✅ 상태).
7. Step 8 진입 결정 (HKVD formula 확정 + 필요 환경 검토).
