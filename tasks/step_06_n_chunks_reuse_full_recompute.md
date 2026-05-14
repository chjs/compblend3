# Step 06 — N chunks reuse, recompute_ratio=1.0 = vanilla forward

> 이전 step 통과 후 이 stub을 self-contained task 파일로 확장한다.

---

## 목표

N개의 chunk를 별도 prefill하여 KV cache를 만든 뒤, 그 cache를 reuse하면서 **모든 token을 모든 layer에서 재계산**(`recompute_ratio = 1.0`)하는 forward 경로를 구현.
이 경로가 vanilla full prefill과 동등함을 검증한다.

이것이 CacheBlend의 "selective recompute" 알고리즘의 정상 경로 검증 — 만약 100% recompute에서도 vanilla와 안 맞으면, partial recompute는 더 안 맞는다.

## 사전 조건

- Step 05 통과
- Branch: `step/step_06_n_chunks_reuse_full_recompute` (구체적 이름은 확장 시 명명)

## 통과 기준 / Invariants

이 step의 invariant는 **2개로 분리**한다. ChatGPT 검토(2026-05-14) 반영.

### Invariant 6.1 — Full recompute path == vanilla (bitwise)

`recompute_ratio = 1.0` 으로 모든 token을 모든 layer에서 재계산했을 때,
출력 logits가 vanilla full prefill과 **bitwise 일치** (SHA-256 동일).

**전제조건** (모두 만족해야 invariant 성립):
- 동일한 `input_ids`
- 동일한 `position_ids`
- 동일한 `attention_mask` (chunk-aware mask 미사용, vanilla causal mask와 동일)
- chunk separator 포함 여부 동일 (분리 tokenize했어도 최종 concat input_ids는 동일)
- 동일한 dtype (fp32)
- 동일한 attention backend (eager)
- `model.eval()`, dropout 비활성
- 같은 seed
- KV cache가 chunk별로 분리 저장되어 있어도, attention 계산 시 들어가는 K/V tensor가 vanilla와 element-wise 동일
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.use_deterministic_algorithms(True)`

```
sha256(blend_logits @ ratio=1.0) == sha256(vanilla_logits)
```

### Invariant 6.2 — Token-level q/k/v == vanilla (selected token, atol)

특정 layer L과 token position p에 대해, 그 layer의 입력 hidden state가 vanilla와 같을 때,
recompute path가 만든 q/k/v가 vanilla의 그 위치 q/k/v와 일치 (atol 1e-5).

**주의 — 전제조건의 중요성**:
"recomputed token의 q/k/v가 vanilla와 같다"는 명제는 **Layer L의 입력 hidden state가 vanilla와 동일할 때만 성립**한다.
CacheBlend가 일부 token의 KV만 업데이트하고 나머지는 reuse하면, 중간 layer의 hidden state가 vanilla와 달라진다 → q/k/v도 달라진다.
따라서 Invariant 6.2는 **Invariant 6.1과 같은 조건 (100% recompute, 모든 layer 입력이 vanilla와 동일)** 에서만 의미가 있다.

partial recompute (recompute_ratio < 1.0) 의 검증은 Step 07/08에서 별도 invariant로:
- HKVD 선택 인덱스가 oracle과 일치 (Step 07)
- Blend path의 logits가 vanilla와 numerical 동등 (atol, Step 08)
- Task quality (F1) 가 vanilla 대비 ε 이내 유지 (Step 08)

### Invariant 6.3 — Chunk ordering invariance (preview)

여러 chunk를 다른 순서로 prefill해도 (system 고정, doc 셔플), `recompute_ratio = 1.0` 에서 logits가 동등.

이건 Step 06의 통과 기준에 포함하지 말고 **Step 08에서 본격 검증** (셔플 robustness가 F1 차원에서 어떻게 보이는지가 본질). Step 06에서는 sanity check로 1~2회 셔플 시도해보고 결과만 기록.

## 다음 step에서 확장할 내용

이 stub을 본격 task 파일로 확장 시:
- ChunkedKVStore → DynamicCache 변환 코드의 정확한 시그니처
- 100% recompute path의 구현 (어느 함수에서 분기?)
- Vanilla full prefill의 SHA-256 baseline 어떻게 측정/저장하나
- Bitwise 일치가 안 될 때의 디버깅 절차 (layer별 hash 비교, position별 diff)
- 결과 저장 schema (per_layer_hashes, blend_logits_sha256, vanilla_logits_sha256)
- 보고서 가이드
- 다음 step (Step 07 HKVD) 게이트
