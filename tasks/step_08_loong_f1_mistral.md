# Step 08 — Loong F1 측정 (Mistral)

> 이전 step 통과 후 이 stub을 self-contained task 파일로 확장한다.

---

## 목표

Mistral-7B-Instruct-v0.2 + Loong Level 1 (English, Academic) 에서:

1. CacheBlend 적용 시 (recompute_ratio < 1.0) F1 측정
2. Vanilla full prefill 대비 F1 parity 검증
3. 문서 순서 셔플 robustness 검증

## 사전 조건

- Step 07 통과 (HKVD oracle 일치 확인됨)
- Branch: `step/step_08_loong_f1_mistral`

## 통과 기준 / Invariants

본격 시작 시 명시적 invariant 정의. 현재 시점에서의 1차 가설:

- **F1 parity**: vanilla F1 대비 CacheBlend F1 손실 ≤ ε (ε = 1.0 F1 point 가설, 본격 시작 시 확정)
- **셔플 robustness**: 같은 instance에서 doc 순서 N회 셔플 시 F1 표준편차 ≤ noise band
- **Hit ratio**: token-level cache hit ratio 와 F1의 trade-off 곡선 기록

## Generation 방식

F1 측정은 generation 결과를 필요로 한다. Step 0~7은 prefill-only 였으나, 이 step에서는 generation까지.

**1차 가설** (본격 시작 시 확정):
- Decoding: **greedy** (temperature=0, do_sample=False)
- max_new_tokens: 256 (Loong answer 길이 분포 보고 조정)
- Stop condition: EOS token 또는 max_new_tokens
- **Generation 시점**의 KV cache 구성: prefill 완료 후 DynamicCache 그대로 사용 (CacheBlend 적용은 prefill 단계까지만)
- 즉, **"first-token-then-fresh-prefill"이 아니라** CacheBlend prefill 후 그 cache로 standard generation

**3-단계 분할 검증** (ChatGPT 검토 반영):

| 단계 | 측정 대상 | 통과 기준 |
|---|---|---|
| 8-A | prefill-only correctness | vanilla과 logits diff (atol) |
| 8-B | greedy decode smoke | max_new_tokens=1, 8, 64. CacheBlend vs vanilla의 생성 토큰 시퀀스 일치 (적어도 첫 1~8 token) |
| 8-C | task quality | F1 (Loong evaluation script) |

각 단계는 별도 sub-task. 8-A 통과 못 하면 8-B 진행 ❌.

## 평가 데이터셋

- Loong Level 1, English, Academic
- max_tokens 16K 이하 instance만 (Mistral 32K context 안전 마진)
- `data/manifests/loong_level1_eng_academic_16k.json` 사용
- 1차: 모든 instance 평가
- 결과 셔플 robustness 검증: 10개 instance × 5 셔플 = 50 측정

## 성능 비교 기준

DECISIONS.md §1 "성공의 정의" 4번:

- CacheBlend (recompute_ratio = 0.15) vs vanilla full prefill
- F1 손실 ≤ ε
- TTFT 단축: 비교 데이터로만 기록, primary metric ❌

**논문 숫자 재현은 목표 ❌** (DECISIONS.md §10). Trend reproduction (CacheBlend가 vanilla에 가까운 F1을 낸다는 패턴) + 우리 Loong 환경에서의 implementation equivalence.

## 결과 저장 형식

`results/step_08/vastai/summary.json`:

```json
{
  "step": 8,
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "dataset": "Loong Level 1 English Academic",
  "n_instances": ...,
  "generation": {
    "decoding": "greedy",
    "max_new_tokens": 256,
    "stop_tokens": [...]
  },
  "results": {
    "vanilla": {"f1": ..., "exact_match": ..., "avg_time_s": ...},
    "cacheblend_r0.15": {"f1": ..., "exact_match": ..., "hit_ratio_token": ..., "hit_ratio_doc": ..., "avg_time_s": ...},
    "cacheblend_r0.30": {...}
  },
  "shuffle_robustness": {
    "n_shuffles_per_instance": 5,
    "f1_std_per_instance": [...],
    "f1_std_overall": ...
  },
  "f1_delta_vs_vanilla": ...,
  "parity_pass": true | false
}
```

## 다음 step에서 확장할 내용

이 stub을 본격 task 파일로 확장 시:
- 정확한 ε threshold 결정 (F1 손실 허용량)
- Loong evaluation script 호출 방식 (Loong repo의 metric 코드 활용)
- generation 시 attention backend 선택 (eager? sdpa?)
- batch generation 가능성 (메모리 한도)
- shuffle seed 정책
- 보고서 가이드 (F1 curve, hit ratio curve)
- 다음 step (Phase 5 Llama-3.1) 게이트
