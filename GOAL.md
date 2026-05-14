# GOAL.md — compblend3 프로젝트 목표

> 이 파일은 작업을 시작할 때마다 가장 먼저 읽는다.
> 길지 않다. 1분 안에 다 읽힌다.
> 변하지 않는다.

---

## 우리가 풀려는 문제

LLM 서빙에서 KV cache는 메모리와 대역폭의 병목이다.

- **KV cache reuse (blending)**: 여러 chunk의 KV를 재조합해 prefill 비용을 줄인다. (CacheBlend, EPIC)
- **KV cache compression**: KV의 크기를 줄여 메모리/대역폭을 줄인다. (KVzip, SnapKV, KIVI)

이 둘은 보완적이지만, 압축된 KV를 chunk reuse 시나리오에서 다시 쓰는 방법은 아직 잘 풀려있지 않다.

이전 두 시도(CompBlend-old, compblend2)는 이 방향으로 가다가 다음 문제에서 막혔다:

- **(a) 코드 검증이 어려움** — CacheBlend 구현이 원본 CacheBlend와 정말 같은지 확신하기 어려웠음
- **(b) F1 성능이 논문대로 안 나옴**

이 두 문제는 사실 같은 문제의 두 얼굴이다. **검증이 안 되니까 어디가 잘못됐는지 모르고, 그래서 성능이 안 나와도 원인을 못 찾는다.**

## 최종 목표

**CompBlend** 구현 — 압축된 KV cache를 chunk reuse 시나리오에서 안전하게 재사용하는 시스템.

- Gated HKVD: 압축 기반 importance를 gate로, HKVD를 그 안에서 적용
- Pluggable compression backend (1차 backend = KVzip)
- RoPE-aware position realignment

## 현재 phase의 목표

**검증 가능한 CacheBlend의 HuggingFace Transformers 재구현.**

이 토대 위에서만 CompBlend로 확장한다. 토대가 흔들리면 그 위의 모든 결과를 의심해야 한다.

## 시나리오

```
system prompt + doc_1 + doc_2 + ... + doc_N + user query
```

- doc들의 크기, 갯수, 순서가 다양함
- 모든 doc이 답에 필요 (Loong 데이터셋 특성: Leave No Document Behind)
- 문서 순서가 바뀌어도 blending 결과 F1이 robust해야 함

## 모델

- **1차 검증**: Mistral-7B-Instruct-v0.2 (논문 모델)
- **일반화 검증**: Llama-3.1-8B

## 평가

- **데이터셋**: Loong (EMNLP 2024)
- **Primary metric**: vanilla full prefill 대비 F1 parity
- **Secondary**: KV cache reuse ratio, recompute ratio

## 성공의 정의

이 프로젝트는 다음 4개가 모두 충족되었을 때 "성공"이라 부른다:

1. **CacheBlend HF 구현이 invariant 검증을 모두 통과** (Step 0~7)
2. **Mistral-7B-v0.2 + Loong Level 1에서 vanilla full prefill 대비 F1 parity 달성** (Step 8)
3. **문서 순서를 셔플해도 F1 표준편차가 noise band 안** (CacheBlend의 핵심 selling point)
4. **CompBlend(KVzip backend + Gated HKVD)가 동작하고, 압축 미적용 대비 F1 손실이 작음** (Phase 7~8)

각 step의 통과 기준은 `tasks/step_XX_*.md`에 명세된 invariant를 따른다.

## 비목표 (이번 phase에서 안 함)

- vLLM/LMCache와의 직접 bitwise 일치 (구조적으로 불가)
- TTFT 최적화 (1차 목표는 correctness, 성능은 그 다음)
- Multi-token decode 최적화 (first-token-then-fresh-prefill 방식 우선)
- Quantization 기반 compressor 통합 (Phase 8+)

## 솔직성 약속

- **확신 없는 정보는 명시**. "확인 필요"라고 적는다.
- **F1이 안 나오면 어디서 안 나오는지 추적 가능한 구조**를 만든다.
- **CompBlend-old, compblend2의 결과를 답습하지 않는다**. 참고만 한다.
- **negative result도 그대로 보고**한다. paper-headline을 만들기 위해 데이터를 굽지 않는다.
