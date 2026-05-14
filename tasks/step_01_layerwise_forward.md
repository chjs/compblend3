# Step 1 — Our layerwise forward = HF 표준 forward (no cache)

> 본격 시작 시 이 stub을 self-contained task 파일로 확장한다.

---

## 목표

HF transformers의 `modeling_mistral.py`를 fork해서 우리 repo로 가져오고 (`src/compblend/modeling/`),
HF 표준 forward와 **bitwise 일치**하는 forward를 구현한다.

이 단계에서는 코드를 그대로 가져오기만 한다 — CacheBlend 로직은 아직 안 들어감.
포인트는 우리 코드 베이스 안에서 같은 결과를 낼 수 있는 토대 확보.

## 사전 조건

- Step 0 통과
- Branch: `step/step_01_layerwise_forward`

## 통과 기준 / Invariants

### Invariant 1.1 — Forward 동등성
같은 입력에 대해 우리 forward와 HF 표준 forward의 logits SHA-256 동일.
```
sha256(our_forward(x)) == sha256(HF_forward(x))
```

### Invariant 1.2 — Layer-by-layer 동등성
모든 layer의 hidden state SHA-256 동일.
```
∀ layer i: sha256(our_h_i(x)) == sha256(HF_h_i(x))
```

### Invariant 1.3 — q/k/v projection 동등성
모든 layer의 attention input (q, k, v)가 element-wise 동일.

## 다음 step에서 확장할 내용

- 어떤 파일을 가져오고 어떻게 수정하나
- 그 외 self-contained 섹션들 (구현 사양, 실행 방법, 보고서 가이드 등)

Step 0 통과 후 이 stub을 본격 task 파일로 확장.
