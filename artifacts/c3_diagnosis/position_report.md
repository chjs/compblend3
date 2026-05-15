# C-3 Position Report

## Tokenizer special tokens
- bos: `<s>` (id=1)
- eos: `</s>` (id=2)
- add_bos_token: True
- add_eos_token: False

## Tokenization (default policy)
- prefix_ids: [1, 415, 5565, 302, 4843, 349]
- prefix_tokens: ['<s>', '▁The', '▁capital', '▁of', '▁France', '▁is']
- full_ids: [1, 415, 5565, 302, 4843, 349, 5465]
- full_tokens: ['<s>', '▁The', '▁capital', '▁of', '▁France', '▁is', '▁Paris']
- next_ids: [5465]
- next_tokens: ['▁Paris']
- prefix_len: 6
- full_len: 7
- split_invariant (prefix_ids == full_ids[:, :prefix_len]): True

## All 3 policies — split invariant
| policy | prefix_len | full_len | split_ok | next_len_1 |
|---|---|---|---|---|
| default | 6 | 7 | True | True |
| add_special_True | 6 | 7 | True | True |
| add_special_False | 5 | 6 | True | True |

## Actual position_ids captured (via RoPE forward monkey-patch)
- rotary_emb.forward signature: `(x, position_ids)`
- full prefill: [0, 1, 2, 3, 4, 5, 6]
- prefix prefill: [0, 1, 2, 3, 4, 5]
- decode auto: [0, 1, 2, 3, 4, 5]
- decode explicit_correct: [0, 1, 2, 3, 4, 5]
- decode explicit_wrong: [0, 1, 2, 3, 4, 5]

## Variant max_abs_diff (decode last logits vs B full last logits)
- auto: 6.199e-06
- explicit_correct: 6.199e-06
- explicit_wrong: 1.094e+01

## Position-related conclusions
- auto position == explicit_correct position: **True**
- auto diff ≈ explicit_correct diff (within 1e-10): **True**
- explicit_wrong (position=0) produces noticeably different result: **True** (diff factor ≈ 1765485.8x)

## Earliest divergence
- first prefix KV diff layer (prefix-only vs full-slice): **1**
- first hidden_state diff layer (decode vs B full last-token): **8** (threshold 1e-7)

## Case inference (heuristic)
- case: **C**
- reason: prefix KV already differs at layer 1 (K max_abs > 1e-07) — strongly supported by prefill sequence-length-dependent numerical drift hypothesis

(자세한 layer-level diff는 `earliest_divergence.csv` 참조)

## 표현 원칙
이 보고는 다음 표현을 사용한다 (강한 단정 회피):
- "strongly supported by diagnostic evidence"
- "confirmed within the tested hypotheses"
- "consistent with fp32 GEMM/reduction-order numerical drift"
- "not attributable to position_ids under the tested configuration"
