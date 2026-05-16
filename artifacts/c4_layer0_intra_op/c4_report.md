# C-4 Layer 0 Intra-op Divergence Report

## Config
- model: mistralai/Mistral-7B-Instruct-v0.2, transformers 4.51.3, torch 2.10.0+cu128
- dtype: torch.float32, attn_implementation: eager, sliding_window: None
- prefix_len: 6, full_len: 7, next_ids: [5465]

## §1. Mask
- mask_prefix shape: [1, 1, 6, 6], min=-3.4028234663852886e+38, max=-0.0
- mask_full shape: [1, 1, 7, 7], min=-3.4028234663852886e+38, max=-0.0
- mask_prefix == mask_full[:, :, :prefix_len, :prefix_len]: max_abs=0.0, allclose_0=True
- mask_full future column (query rows 0..5, key 6): max=-3.4028234663852886e+38, fully_masked=True

## §2. RoPE cos/sin
- cos prefix vs full[:prefix_len]: max_abs=0.000e+00, allclose_0=True
- sin prefix vs full[:prefix_len]: max_abs=0.000e+00, allclose_0=True

## §4. Manual Layer 0 vs HF Layer 0 (sanity)
- prefix: max_abs=0.000e+00, allclose_0=True
- full: max_abs=0.000e+00, allclose_0=True

(manual reproduces HF if these are ~ 0)

## §3. Intra-op divergence — **첫 발산 지점**: `02_q_proj`

전체 표는 `intra_op_divergence.csv` 참조. 핵심 단계:
- 00_hidden_in: max_abs=0.000e+00, allclose_0=True
- 01_input_layernorm: max_abs=0.000e+00, allclose_0=True
- 02_q_proj: max_abs=2.384e-06, allclose_0=False
- 03_k_proj: max_abs=0.000e+00, allclose_0=True
- 04_v_proj: max_abs=0.000e+00, allclose_0=True
- 05_q_reshape: max_abs=2.384e-06, allclose_0=False
- 06_k_reshape: max_abs=0.000e+00, allclose_0=True
- 07_v_reshape: max_abs=0.000e+00, allclose_0=True
- 08_q_rot: max_abs=2.384e-06, allclose_0=False
- 09_k_rot: max_abs=0.000e+00, allclose_0=True
- 10_k_rep: max_abs=0.000e+00, allclose_0=True
- 11_v_rep: max_abs=0.000e+00, allclose_0=True
- 12_attn_raw: max_abs=None, allclose_0=None
- 13_attn_masked: max_abs=None, allclose_0=None
- 14_attn_probs: max_abs=None, allclose_0=None
- 15_attn_output_pre_oproj: max_abs=4.098e-08, allclose_0=False
- 16_o_proj: max_abs=9.313e-09, allclose_0=False
- 17_post_attn_residual: max_abs=9.313e-09, allclose_0=False
- 18_post_attn_layernorm: max_abs=9.537e-07, allclose_0=False
- 19_gate_proj: max_abs=3.874e-07, allclose_0=False
- 20_up_proj: max_abs=4.768e-07, allclose_0=False
- 21_act_silu: max_abs=1.192e-07, allclose_0=False
- 22_down_proj: max_abs=2.980e-08, allclose_0=False
- 23_layer0_output: max_abs=2.980e-08, allclose_0=False

## §5. Attention alternatives
- A_prefix vs A_full_slice (HF의 측정 divergence): max_abs=4.098e-08
- A_prefix vs B_full_slice (full q/k/v를 length=6으로 잘라 attention): max_abs=4.098e-08
- A_full_slice vs B_full_slice (같은 q/k/v, mask length=7 vs 6 softmax): max_abs=0.000e+00
- A_prefix vs B_prefix (sanity): max_abs=0.000e+00

해석:
- A_prefix vs B_full_slice 작으면 → full q/k/v를 잘라서 length=6 softmax하면 prefix와 ~동일 → softmax length가 핵심
- A_full_slice vs B_full_slice가 큰 만큼 softmax length 7 vs 6의 차이가 measurable

## §6. use_cache effect
- prefix nc vs wc: max_abs=0.000e+00
- full nc vs wc: max_abs=0.000e+00
- L0 prefix_nc vs full_nc[:prefix_len]: max_abs=2.980e-08
- L0 prefix_wc vs full_wc[:prefix_len]: max_abs=2.980e-08

## §7. sliding_window
- original sliding_window: None
- config.sliding_window=None — 별도 disable 불필요

## 표현 원칙
보고서에서 "100% confirmed", "guaranteed", "impossible"는 회피. 대신:
- "strongly supported by diagnostic evidence"
- "confirmed within the tested hypotheses"
- "consistent with [specific mechanism]"
- "not attributable to [X] under the tested configuration"
