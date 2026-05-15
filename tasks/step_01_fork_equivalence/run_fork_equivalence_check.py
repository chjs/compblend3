"""Step 1: fork 동치성 검증 — fork된 코드 = HF 표준 forward (no cache).

순차 로드 — HF 표준 → forward → 결과 CPU 이동 → del → our fork → forward → CPU 비교.
invariant 1.1 (logits SHA-256 동일), 1.2 (layer hidden state SHA-256 동일),
1.3 (q/k/v_proj 출력 element-wise 동일 — RoPE 적용 전).

fork 코드(src/compblend/modeling/)는 무수정 — 모든 계측은 이 스크립트의 외부
forward hook으로만 한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from compblend.modeling import MistralForCausalLM

SEED = 42
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PROMPT = "The capital of France is"
QKV_PROJ_NAMES = ("q_proj", "k_proj", "v_proj")


def set_all_seeds(seed: int = SEED) -> None:
    """결정론을 위한 모든 seed 설정 (Step 0와 동일 패턴)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def setup_deterministic() -> None:
    """PyTorch deterministic mode."""
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tensor_sha256(t: torch.Tensor) -> str:
    """텐서를 cpu fp32 numpy bytes로 변환 후 SHA-256."""
    arr = t.detach().cpu().to(torch.float32).numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def register_qkv_hooks(model, store: dict) -> list:
    """각 layer의 q_proj/k_proj/v_proj에 forward hook 등록.

    hook은 모듈 출력(projection 결과 — RoPE 적용 전)을 즉시 CPU(fp32)로 옮겨 store에 저장.
    store key: (layer_idx, proj_name). 반환: hook handle 리스트.
    """
    handles = []
    # model.model.layers: list[MistralDecoderLayer]
    for layer_idx, layer in enumerate(model.model.layers):
        for proj_name in QKV_PROJ_NAMES:
            module = getattr(layer.self_attn, proj_name)

            def make_hook(li: int, pn: str):
                def hook(_mod, _inp, output):
                    # output: (B, T, H_*·D) — q_proj는 H_q·D, k/v_proj는 H_kv·D
                    store[(li, pn)] = output.detach().to("cpu", torch.float32)
                return hook

            handles.append(module.register_forward_hook(make_hook(layer_idx, proj_name)))
    return handles


def run_one(label: str, load_fn, ids) -> dict:
    """모델 로드 → hook 등록 → forward → logits/hidden_states/qkv를 CPU 캡처 → 모델 해제.

    반환: {"logits": Tensor(cpu), "hidden_states": list[Tensor(cpu)],
           "qkv": dict[(int,str) -> Tensor(cpu)], "n_layers": int}
    """
    print(f"[{label}] 모델 로드")
    model = load_fn()
    qkv_store: dict = {}
    handles = register_qkv_hooks(model, qkv_store)

    print(f"[{label}] forward (output_hidden_states=True, use_cache=False)")
    set_all_seeds(SEED)
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True, use_cache=False)

    result = {
        # logits: (B, T, vocab)
        "logits": out.logits.detach().to("cpu", torch.float32),
        # hidden_states: tuple 길이 = num_hidden_layers + 1 (embedding 출력 포함), 각 (B, T, H)
        "hidden_states": [h.detach().to("cpu", torch.float32) for h in out.hidden_states],
        "qkv": qkv_store,  # hook이 이미 CPU로 옮김
        "n_layers": len(model.model.layers),
    }
    for h in handles:
        h.remove()
    del model, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="결과 출력 디렉토리 (예: results/step_01/vastai)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        print("[ERROR] CUBLAS_WORKSPACE_CONFIG 미설정. 'export CUBLAS_WORKSPACE_CONFIG=:4096:8' 후 재실행.")
        sys.exit(1)

    setup_deterministic()
    token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
    # ids: input_ids (1, T) — Step 0와 동일 prompt
    ids = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    def load_hf():
        return AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32,
            attn_implementation="eager", token=token,
        ).to("cuda").eval()

    def load_our():
        return MistralForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32,
            attn_implementation="eager", token=token,
        ).to("cuda").eval()

    # 순차 로드 — HF 먼저 (CPU 캡처 후 해제), 그 다음 our fork
    hf = run_one("HF 표준", load_hf, ids)
    our = run_one("our fork", load_our, ids)

    # --- invariant 1.1: forward logits 동등성 ---
    hf_logits_sha = tensor_sha256(hf["logits"])
    our_logits_sha = tensor_sha256(our["logits"])
    inv_1_1 = hf_logits_sha == our_logits_sha

    # --- invariant 1.2: layer-by-layer hidden state 동등성 ---
    mismatched_layers: list = []
    if len(hf["hidden_states"]) != len(our["hidden_states"]):
        mismatched_layers.append("LENGTH_MISMATCH")
    for i in range(min(len(hf["hidden_states"]), len(our["hidden_states"]))):
        if tensor_sha256(hf["hidden_states"][i]) != tensor_sha256(our["hidden_states"][i]):
            mismatched_layers.append(i)
    inv_1_2 = len(mismatched_layers) == 0

    # --- invariant 1.3: q/k/v projection 출력 element-wise 동등성 ---
    mismatched_qkv: list = []
    for key in sorted(set(hf["qkv"].keys()) | set(our["qkv"].keys())):
        if key not in hf["qkv"] or key not in our["qkv"]:
            mismatched_qkv.append([list(key), "KEY_MISSING"])
        elif not torch.equal(hf["qkv"][key], our["qkv"][key]):
            mismatched_qkv.append([list(key), "NOT_EQUAL"])
    inv_1_3 = len(mismatched_qkv) == 0

    all_passed = inv_1_1 and inv_1_2 and inv_1_3

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "step": 1,
        "env_tag": os.environ.get("COMPBLEND_ENV_TAG", "unknown"),
        "model": args.model,
        "torch_dtype": "float32",
        "attention_implementation": "eager",
        "transformers_version": transformers.__version__,
        "fork_source": "transformers/models/mistral/modeling_mistral.py @ 4.51.3 (import 문 외 byte 무수정)",
        "prompt": PROMPT,
        "invariants": {
            "1.1_forward_logits_equiv": {
                "passed": inv_1_1,
                "our_logits_sha256": our_logits_sha,
                "hf_logits_sha256": hf_logits_sha,
            },
            "1.2_per_layer_hidden_equiv": {
                "passed": inv_1_2,
                "n_hidden_states": len(hf["hidden_states"]),
                "mismatched_layers": mismatched_layers,
            },
            "1.3_qkv_projection_equiv": {
                "passed": inv_1_3,
                "n_layers": hf["n_layers"],
                "mismatched": mismatched_qkv,
            },
        },
        "all_invariants_passed": all_passed,
    }
    out_file = out_dir / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  저장: {out_file}")

    print()
    print(f"  invariant 1.1 (logits):       {'✅' if inv_1_1 else '❌'}")
    print(f"  invariant 1.2 (hidden state): {'✅' if inv_1_2 else '❌'}")
    print(f"  invariant 1.3 (q/k/v proj):   {'✅' if inv_1_3 else '❌'}")
    if all_passed:
        print("==> 모든 invariant 통과 ✅")
    else:
        print("==> Invariant 실패 ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
