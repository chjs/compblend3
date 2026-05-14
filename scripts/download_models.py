#!/usr/bin/env python3
"""모델 가중치를 HF cache에 미리 다운로드.

사용법:
    python scripts/download_models.py --model mistralai/Mistral-7B-Instruct-v0.2
    python scripts/download_models.py --model meta-llama/Llama-3.1-8B
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def load_dotenv():
    """간이 .env 로더."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and v.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 모델 ID")
    ap.add_argument("--dtype", default="auto", help="원하는 dtype 검증용 (실제 다운로드는 dtype 무관)")
    args = ap.parse_args()

    load_dotenv()

    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("[WARN] HF_TOKEN 미설정. gated 모델은 받을 수 없습니다.")
        print("       .env 파일에 HF_TOKEN=hf_... 를 추가하거나 환경변수로 설정하세요.")

    print(f"==> 다운로드 시작: {args.model}")
    print(f"    HF_HOME: {os.environ.get('HF_HOME', '~/.cache/huggingface (default)')}")
    print()

    try:
        from transformers import AutoTokenizer, AutoConfig
    except ImportError as e:
        print(f"[ERROR] transformers 미설치: {e}")
        sys.exit(1)

    # Config (가장 가볍게 먼저 받아서 access 확인)
    print("[1/3] Config 다운로드...")
    config = AutoConfig.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    print(f"    OK. model_type={config.model_type}, hidden_size={config.hidden_size}")

    # Tokenizer
    print("[2/3] Tokenizer 다운로드...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.environ.get("HF_TOKEN"))
    print(f"    OK. vocab_size={len(tokenizer)}")

    # 가중치 (HF cache에 저장만 하고 로드는 안 함, 메모리 절약)
    print("[3/3] 가중치 다운로드 (시간 소요)...")
    from huggingface_hub import snapshot_download
    snapshot_path = snapshot_download(
        repo_id=args.model,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"    OK. snapshot_path={snapshot_path}")

    # 크기 확인
    total_gb = sum(
        f.stat().st_size for f in Path(snapshot_path).rglob("*") if f.is_file()
    ) / 1024**3
    print()
    print(f"==> 완료. 총 크기: {total_gb:.1f}GB")
    print()
    print("다음: python scripts/check_env.py 로 환경 검증 후, Step 0 진행")


if __name__ == "__main__":
    main()
