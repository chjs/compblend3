#!/usr/bin/env python3
"""Loong 데이터셋을 필터링하여 manifest를 생성한다.

Loong은 평균 11개 문서/instance, 길이 10K~200K+이지만, 우리는
Mistral-7B-v0.2의 32K context window를 고려해 작은 instance만 선별.

사용법:
    python scripts/build_loong_manifest.py \
        --level 1 \
        --language english \
        --domain academic \
        --max-tokens 16000 \
        --output data/manifests/loong_level1_eng_academic_16k.json

Level:
    1 = Spotlight Locating  (가장 쉬움)
    2 = Comparison
    3 = Clustering
    4 = Chain of Reasoning  (가장 어려움)

출력 manifest 형식:
    {
        "source": "Loong",
        "filters": {...},
        "tokenizer": "mistralai/Mistral-7B-Instruct-v0.2",
        "n_instances": N,
        "instances": [
            {
                "id": "...",
                "level": 1,
                "language": "english",
                "domain": "academic",
                "n_docs": 8,
                "doc_lengths": [1234, ...],   # tokens
                "total_input_tokens": 12000,  # docs + question
                "question": "...",
                "answer": "...",
                "documents": [...],
                "metadata": {...}
            },
            ...
        ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_dotenv():
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
    ap.add_argument("--level", type=int, choices=[1, 2, 3, 4], required=True,
                    help="Loong level (1=Spotlight, 2=Comparison, 3=Clustering, 4=Chain)")
    ap.add_argument("--language", choices=["english", "chinese"], default="english")
    ap.add_argument("--domain", choices=["academic", "financial", "legal"], default="academic")
    ap.add_argument("--max-tokens", type=int, default=16000,
                    help="필터링: total input tokens 한도")
    ap.add_argument("--tokenizer", default="mistralai/Mistral-7B-Instruct-v0.2",
                    help="토큰 카운트에 사용할 tokenizer")
    ap.add_argument("--loong-dir", default=None,
                    help="Loong 저장소 경로 (default: $LOONG_DATA_DIR 또는 ~/Loong)")
    ap.add_argument("--output", required=True, help="출력 manifest 경로")
    ap.add_argument("--limit", type=int, default=None,
                    help="최대 instance 수 (디버깅용)")
    args = ap.parse_args()

    load_dotenv()

    loong_dir = args.loong_dir or os.environ.get("LOONG_DATA_DIR") or os.path.expanduser("~/Loong")
    loong_path = Path(loong_dir)
    if not loong_path.exists():
        print(f"[ERROR] Loong dir not found: {loong_path}")
        print("        git clone https://github.com/MozerWang/Loong 후")
        print("        .env에 LOONG_DATA_DIR=... 추가하거나 --loong-dir 지정")
        sys.exit(1)

    # data/loong.jsonl 로드
    jsonl_path = loong_path / "data" / "loong.jsonl"
    if not jsonl_path.exists():
        print(f"[ERROR] {jsonl_path} not found")
        sys.exit(1)

    # Tokenizer 로드 (token 길이 측정용)
    print(f"[1/3] Tokenizer 로드: {args.tokenizer}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, token=os.environ.get("HF_TOKEN"))

    # 데이터 로드 및 필터
    print(f"[2/3] Loong 데이터 로드 및 필터")
    print(f"      level={args.level}, language={args.language}, domain={args.domain}")
    print(f"      max_tokens={args.max_tokens}")

    level_key = f"level{args.level}"
    instances = []
    skipped = {"level": 0, "language": 0, "domain": 0, "max_tokens": 0}
    total = 0

    with jsonl_path.open() as f:
        for line in f:
            total += 1
            d = json.loads(line)
            # Loong jsonl 형식은 정확히 모름 — 실제 사용 시 fields 검증 필요.
            # 여기서는 일반적인 필드를 가정. 실제 구조에 맞춰 조정 필요.
            if d.get("level") != level_key:
                skipped["level"] += 1
                continue
            if d.get("language", "english") != args.language:
                skipped["language"] += 1
                continue
            if d.get("domain", "academic") != args.domain:
                skipped["domain"] += 1
                continue

            # Token count
            docs = d.get("documents") or d.get("doc") or []
            question = d.get("question", "")
            answer = d.get("answer", "")

            # 가장 안전한 방식: 전체 텍스트 concat 후 tokenize
            full_text = "\n\n".join(str(doc) for doc in docs) + "\n\n" + question
            n_tokens = len(tokenizer.encode(full_text, add_special_tokens=True))

            if n_tokens > args.max_tokens:
                skipped["max_tokens"] += 1
                continue

            doc_lengths = [len(tokenizer.encode(str(doc), add_special_tokens=False)) for doc in docs]

            instances.append({
                "id": d.get("id") or f"loong_{level_key}_{len(instances)}",
                "level": args.level,
                "language": args.language,
                "domain": args.domain,
                "n_docs": len(docs),
                "doc_lengths": doc_lengths,
                "total_input_tokens": n_tokens,
                "question": question,
                "answer": answer,
                "documents": docs,
                "metadata": {k: v for k, v in d.items()
                             if k not in ("documents", "doc", "question", "answer")},
            })

            if args.limit and len(instances) >= args.limit:
                break

    # 저장
    print(f"[3/3] Manifest 저장")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": "Loong",
        "filters": {
            "level": args.level,
            "language": args.language,
            "domain": args.domain,
            "max_tokens": args.max_tokens,
        },
        "tokenizer": args.tokenizer,
        "n_instances": len(instances),
        "stats": {
            "total_in_loong": total,
            "skipped": skipped,
            "kept": len(instances),
            "token_distribution": _token_stats(instances),
            "doc_count_distribution": _doc_count_stats(instances),
        },
        "instances": instances,
    }

    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"    저장됨: {out_path}")
    print(f"    {len(instances)} / {total} instances kept")
    print(f"    skipped: {skipped}")


def _token_stats(instances: list[dict]) -> dict:
    if not instances:
        return {}
    tokens = [i["total_input_tokens"] for i in instances]
    return {
        "min": min(tokens),
        "max": max(tokens),
        "mean": sum(tokens) / len(tokens),
        "median": sorted(tokens)[len(tokens) // 2],
    }


def _doc_count_stats(instances: list[dict]) -> dict:
    if not instances:
        return {}
    counts = [i["n_docs"] for i in instances]
    return {
        "min": min(counts),
        "max": max(counts),
        "mean": sum(counts) / len(counts),
        "median": sorted(counts)[len(counts) // 2],
    }


if __name__ == "__main__":
    main()
