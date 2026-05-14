#!/usr/bin/env python3
"""환경 검증 스크립트 (3환경 분기: macbook / vastai / local_a100).

실행: python scripts/check_env.py

결과: results/phase_00/{env}/env_check.json
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def get_env_tag() -> str:
    """환경 태그: COMPBLEND_ENV_TAG 또는 자동 감지."""
    tag = os.environ.get("COMPBLEND_ENV_TAG", "")
    if tag in ("macbook", "vastai", "local_a100"):
        return tag
    # 자동 감지
    system = platform.system()
    machine = platform.machine()
    if system == "Darwin":
        return "macbook"
    hostname = platform.node().lower()
    if any(k in hostname for k in ("vast", "runpod", "lambda")):
        return "vastai"
    if system == "Linux" and machine == "x86_64":
        return "local_a100"
    return "unknown"


def check(name: str, value: Any, status: str, expected: str = "") -> dict:
    """체크 결과 한 줄. status는 'OK' / 'FAIL' / 'INFO'."""
    msg = f"[{status}] {name}: {value}"
    if expected:
        msg += f" (expected: {expected})"
    print(msg)
    return {"name": name, "value": str(value), "status": status, "expected": expected}


def check_python() -> dict:
    v = sys.version_info
    val = f"{v.major}.{v.minor}.{v.micro}"
    return check("Python", val, "OK" if v.major == 3 and v.minor == 10 else "FAIL", "3.10.x")


def check_pytorch(env_tag: str) -> dict:
    try:
        import torch
        val = torch.__version__
        if env_tag == "macbook":
            # CPU 빌드 OK, cu128 wheel 아니어도 OK
            ok = val.startswith("2.10.")
            return check("PyTorch", val, "OK" if ok else "FAIL", "2.10.x (CPU OK)")
        else:
            ok = val.startswith("2.10.0") and "cu128" in val
            return check("PyTorch", val, "OK" if ok else "FAIL", "2.10.0+cu128")
    except ImportError as e:
        return check("PyTorch", f"ImportError: {e}", "FAIL", "2.10.x")


def check_cuda(env_tag: str) -> dict:
    try:
        import torch
        val = torch.version.cuda
        if env_tag == "macbook":
            return check("CUDA (from torch)", val or "None", "INFO", "not required on macbook")
        ok = val is not None and val.startswith("12.")
        return check("CUDA (from torch)", val, "OK" if ok else "FAIL", "12.x")
    except ImportError:
        return check("CUDA (from torch)", "torch not installed", "FAIL", "12.x")


def check_gpu(env_tag: str) -> dict:
    try:
        import torch
        avail = torch.cuda.is_available()
        if env_tag == "macbook":
            return check("GPU", "not available (CPU mode)", "INFO", "not required on macbook")
        if not avail:
            return check("GPU", "CUDA not available", "FAIL", "A100 80GB")
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / 1024**3
        val = f"{name} ({mem_gb:.0f}GB)"
        ok = "A100" in name and mem_gb >= 75
        return check("GPU", val, "OK" if ok else "FAIL", "A100 80GB")
    except Exception as e:
        return check("GPU", f"Error: {e}", "FAIL", "A100 80GB")


def check_deterministic(env_tag: str) -> dict:
    try:
        import torch
        if env_tag == "macbook":
            return check("Deterministic mode", "not testable without GPU", "INFO",
                         "not required on macbook")
        cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        if cublas_config not in (":4096:8", ":16:8"):
            return check("Deterministic mode",
                         f"CUBLAS_WORKSPACE_CONFIG not set ({cublas_config!r})",
                         "FAIL", "CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.use_deterministic_algorithms(True)
        x = torch.randn(8, 8, device="cuda")
        _ = x @ x.T
        torch.cuda.synchronize()
        torch.use_deterministic_algorithms(False)
        return check("Deterministic mode", "available", "OK")
    except Exception as e:
        return check("Deterministic mode", f"Error: {e}", "FAIL")


def check_transformers() -> dict:
    try:
        import transformers
        val = transformers.__version__
        major, minor, *_ = val.split(".")
        ok = int(major) == 4 and int(minor) >= 50
        return check("Transformers", val, "OK" if ok else "FAIL", ">=4.50")
    except ImportError as e:
        return check("Transformers", f"ImportError: {e}", "FAIL")


def check_hf_token(env_tag: str) -> dict:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_tag == "macbook":
        return check("HF_TOKEN", "set" if token else "not set",
                     "INFO", "optional on macbook (실험은 vast.ai에서)")
    ok = bool(token)
    val = "set" if ok else "not set"
    return check("HF_TOKEN", val, "OK" if ok else "FAIL")


def check_env_tag() -> dict:
    tag = get_env_tag()
    ok = tag in ("macbook", "vastai", "local_a100")
    return check("Environment tag", tag, "OK" if ok else "FAIL")


def check_nvidia_smi(env_tag: str) -> dict:
    if env_tag == "macbook":
        return check("nvidia-smi", "not available (macOS)", "INFO", "not required on macbook")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return check("nvidia-smi", "failed", "FAIL")
        line = result.stdout.strip().split("\n")[0]
        return check("nvidia-smi", line, "OK")
    except Exception as e:
        return check("nvidia-smi", f"Error: {e}", "FAIL")


def main():
    # .env 로드
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() and v.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

    env_tag = get_env_tag()

    print(f"==> compblend3 환경 검증 (tag: {env_tag})")
    print()

    checks = [
        check_python(),
        check_pytorch(env_tag),
        check_cuda(env_tag),
        check_gpu(env_tag),
        check_deterministic(env_tag),
        check_transformers(),
        check_hf_token(env_tag),
        check_env_tag(),
        check_nvidia_smi(env_tag),
    ]

    # FAIL이 하나라도 있으면 전체 FAIL. INFO는 OK 취급.
    all_ok = not any(c["status"] == "FAIL" for c in checks)

    print()
    if all_ok:
        print(f"==> [{env_tag}] 모든 필수 검증 통과 ✅")
    else:
        print(f"==> [{env_tag}] 일부 검증 실패 ❌")
        print(f"    docs/setup/troubleshooting.md 참조")
        if env_tag == "macbook":
            print(f"    또는 docs/setup/macbook_setup.md")

    # 결과 저장
    out_dir = Path("results/phase_00") / env_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "env_check.json"

    out_data = {
        "env_tag": env_tag,
        "all_ok": all_ok,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "checks": checks,
        "timestamp": subprocess.run(
            ["date", "-Iseconds"], capture_output=True, text=True
        ).stdout.strip(),
    }

    out_file.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))
    print(f"\n결과 저장: {out_file}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
