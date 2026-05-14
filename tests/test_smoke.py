"""Smoke test — 패키지가 import되고 기본 환경이 OK한지 확인.

3머신 환경에서 모두 통과해야 함:
- macbook: CPU PyTorch, GPU 없음 (cu128 검증 스킵)
- vastai / local_a100: PyTorch 2.10.0+cu128
"""
from __future__ import annotations

import platform


def is_macbook() -> bool:
    return platform.system() == "Darwin"


def test_compblend_imports():
    """compblend 패키지가 import 가능해야 한다."""
    import compblend  # noqa: F401
    assert hasattr(compblend, "__version__")


def test_python_version():
    """Python 3.10 환경인지 확인."""
    import sys
    assert sys.version_info[:2] == (3, 10), (
        f"Python 3.10 필요, 현재 {sys.version_info[:2]}. "
        f"DECISIONS.md §3 참조."
    )


def test_pytorch_version():
    """PyTorch 2.10.x. CUDA 빌드 여부는 환경별 분기."""
    import torch
    assert torch.__version__.startswith("2.10."), (
        f"PyTorch 2.10.x 필요, 현재 {torch.__version__}. "
        f"DECISIONS.md §3 참조."
    )
    # MacBook은 CPU 빌드 OK. vast.ai/local_a100은 cu128 필수.
    if not is_macbook():
        assert "cu128" in torch.__version__, (
            f"vast.ai/local_a100에서는 cu128 wheel 필수, 현재 {torch.__version__}. "
            f"DECISIONS.md §3 참조."
        )


def test_transformers_available():
    """transformers import 가능해야 한다."""
    import transformers  # noqa: F401


def test_cuda_available_when_not_macbook():
    """MacBook이 아니면 CUDA 사용 가능해야 한다."""
    if is_macbook():
        # MacBook에서는 GPU 없음. 정상.
        return
    import torch
    assert torch.cuda.is_available(), (
        "vast.ai/local_a100에서는 CUDA 필요. nvidia-smi 확인."
    )
