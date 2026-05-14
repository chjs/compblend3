#!/usr/bin/env bash
# compblend3 로컬 A100 환경 자동 설치
# 사용법: bash setup/install_local.sh
#
# 가정: Ubuntu 24.04, A100-SXM4 80GB, CUDA driver 575.x, nvcc 12.8
# 결과: .venv (Python 3.10), torch 2.10.0+cu128, 본 프로젝트 editable 설치

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

echo "==> compblend3 로컬 환경 설치 시작"
echo "    PROJECT_DIR=$PROJECT_DIR"

# 1. 시스템 정보 출력 (검증용)
echo ""
echo "[1/8] 시스템 정보 확인"
echo "  OS: $(lsb_release -d | cut -f2)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
if command -v nvcc &> /dev/null; then
    echo "  nvcc: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
fi

# 2. uv 설치
echo ""
echo "[2/8] uv 설치 (이미 있으면 스킵)"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # PATH 갱신
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
uv --version

# 3. Python 3.10 다운로드 및 venv 생성
echo ""
echo "[3/8] Python 3.10 venv 생성"
if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
else
    echo "  .venv 이미 존재 — 스킵"
fi

# 4. 활성화
echo ""
echo "[4/8] venv 활성화"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "  python: $(which python)"
echo "  version: $(python --version)"

# 5. PyTorch 2.10.0+cu128 설치
echo ""
echo "[5/8] PyTorch 2.10.0+cu128 설치"
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 6. transformers 설치
echo ""
echo "[6/8] transformers 설치 (Phase 0 task 0.1에서 최종 버전 확정)"
uv pip install "transformers>=4.51,<4.52"

# 7. 본 프로젝트 editable 설치
echo ""
echo "[7/8] compblend3 editable 설치"
uv pip install -e ".[test]"

# 8. .env 템플릿 복사
echo ""
echo "[8/8] .env 템플릿 처리"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  .env 생성됨. HF_TOKEN 등을 채워주세요."
    # 자동으로 local_a100 태그 추가
    echo "COMPBLEND_ENV_TAG=local_a100" >> .env
else
    echo "  .env 이미 존재 — 스킵"
fi

# 결과 검증
echo ""
echo "==> 설치 완료"
echo ""
echo "다음 단계:"
echo "  1. .env 편집 (HF_TOKEN 채우기)"
echo "  2. source .venv/bin/activate"
echo "  3. python scripts/check_env.py"
