#!/usr/bin/env bash
# compblend3 vast.ai 인스턴스 자동 설치
# 사용법: bash setup/install_vastai.sh
#
# 가정: A100-SXM4 80GB, Ubuntu 22.04 또는 24.04, CUDA driver ≥ 525
# 결과: .venv (Python 3.10), torch 2.10.0+cu128, 본 프로젝트 editable 설치

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

echo "==> compblend3 vast.ai 환경 설치 시작"
echo "    PROJECT_DIR=$PROJECT_DIR"

# 1. 시스템 정보 출력 + 저장 (검증용)
echo ""
echo "[1/9] 시스템 정보 확인"
ENV_INFO_DIR="results/phase_00/vastai"
mkdir -p "$ENV_INFO_DIR"
ENV_INFO_FILE="$ENV_INFO_DIR/env_info.txt"
{
    echo "# compblend3 vast.ai 인스턴스 환경 정보"
    echo "Date: $(date -Iseconds)"
    echo ""
    echo "## OS"
    lsb_release -a 2>/dev/null || cat /etc/os-release
    echo ""
    echo "## GPU"
    nvidia-smi
    echo ""
    echo "## nvcc"
    if command -v nvcc &> /dev/null; then
        nvcc --version
    else
        echo "nvcc not found (시스템 CUDA toolkit 없음, PyTorch wheel만으로 충분)"
    fi
    echo ""
    echo "## Python"
    python3 --version || true
} | tee "$ENV_INFO_FILE"

echo ""
echo "  환경 정보 저장됨: $ENV_INFO_FILE"

# 2. 필수 시스템 패키지
echo ""
echo "[2/9] 시스템 패키지 설치"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    git curl wget build-essential \
    python3-pip python3-dev \
    tmux htop \
    > /dev/null

# 3. uv 설치
echo ""
echo "[3/9] uv 설치"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    # 다음 셸에서도 PATH 유지
    if ! grep -q ".local/bin" ~/.bashrc; then
        echo 'export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
fi
uv --version

# 4. Python 3.10 다운로드 및 venv 생성
echo ""
echo "[4/9] Python 3.10 venv 생성"
if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
else
    echo "  .venv 이미 존재 — 스킵"
fi

# 5. 활성화
echo ""
echo "[5/9] venv 활성화"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "  python: $(which python)"
echo "  version: $(python --version)"

# 6. PyTorch 2.10.0+cu128 설치
echo ""
echo "[6/9] PyTorch 2.10.0+cu128 설치"
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 7. transformers 설치
echo ""
echo "[7/9] transformers 설치 (Phase 0 task 0.1에서 최종 버전 확정)"
uv pip install "transformers>=4.51,<4.52"

# 8. 본 프로젝트 editable 설치
echo ""
echo "[8/9] compblend3 editable 설치"
uv pip install -e ".[test]"

# 9. .env 템플릿 처리
echo ""
echo "[9/9] .env 템플릿 처리"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "COMPBLEND_ENV_TAG=vastai" >> .env
    echo "  .env 생성됨. HF_TOKEN을 반드시 채워주세요."
else
    echo "  .env 이미 존재 — 스킵"
fi

# 결과 검증
echo ""
echo "==> 설치 완료"
echo ""
echo "다음 단계:"
echo "  1. .env 편집 (HF_TOKEN 채우기): nano .env"
echo "  2. source .venv/bin/activate"
echo "  3. python scripts/check_env.py"
echo "  4. python scripts/download_models.py --model mistralai/Mistral-7B-Instruct-v0.2"
