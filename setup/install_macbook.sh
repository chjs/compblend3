#!/usr/bin/env bash
# compblend3 MacBook M2 환경 자동 설치 (Claude Code 실행 환경)
# 사용법: bash setup/install_macbook.sh
#
# 가정: macOS, Apple Silicon (M1/M2/M3), GPU 의존 코드는 ssh vast로 전송
# 결과: .venv (Python 3.10), CPU PyTorch, transformers, 본 프로젝트 editable

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

echo "==> compblend3 MacBook 환경 설치 시작"
echo "    PROJECT_DIR=$PROJECT_DIR"

# 1. 시스템 정보 출력
echo ""
echo "[1/7] 시스템 정보 확인"
echo "  OS: $(sw_vers -productName) $(sw_vers -productVersion)"
echo "  Arch: $(uname -m)"
if [[ "$(uname -m)" != "arm64" ]]; then
    echo "  경고: Apple Silicon이 아닌 것 같습니다 (uname -m = $(uname -m))"
fi

# 2. uv 설치
echo ""
echo "[2/7] uv 설치"
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi
uv --version

# 3. Python 3.10 다운로드 및 venv 생성
echo ""
echo "[3/7] Python 3.10 venv 생성"
if [ ! -d ".venv" ]; then
    uv venv --python 3.10 .venv
else
    echo "  .venv 이미 존재 — 스킵"
fi

# 4. 활성화
echo ""
echo "[4/7] venv 활성화"
# shellcheck disable=SC1091
source .venv/bin/activate
echo "  python: $(which python)"
echo "  version: $(python --version)"

# 5. PyTorch CPU 빌드 설치 (macOS Apple Silicon)
echo ""
echo "[5/7] PyTorch CPU 빌드 설치 (MacBook은 실험 ❌, smoke test만)"
# macOS arm64는 wheel 자동 선택 (MPS 백엔드 포함되어 오지만 우리는 안 씀)
uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# 6. 본 프로젝트 + transformers 설치
echo ""
echo "[6/7] transformers + 본 프로젝트 editable 설치"
uv pip install "transformers>=4.51,<4.52"
uv pip install -e ".[test]"

# 7. .env 템플릿 처리
echo ""
echo "[7/7] .env 템플릿 처리"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "COMPBLEND_ENV_TAG=macbook" >> .env
    echo "  .env 생성됨. 시크릿(HF_TOKEN 등)은 MacBook의 .env에는 안 넣어도 됩니다."
    echo "  (MacBook은 실험 안 함. 모델 다운로드는 vast.ai에서)"
else
    echo "  .env 이미 존재 — 스킵"
fi

# 결과 검증
echo ""
echo "==> MacBook 환경 설치 완료"
echo ""
echo "다음 단계:"
echo "  1. ~/.ssh/config에 'Host vast' alias 설정 (사용자 직접)"
echo "  2. ssh vast 'echo hello' 로 alias 확인"
echo "  3. source .venv/bin/activate"
echo "  4. python scripts/check_env.py  (macbook tag로 부분 OK 기대)"
echo "  5. ssh vast로 Phase 0-B 진행"
