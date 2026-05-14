# 로컬 A100 환경 셋업 가이드 (occasional 검증 환경)

대상: 사용자의 별도 로컬 A100-SXM4 80GB 머신 (Ubuntu 24.04, nvcc 12.8).
사용자가 "꼭 필요한 검증"이라 판단한 step에서만 별도로 실험을 재현하는 용도.

---

## 1. 로컬 A100의 역할

- **occasional 검증 환경**: 사용자가 결정한 step의 재현
- **결과 저장 위치**: `results/step_XX/local_a100/summary.json`
- **사용자가 직접 운영**: Claude가 ssh로 자동 트리거 ❌ (vast.ai와 다른 점)
- 대부분 step에서는 빈 상태로 둠

### 검증 권장 step

- **Step 0** (결정론 확인) — vast.ai 결정론과 로컬 결정론 모두 보장 확인
- **Step 8** (F1 측정) — 결과 reproducibility 핵심
- 그 외 step은 사용자 판단

---

## 2. 환경 사양 확인

```bash
lsb_release -a   # 기대: Ubuntu 24.04 LTS
nvidia-smi       # 기대: A100-SXM4-80GB, Driver 575.x, CUDA 12.9
nvcc --version   # 기대: 12.8
python3 --version
```

vast.ai와 환경 일치가 SHA-256 검증 가능성을 높임. 사양이 많이 다르면 atol 1e-5 fallback.

---

## 3. uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv --version
```

---

## 4. 저장소 클론

```bash
mkdir -p ~/work
cd ~/work
git clone https://github.com/chjs/compblend3.git
cd compblend3
```

---

## 5. 자동 셋업

```bash
bash setup/install_local.sh
```

스크립트가 하는 것:
- `.venv` (Python 3.10)
- PyTorch 2.10.0+cu128
- transformers
- 본 프로젝트 editable
- `.env` 템플릿 + `COMPBLEND_ENV_TAG=local_a100` 자동

---

## 6. .env에 HF_TOKEN

```bash
cp .env.example .env
nano .env
# HF_TOKEN=hf_...
# COMPBLEND_ENV_TAG=local_a100 (자동 추가됨)
```

---

## 7. 환경 검증

```bash
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/check_env.py
```

기대: local_a100 tag로 전부 OK. 결과는 `results/phase_00/local_a100/env_check.json`.

---

## 8. 모델 다운로드 (필요 시)

```bash
python scripts/download_models.py --model mistralai/Mistral-7B-Instruct-v0.2
```

vast.ai와 별도 다운로드. 약 15GB.

---

## 9. Step 검증 워크플로우 (사용자 직접 진행)

vast.ai에서 step XX가 완료되어 git push까지 되면 사용자가 로컬 A100에서:

```bash
# 1. 최신 코드 + 결과 받기
cd ~/work/compblend3
git pull

# 2. 같은 실험 재현
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python tasks/step_XX/run_*.py --out results/step_XX/local_a100/

# 3. 비교
python scripts/compare_results.py --step XX
# Tier 1 (SHA-256) 또는 Tier 2 (atol) 통과 확인

# 4. 결과 push
git add results/step_XX/local_a100/
git commit -m "[step_XX] verification on local_a100"
git push
```

---

## 10. 비교 결과 해석

```bash
python scripts/compare_results.py --step XX --left vastai --right local_a100
```

- **Tier 1 PASS** (SHA-256 일치): 두 환경 결정론 완전 동등. 이상적.
- **Tier 2 PASS** (atol 1e-5): 수치 동등. 정상.
- **Tier 3 PASS** (token sequence): F1 같은 task metric만 같음.
- **모두 FAIL**: 환경 차이가 큼. 보고서에 기록 후 사용자와 검토.

---

## 11. 결과가 vast.ai와 다를 때

흔한 원인:
- cuDNN 버전 차이
- 다른 background process가 GPU 사용 중
- `CUBLAS_WORKSPACE_CONFIG` 미설정
- 다른 driver 버전

`scripts/compare_results.py` 가 어느 layer에서 갈렸는지 리포트해줌. 보고서에 기록.

---

## 12. 로컬 A100을 안 쓰는 step

대부분 step은 vast.ai 단독으로 진행. 로컬 A100 검증 안 해도 OK. 그 경우 `results/step_XX/local_a100/` 디렉토리는 비어 있음. `scripts/compare_results.py` 가 vastai 단독 모드로 통과 판정.

---

## 13. 트러블슈팅

자세한 내용: `docs/setup/troubleshooting.md`

자주 발생:
- vast.ai와 SHA-256이 안 맞음 → atol 1e-5 fallback. 정상적 범위 안이면 OK.
- OOM: 다른 프로세스 점유 확인. `nvidia-smi` 로 점유 PID 찾아 kill.
- `CUBLAS_WORKSPACE_CONFIG` 미설정 에러: deterministic 모드 켜기 전 환경변수 설정 필수.
