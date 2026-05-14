# 트러블슈팅 가이드

자주 발생하는 환경 문제와 해결책.

---

## 1. PyTorch 설치 관련

### 증상: `pip install torch==2.10.0 --index-url .../cu128` 이 실패

가능 원인:
- Python 버전이 3.10이 아님 → `python --version` 확인
- 네트워크 이슈 → `--index-url` 직접 wget 가능한지 확인
- uv 캐시 충돌 → `uv cache clean` 후 재시도

### 증상: `torch.cuda.is_available()` 이 False

가능 원인:
- Driver와 PyTorch CUDA 버전 mismatch
  - PyTorch CUDA 12.8 → Driver ≥ 525 필요 (575는 OK)
- `LD_LIBRARY_PATH`에 시스템 CUDA가 끼어들어 충돌
  - 해결: venv 활성화 후 `unset LD_LIBRARY_PATH` 시도

### 증상: `torch.use_deterministic_algorithms(True)` 후 일부 op에서 RuntimeError

PyTorch 2.10에서 deterministic 알고리즘이 일부 op에 없음.
- 우회: `CUBLAS_WORKSPACE_CONFIG=:4096:8` 환경변수 필수
- 그래도 실패하면: 해당 op만 `torch.use_deterministic_algorithms(True, warn_only=True)` 로 완화

---

## 2. transformers / 모델 로딩

### 증상: `OSError: ... mistralai/Mistral-7B-Instruct-v0.2 is gated`

원인: HF_TOKEN이 설정 안 됐거나, 해당 모델에 대한 access 승인 안 받음.

해결:
1. https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 방문
2. "Access this model" 클릭 → 동의
3. https://huggingface.co/settings/tokens 에서 token 생성
4. `.env`에 `HF_TOKEN=hf_...` 추가

### 증상: 모델 로딩이 매우 느림

- 첫 다운로드: 정상 (~15GB)
- 두 번째부터: `HF_HOME` 캐시 활용. `.env`에 `HF_HOME=/path/to/cache` 명시.

### 증상: `attention_implementation="eager"` 인데 flash-attn 경고가 뜸

무시해도 OK. eager 사용 중이면 flash-attn은 안 쓰임. 경고가 시끄러우면:
```python
import warnings
warnings.filterwarnings("ignore", message=".*flash.*")
```

---

## 3. 결정론 / 재현성

### 증상: 같은 코드 같은 입력인데 출력이 매번 다름

체크리스트:
- [ ] `torch.manual_seed(SEED)` 호출했나
- [ ] `torch.cuda.manual_seed_all(SEED)` 호출했나
- [ ] `CUBLAS_WORKSPACE_CONFIG=:4096:8` 설정했나
- [ ] `torch.use_deterministic_algorithms(True)` 호출했나
- [ ] `torch.backends.cudnn.deterministic = True`?
- [ ] `torch.backends.cudnn.benchmark = False`?
- [ ] `PYTHONHASHSEED=0` 환경변수?
- [ ] DataLoader면 `worker_init_fn` 으로 worker별 seed 설정했나

모두 OK인데도 다르면: dtype이 fp16/bf16일 가능성. fp32로 시도.

### 증상: 같은 환경끼리 결정론 보장, 환경 다르면 안 됨

이건 정상. CUDA driver, GPU SKU, kernel implementation 버전이 다르면 numerical 차이.
- 해결책: SHA-256 일치는 못 보장, atol 1e-5로 fallback

---

## 4. GPU 메모리

### 증상: A100 80GB인데 OOM

가능 원인:
- 다른 프로세스가 GPU 점유 → `nvidia-smi` 로 확인, `kill -9 <pid>`
- fp32로 7B 모델 로딩 시: 약 28GB. + activation, gradient 등이 추가됨.
- KV cache가 크면 추가 메모리 필요
- 해결: fp32 → bf16, batch_size=1, gradient_checkpointing 등

### 증상: OOM이 일관적으로 발생하는 input length

- Loong manifest 생성 시 `--max-tokens 16000` 이하로 필터링
- Step 8에서 Mistral 32K context는 위험. ~24K 이하로 안전 마진

---

## 5. uv 관련

### 증상: `uv venv --python 3.10` 이 실패

원인: Python 3.10이 시스템에 없고 uv가 다운로드도 못 함.

해결:
```bash
# uv가 자체적으로 Python 3.10 다운로드
uv python install 3.10
uv venv --python 3.10 .venv
```

### 증상: `uv pip install -e .` 이 실패

원인: `pyproject.toml` 의 dependency 충돌, 또는 setuptools 너무 오래됨.

해결:
```bash
uv pip install --upgrade setuptools wheel pip
uv pip install -e ".[test]" --no-build-isolation
```

---

## 6. Git / 결과 동기화

### 증상: `git push` 가 너무 큰 파일로 실패

원인: `.gitignore` 에 누락된 큰 파일이 commit됨.
- `.pt`, `.safetensors`, `*.npy`, `full_logits/`, `tensor_dumps/` 같은 것

해결:
```bash
# 어떤 큰 파일이 있는지 확인
git ls-files | xargs du -h | sort -h | tail -20

# 큰 파일을 .gitignore 추가, history에서 제거
git rm --cached <large_file>
git commit --amend
```

### 증상: vast.ai ↔ 로컬 결과가 너무 많이 다름

- atol 1e-5 도 실패: 환경 차이가 큼. SHA-256은 어차피 안 나옴. atol 1e-3까지 완화 후 사유 명시.
- 특정 layer/position에서 갈리는 패턴: 보고서에 명시. 같은 step의 invariant 통과한다면 진행 가능.
- 모든 step에서 일관되게 다르면: cuDNN 버전 확인. `python -c "import torch; print(torch.backends.cudnn.version())"`

---

## 7. flash-attn (Phase 6 이후)

### 증상: `flash-attn` 빌드 실패

- 사전 빌드된 wheel 사용 시도:
  ```bash
  pip install flash-attn==2.7.4 --no-build-isolation
  ```
- 직접 빌드해야 한다면 nvcc 12.8 + torch 2.10 호환 wheel 필요
- 최후의 수단: flash-attn 없이 sdpa backend로 진행 (PyTorch 내장 효율 attention)

---

## 8. 마지막 수단

위 모든 게 안 되면:
1. `pip freeze > frozen.txt` 로 현재 패키지 상태 캡처
2. `nvidia-smi`, `nvcc --version`, `python --version`, `uv --version` 결과 캡처
3. 정확한 에러 메시지 전체
4. 이 정보를 `docs/reports/` 아래 새 파일로 기록 후 사용자에게 보고
