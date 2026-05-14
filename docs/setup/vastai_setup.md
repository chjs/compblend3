# vast.ai 인스턴스 셋업 가이드 (primary 실험 환경)

대상: 모든 step의 실험이 실행되는 vast.ai 인스턴스. Claude Code(MacBook)가 ssh로 접속해서 명령을 트리거한다.

---

## 1. vast.ai의 역할

- **primary 실험 환경**: 모든 step의 forward, 결정론 검증, F1 측정
- **결과 저장 위치**: `results/step_XX/vastai/summary.json`
- **MacBook에서 ssh로 명령 받음**: 사용자 중간 개입 없이 Claude가 직접 트리거

사용자 로컬 A100은 가끔의 검증용. 대부분 step에서는 vast.ai 단독으로 진행.

---

## 2. 인스턴스 선택

### 권장 사양

| 항목 | 권장 값 |
|---|---|
| GPU | **A100-SXM4 80GB × 1** |
| GPU 수 | 1 |
| VRAM | 80GB |
| Host RAM | ≥ 64GB |
| Disk | ≥ 200GB |
| CUDA | 12.4 이상 (driver, 12.8 wheel과 호환) |
| OS | Ubuntu 22.04 또는 24.04 |

### 사양 일치 우선순위 (사용자 로컬 A100과 SHA-256 비교 위해)

1. **A100-SXM4 80GB** ← 강력 권장
2. A100-SXM4 40GB
3. A100-PCIe 80GB
4. H100 (numerical 차이 가능)

### 가격 참고
- A100-SXM4 80GB: ~$1.5/hour
- 작업 빈도에 따라 stop/resume 활용

---

## 3. 인스턴스 생성 후 첫 작업 (사용자)

### 3.1 SSH 정보 확인
vast.ai 콘솔 → 인스턴스 → SSH 정보 (host, port)

### 3.2 MacBook의 `~/.ssh/config` 에 alias 등록
```
Host vast
    HostName <vast-host>
    Port <vast-port>
    User root
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
```

### 3.3 공개키 vast.ai에 등록
vast.ai 콘솔 → Account → SSH Keys → MacBook의 `~/.ssh/id_rsa.pub` 내용 등록.

### 3.4 동작 확인 (MacBook에서)
```bash
ssh vast 'echo hello && nvidia-smi --query-gpu=name --format=csv,noheader'
```
**기대**: 비밀번호 없이 즉시 `hello` + `NVIDIA A100-SXM4-80GB` 출력.

---

## 4. 초기 셋업 (Claude가 ssh로 트리거)

이 시점부터는 Claude Code가 MacBook에서 모든 명령을 자동 트리거.

### 4.1 시스템 정보 캡처
```bash
ssh vast 'lsb_release -a 2>/dev/null; nvidia-smi; nvcc --version 2>/dev/null || echo "no system nvcc (OK)"'
```

### 4.2 저장소 clone
```bash
ssh vast 'cd ~ && git clone https://github.com/chjs/compblend3.git'
```

### 4.3 설치 스크립트 실행
```bash
ssh vast 'cd compblend3 && bash setup/install_vastai.sh'
```

스크립트가 하는 것:
- 시스템 패키지 (git, build-essential, tmux 등)
- uv 설치
- `.venv` (Python 3.10)
- PyTorch 2.10.0+cu128
- transformers
- 본 프로젝트 editable 설치
- `.env` 템플릿 + `COMPBLEND_ENV_TAG=vastai` 자동 추가

### 4.4 .env에 HF_TOKEN 채움 (사용자가 직접)

**중요**: 시크릿이라 Claude가 ssh inline으로 넣지 ❌.

사용자가 vast.ai에 직접 SSH:
```bash
ssh vast
# 인스턴스 안에서
cd compblend3
nano .env
# HF_TOKEN=hf_... 한 줄 추가
exit
```

---

## 5. 환경 검증 (Claude가 ssh로)

```bash
ssh vast 'cd compblend3 && source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python scripts/check_env.py'
```

기대: vastai tag로 전부 OK. 결과는 `results/phase_00/vastai/env_check.json`.

---

## 6. 모델 다운로드 (Claude가 ssh로)

```bash
ssh vast 'cd compblend3 && source .venv/bin/activate && python scripts/download_models.py --model mistralai/Mistral-7B-Instruct-v0.2'
```

약 15GB. 시간 소요.

---

## 7. Loong 데이터셋 + manifest (Claude가 ssh로)

```bash
ssh vast 'cd ~ && git clone https://github.com/MozerWang/Loong.git'
ssh vast 'cd compblend3 && grep -q LOONG_DATA_DIR .env || echo "LOONG_DATA_DIR=\$HOME/Loong" >> .env'

ssh vast 'cd compblend3 && source .venv/bin/activate && \
    python scripts/build_loong_manifest.py \
        --level 1 --language english --domain academic \
        --max-tokens 16000 \
        --tokenizer mistralai/Mistral-7B-Instruct-v0.2 \
        --output data/manifests/loong_level1_eng_academic_16k.json'
```

---

## 8. Step 실험 워크플로우

각 step의 일반 흐름:

```bash
# 1. MacBook에서 코드 작성/편집 후 push
git add . && git commit -m "[step_XX] ..." && git push

# 2. vast.ai에서 pull + 실험
ssh vast 'cd compblend3 && git pull'
ssh vast 'cd compblend3 && source .venv/bin/activate && \
    export CUBLAS_WORKSPACE_CONFIG=:4096:8 && \
    python tasks/step_XX/run_*.py --out results/step_XX/vastai/'

# 3. vast.ai에서 결과 push
ssh vast 'cd compblend3 && git add results docs && git commit -m "[step_XX] results from vastai" && git push'

# 4. MacBook에서 회수
git pull
```

긴 실험은 tmux 안에서:
```bash
ssh vast 'cd compblend3 && tmux new -d -s step_XX "source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python tasks/step_XX/run_*.py --out results/step_XX/vastai/ 2>&1 | tee log_step_XX.txt"'

# 진행 확인
ssh vast 'tmux capture-pane -t step_XX -p | tail -50'
```

---

## 9. Persistent storage

vast.ai 인스턴스 stop/resume 시 다음이 보존되어야 함:
- 모델 캐시 (`~/.cache/huggingface`)
- venv (`~/compblend3/.venv`)
- Loong 데이터 (`~/Loong`)
- 결과 (`~/compblend3/results`)

인스턴스 destroy하면 다 사라짐. destroy 전 git push 필수.

---

## 10. SSH 자동화 규칙 (재확인)

1. **alias만 사용** — `ssh vast '...'`. IP/포트 inline ❌.
2. **인스턴스 ping 먼저** — `ssh vast 'nvidia-smi'` 실패 시 사용자에게 인스턴스 켜달라고 요청.
3. **시크릿 inline ❌** — `HF_TOKEN=...`을 ssh 명령 내부에 박지 않음. vast.ai .env에서만.
4. **파괴적 명령은 자유** — 가상 인스턴스라 `rm -rf` 등 OK. 단 인스턴스 destroy는 사용자 권한.

---

## 11. 트러블슈팅

### 인스턴스가 죽어있음
```bash
ssh vast 'echo hello'
# Connection refused 또는 timeout
```
→ vast.ai 콘솔에서 인스턴스 켜달라고 사용자에게 요청. Claude가 자동 spawn ❌.

### .venv가 인스턴스 reboot 후 깨짐
드물지만 가능. `bash setup/install_vastai.sh` 재실행하면 복구.

### 모델 다운로드 도중 끊김
HF cache는 resumable. 같은 명령 다시 실행하면 이어서 받음.

### git push 시 권한 에러
vast.ai에 GitHub 인증 정보가 없을 수 있음:
```bash
ssh vast 'git config --global credential.helper store'
ssh vast 'cd compblend3 && git push'
# 첫 push 시 username + Personal Access Token 입력
```

자세한 내용: `docs/setup/troubleshooting.md`
