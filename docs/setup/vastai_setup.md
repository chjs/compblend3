# vast.ai 인스턴스 셋업 가이드 (primary 실험 환경)

대상: step별로 신규 할당되는 vast.ai 인스턴스. Claude Code(MacBook)가 `scripts/vast_helper.py`로 할당·셋업·destroy를 자동 수행하고, `ssh vast`로 명령을 트리거한다. (인스턴스 lifecycle 정책: DECISIONS.md §8.4)

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
- step별 신규 할당 → step 완료 시 destroy. stop/resume 안 함 (DECISIONS.md §8.4).
- 비용 모니터링은 사용자 책임 (DECISIONS.md §8.4).

---

## 3. 사용자 1회성 준비 (최초 1회만)

step별 인스턴스 할당·SSH alias 등록·셋업은 Claude가 `scripts/vast_helper.py`로 자동 수행한다 (DECISIONS.md §8.4). 사용자는 다음 두 가지만 **최초 1회** 준비한다.

### 3.1 vast.ai 계정에 SSH 공개키 등록
vast.ai 콘솔 → Account → SSH Keys → MacBook의 `~/.ssh/id_rsa.pub` 내용 등록.
계정 단위 설정이라 인스턴스마다 다시 할 필요 없음.

### 3.2 MacBook `.env`에 `VAST_API_KEY` 채움
Claude가 인스턴스를 할당/destroy하려면 필요. 시크릿이므로 stdout/log/명령 인라인에 노출 ❌ (DECISIONS.md §8.2 규칙 3).

> SSH 정보 확인, `~/.ssh/config`의 `Host vast` alias 등록, 동작 확인은
> `scripts/vast_helper.py`의 `allocate_instance()` / `ssh_alias_register()`가 step
> 시작 시 자동 수행한다 — 사용자 수동 절차 없음. alias 블록 형태는 해당 docstring 참조.

---

## 4. 인스턴스 셋업 (Claude가 `vast_helper.py setup_instance()`로 자동 수행)

step 시작 시 `allocate_instance()` 직후 `setup_instance()`가 아래를 자동 실행한다. 아래 명령은 그 내부 동작을 사람이 읽기 위한 reference다.

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

### 4.4 .env에 HF_TOKEN 전송 (`setup_instance()`가 자동)

`setup_instance()`가 MacBook `.env`의 `HF_TOKEN`을 인스턴스 `~/compblend3/.env`로 전송한다.
**시크릿이므로 명령 인라인에 값을 박지 않는다** — scp 또는 ssh stdin redirect로, stdout 노출 ❌ (DECISIONS.md §8.2 규칙 3, `scripts/vast_helper.py` `setup_instance()` docstring 참조).

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

## 9. 영속성 — 없음 (step별 destroy)

인스턴스는 step 완료 시 destroy되므로 **영속 저장은 없다** (DECISIONS.md §8.4).

- 매 step `setup_instance()`가 venv·PyTorch·transformers를 새로 구성한다.
- 모델 캐시(`~/.cache/huggingface`)·Loong 데이터(`~/Loong`)도 매 step 재다운로드된다 (~15GB+, 시간 비용 감수).
- 결과(`~/compblend3/results`)는 **destroy 전에 반드시 git push**로 회수한다 — push 안 한 결과는 destroy와 함께 사라진다.

---

## 10. SSH 자동화 규칙 (재확인)

1. **alias만 사용** — `ssh vast '...'`. IP/포트 inline ❌.
2. **인스턴스 없으면 Claude가 자동 할당** — step 시작 시 `scripts/vast_helper.py allocate_instance()`로 신규 할당. 사용자에게 켜달라고 요청하지 않음 (DECISIONS.md §8.4).
3. **시크릿 inline ❌** — `HF_TOKEN`, `VAST_API_KEY` 값을 ssh 명령 내부·stdout·log에 박지 않음. vast.ai .env에서만 source.
4. **파괴적 명령은 자유** — 가상 인스턴스라 `rm -rf` 등 OK. 인스턴스 destroy도 Claude가 자동 수행 (사용자 승인 게이트 ❌, 콘솔에서 사용자가 확인).

---

## 11. 트러블슈팅

### 인스턴스가 죽어있음 / 없음
```bash
ssh vast 'echo hello'
# Connection refused 또는 timeout
```
→ Claude가 `scripts/vast_helper.py allocate_instance()`로 신규 인스턴스를 자동 할당하고 `ssh_alias_register()`로 `Host vast`를 갱신한다 (DECISIONS.md §8.4). 사용자 개입 불필요. step 도중 인스턴스가 죽었고 결과가 git push 전이면 해당 step을 재실행한다.

### .venv가 인스턴스 reboot 후 깨짐
드물지만 가능. `bash setup/install_vastai.sh` 재실행하면 복구.

### 모델 다운로드 도중 끊김
HF cache는 resumable. 같은 명령 다시 실행하면 이어서 받음.

### git push 시 권한 에러
인스턴스는 step별 신규 할당이라 매번 GitHub 인증이 새로 필요하다. `setup_instance()`가 인증을 함께 셋업해야 한다 (구현 시 처리 — 현재 `vast_helper.py` placeholder).
수동 우회:
```bash
ssh vast 'git config --global credential.helper store'
ssh vast 'cd compblend3 && git push'   # 첫 push 시 username + PAT 입력
```
> 참고: PAT를 인라인에 박지 않는다 (시크릿). 구현 시 deploy key 또는 stdin redirect 검토 — DECISIONS.md §8.2 규칙 3.

자세한 내용: `docs/setup/troubleshooting.md`
