# MacBook M2 환경 셋업 (Claude Code 실행 환경)

대상: 사용자의 MacBook Air M2. 여기서 Claude Code가 돌면서 코드 작성과 git을 하고, 실험은 `ssh vast`로 트리거한다.

---

## 1. MacBook의 역할

- **Claude Code 실행 환경**: 모든 코드 편집, git 작업, 보고서 작성
- **SSH 트리거**: vast.ai 인스턴스에 명령 전송 (`ssh vast '...'`)
- **GPU 의존 실험 ❌**: Mistral forward, 결정론 검증, F1 측정 등 모든 실험은 vast.ai에서

MacBook에 PyTorch를 깔긴 하지만 smoke test (import 가능한지 확인)와 코드 정적 분석용도. **실제 모델 inference는 절대 MacBook에서 안 한다**.

---

## 2. 환경 사양 확인

```bash
# OS 확인
sw_vers
# 기대: macOS, Apple Silicon

# 아키텍처 확인
uname -m
# 기대: arm64
```

---

## 3. uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.zshrc  # 또는 ~/.bashrc

# 확인
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

## 5. 자동 셋업 스크립트

```bash
bash setup/install_macbook.sh
```

이 스크립트가 하는 것:
- `.venv` (Python 3.10) 생성
- PyTorch CPU 빌드 설치 (cu128 wheel ❌)
- transformers, datasets, numpy 등 의존성
- 본 프로젝트 editable 설치
- `.env` 템플릿 생성 (`COMPBLEND_ENV_TAG=macbook` 자동)

---

## 6. SSH config 설정 (사용자 직접)

이게 핵심. Claude는 `ssh vast '...'` 형식으로만 명령 트리거한다. SSH alias 설정이 안 되어 있으면 Claude가 vast.ai에 접근할 수 없다.

`~/.ssh/config` 에 다음을 추가:

```
Host vast
    HostName <vast.ai 콘솔에서 받은 host>
    Port <vast.ai 콘솔에서 받은 port>
    User root
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

`<...>` 부분은 vast.ai 콘솔에서 인스턴스 SSH 정보 보면 나옴 (보통 `ssh -p <port> root@<host>` 형식).

### SSH 키 등록

vast.ai 콘솔 → Account → SSH Keys 에 본인 MacBook의 `~/.ssh/id_rsa.pub` 내용 추가.

```bash
# 공개키 보기
cat ~/.ssh/id_rsa.pub
# 이걸 vast.ai 콘솔에 붙여넣기
```

### 동작 확인

```bash
ssh vast 'echo hello && hostname'
```

비밀번호 묻지 않고 즉시 `hello` + 인스턴스 hostname 출력되면 OK.

비밀번호 묻거나 권한 에러나면:
- 공개키가 vast.ai에 등록 안 됨 → 콘솔 확인
- 또는 private key permission 문제: `chmod 600 ~/.ssh/id_rsa`

---

## 7. 환경 검증

```bash
source .venv/bin/activate
python scripts/check_env.py
```

기대 출력 (macbook tag):
```
[OK] Python: 3.10.x
[OK] PyTorch: 2.10.0  (cu128 아님, 정상)
[INFO] CUDA: not available (macbook tag, 정상)
[INFO] GPU: not available (macbook tag, 정상)
[INFO] Deterministic mode: not testable without GPU (macbook tag, 정상)
[OK] Transformers: 4.51.x
[INFO] HF_TOKEN: not set on macbook (정상, 실험은 vast.ai에서)
[OK] Environment tag: macbook
```

GPU/CUDA/deterministic 항목이 `INFO` (FAIL 아님)으로 표시되면 정상.

---

## 8. .env 설정 (선택)

MacBook 측 `.env`는 사실상 필요 없음. `COMPBLEND_ENV_TAG=macbook` 한 줄이면 충분.

실험 시크릿(HF_TOKEN)은 vast.ai의 `.env`에만 둔다. MacBook의 `.env`에는 가능한 한 시크릿 ❌.

만약 MacBook에서도 tokenizer 같은 가벼운 작업을 한다면:
```bash
# 정 필요하면 추가
echo "HF_TOKEN=hf_..." >> .env
```
하지만 이 토큰은 모델 다운로드용도(15GB)라 MacBook에선 쓸 일 거의 없음.

---

## 9. Claude Code 실행

이 시점에서 MacBook은 Claude Code 실행 준비 완료. Claude Code 첫 메시지로 (Claude Code app에서):

```
compblend3 프로젝트입니다. CLAUDE.md를 먼저 읽어주세요.
```

또는 새 세션에서 더 명확하게:

```
compblend3, 새 세션입니다.
다음 순서로 읽어주세요: GOAL.md → PROGRESS.md → CLAUDE.md → DECISIONS.md
그 다음 ssh vast 'nvidia-smi' 로 인스턴스 살아있는지 확인하고, PROGRESS.md의 next task를 진행해주세요.
```

---

## 10. 트러블슈팅

### SSH alias가 자꾸 끊김
vast.ai 인스턴스가 idle 상태가 되면 SSH session timeout. `~/.ssh/config`에 `ServerAliveInterval 60` 추가됐는지 확인.

### `ssh vast 'long command'` 가 중간에 끊김
긴 명령은 tmux/nohup 권장:
```bash
ssh vast 'cd compblend3 && tmux new -d -s exp "python ..."'
ssh vast 'tmux capture-pane -t exp -p'  # 결과 확인
```

### MacBook의 PyTorch가 import 실패
- ARM64 wheel이 맞는지 확인: `pip show torch | grep Location`
- 자주 발생: `OMP_NUM_THREADS` 충돌. `export OMP_NUM_THREADS=1` 시도.

### .venv가 Apple Silicon 네이티브가 아님
```bash
python -c "import platform; print(platform.machine())"
# 기대: arm64
# 만약 x86_64 나오면 Rosetta 모드로 실행 중. venv 재생성 필요.
```

---

## 11. 다음 단계

MacBook 셋업 완료 → Phase 0-B (vast.ai 셋업)으로 진행.

자세한 절차: `tasks/phase_00_setup.md` §Phase 0-B
