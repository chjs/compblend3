# Phase 0 — 환경 셋업 (3머신)

> Self-contained task. 이 파일만 읽고도 작업 가능해야 한다.

---

## 목표

compblend3 작업을 위한 3머신 환경을 셋업한다.

| 머신 | 용도 | 셋업 단계 |
|---|---|---|
| **MacBook M2** | Claude Code 실행 | Phase 0-A |
| **vast.ai A100 80GB** | primary 실험 | Phase 0-B |
| **사용자 로컬 A100 80GB** | occasional 검증 | Phase 0-C (선택) |

Claude Code는 MacBook에서 돌고, 실험 명령은 ssh vast로 트리거한다.

## 사전 조건

- MacBook M2 (사용자 머신)
- vast.ai 계정 + 인스턴스 (A100-SXM4 80GB) 생성됨
- HuggingFace 계정 + Mistral-7B-Instruct-v0.2 access 승인됨
- HF_TOKEN 발급 완료

## 통과 기준 / Invariants

Phase 0 완료 게이트 (vast.ai 셋업 = Phase 0-B는 게이트에서 제외 — Step 0 시작 직전에 진행):

1. **MacBook 환경 검증**: `python scripts/check_env.py` (macbook 모드) 통과
2. **LMCache pinning 확정**: `notes/lmcache_pinning.md` 에 commit SHA + vLLM 버전 기록, `patches/lmcache-vllm-cacheblend.md` 저장
3. **사용자 리뷰 승인**

vast.ai 환경 검증 / SSH alias 동작 / 모델 로드 / Loong manifest는 **Step 0 시작 직전 Phase 0-B에서** 확인한다. vast.ai 인스턴스는 step별로 새로 할당하므로(DECISIONS.md §8.4) Phase 0 시점에 미리 해둘 수 없다.

## 구현 사양

### Phase 0-A: MacBook 셋업

#### 0-A.1 — uv venv + 패키지 설치 (Claude가 실행)
```bash
cd ~/work/compblend3   # 사용자가 git clone 해둔 위치
bash setup/install_macbook.sh
```

이 스크립트가 하는 것:
- uv 설치 (없으면)
- `.venv` (Python 3.10) 생성
- PyTorch CPU 빌드 설치 (cu128 wheel ❌)
- transformers, datasets, numpy 등 가벼운 의존성
- 본 프로젝트 editable 설치 (`pip install -e .`)

#### 0-A.2 — SSH config 설정 (사용자가 직접)
사용자가 `~/.ssh/config`에 다음을 추가 (vast.ai 콘솔에서 받은 호스트/포트로):
```
Host vast
    HostName <vast-instance-host>
    Port <vast-port>
    User root
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

#### 0-A.3 — SSH alias 동작 확인 (Claude가 검증)
```bash
ssh vast 'echo hello && nvidia-smi --query-gpu=name --format=csv,noheader'
```
**기대**: `hello` + `NVIDIA A100-SXM4-80GB` 출력. 비밀번호 묻지 않음.

실패하면 사용자에게 SSH config 또는 ssh key 확인 요청.

#### 0-A.4 — MacBook 환경 검증 (Claude가 실행)
```bash
source .venv/bin/activate
python scripts/check_env.py
```
**기대**: macbook tag로 부분 OK. GPU/CUDA 없음은 macbook 모드에서 정상.

### Phase 0-B: vast.ai 셋업 (Claude가 자동 — Step 0 시작 직전에 진행)

> ⚠️ **Phase 0-B는 Phase 0 완료 게이트의 일부가 아니다.** vast.ai 인스턴스는 step별로 새로 할당하므로(DECISIONS.md §8.4), Phase 0 시점에 미리 셋업해둘 수 없다. 아래 절차는 Step 0 진입 시 `scripts/vast_helper.py`로 자동 수행한다.

#### 0-B.1 — 인스턴스 확인 (사용자가 결정)
사용자가 vast.ai 콘솔에서 인스턴스 띄움. Claude는 이 단계 자동화 ❌.

#### 0-B.2 — 코드 clone (Claude가 ssh로)
```bash
ssh vast 'cd ~ && git clone https://github.com/chjs/compblend3.git || (cd compblend3 && git pull)'
```

#### 0-B.3 — 환경 셋업 (Claude가 ssh로)
```bash
ssh vast 'cd compblend3 && bash setup/install_vastai.sh'
```

#### 0-B.4 — .env에 HF_TOKEN 채움 (사용자가 직접)
**중요**: 시크릿이라 Claude가 ssh inline으로 넣지 ❌.

Claude는 사용자에게 다음을 요청:
> "vast.ai의 `~/compblend3/.env` 파일에 `HF_TOKEN=hf_...` 한 줄 추가해주세요. 시크릿이라 제가 직접 안 넣겠습니다."

#### 0-B.5 — 환경 검증 (Claude가 ssh로)
```bash
ssh vast 'cd compblend3 && source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && python scripts/check_env.py'
```
**기대**: vastai tag로 전부 OK.

#### 0-B.6 — Mistral 모델 다운로드 (Claude가 ssh로)
```bash
ssh vast 'cd compblend3 && source .venv/bin/activate && python scripts/download_models.py --model mistralai/Mistral-7B-Instruct-v0.2'
```
약 15GB. 시간 소요.

#### 0-B.7 — Loong clone + manifest (Claude가 ssh로)
```bash
ssh vast 'cd ~ && git clone https://github.com/MozerWang/Loong.git || (cd Loong && git pull)'
ssh vast 'cd compblend3 && grep -q LOONG_DATA_DIR .env || echo "LOONG_DATA_DIR=$HOME/Loong" >> .env'
ssh vast 'cd compblend3 && source .venv/bin/activate && python scripts/build_loong_manifest.py --level 1 --language english --domain academic --max-tokens 16000 --tokenizer mistralai/Mistral-7B-Instruct-v0.2 --output data/manifests/loong_level1_eng_academic_16k.json'
```

**주의**: Loong jsonl schema가 다르면 script 수정 후 보고서에 기록.

#### 0-B.8 — Sanity forward (Claude가 ssh로)
```bash
ssh vast 'cd compblend3 && source .venv/bin/activate && python scripts/sanity_forward.py'
```
**기대**: top1 token = ` Paris` (또는 합리적인 다음 토큰).

#### 0-B.9 — vast.ai에서 결과 git push (Claude가 ssh로)
```bash
ssh vast 'cd compblend3 && git add results data/manifests && git commit -m "[phase_00] vast.ai setup complete" && git push'
```

#### 0-B.10 — MacBook에서 회수 (Claude가 실행)
```bash
git pull
```

### Phase 0-C: 로컬 A100 셋업 (사용자가 별도로, 선택)

Phase 0 완료 게이트의 일부 ❌. Claude는 자동화 ❌. 사용자가 직접:

```bash
# 사용자 로컬 A100 머신에서
git clone https://github.com/chjs/compblend3.git
cd compblend3
bash setup/install_local.sh
source .venv/bin/activate
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/check_env.py
```

`COMPBLEND_ENV_TAG=local_a100` 자동 추가됨. `results/phase_00/local_a100/env_check.json` 생성.

### Phase 0-D: LMCache reference pinning (MacBook에서 web_fetch)

DECISIONS.md §7.2의 LMCache reference를 확정한다. 이건 Phase 7에서 vLLM/LMCache reference 실행 시 쓰지만, **지금 확정해두지 않으면 reference가 떠다닌다** (LMCache repo가 빠르게 변함).

#### 0-D.1 — LMCache repo 최신 commit SHA 회수

```
MacBook에서, Claude가 web_fetch로:
https://api.github.com/repos/chjs/LMCache/branches/fix/cacheblend-vllm-v0.17.1-compat
```

응답에서 `commit.sha` 추출.

#### 0-D.2 — vLLM patch 정확한 형태 회수

```
MacBook에서, Claude가 web_fetch로:
https://raw.githubusercontent.com/chjs/LMCache/fix/cacheblend-vllm-v0.17.1-compat/examples/blend_kv_v1/README.md
```

이 README의 patch 섹션을 그대로 읽고:
- 어느 파일을 수정하는가 (`vllm/v1/worker/gpu_worker.py` 또는 다른 위치)
- 어느 함수 어느 라인에 무엇을 추가하는가
- 정확한 patch diff

→ `patches/lmcache-vllm-cacheblend.patch` 파일로 저장.

#### 0-D.3 — `notes/lmcache_pinning.md` 작성

```markdown
# LMCache Reference Pinning

확정일: YYYY-MM-DD

## LMCache
- Repo: chjs/LMCache
- Branch: fix/cacheblend-vllm-v0.17.1-compat
- Commit SHA: <40자 SHA>
- 확인 URL: https://github.com/chjs/LMCache/tree/<sha>

## vLLM
- 버전: v0.17.1
- Commit SHA: <40자 SHA>

## Patch
- 파일: patches/lmcache-vllm-cacheblend.patch
- 적용 대상: vllm 소스 트리
- 적용 방법: `cd vllm && git apply ../compblend3/patches/lmcache-vllm-cacheblend.patch`

## 검증
- 위 SHA로 checkout 후 LMCache README의 예제 실행이 가능함을 사용자가 확인.
- 본 검증은 Phase 7 진입 시 수행 (지금은 정보 회수만).
```

#### 0-D.4 — Commit

```
git add notes/lmcache_pinning.md patches/lmcache-vllm-cacheblend.patch
git commit -m "[phase_00] LMCache reference pinning (commit SHA + vLLM patch)"
git push
```

## 결과 저장 형식

- `results/phase_00/macbook/env_check.json` — MacBook 환경 정보 (GPU 없음 표시)
- `results/phase_00/vastai/env_check.json` — vast.ai 환경 정보 (전부 OK 기대)
- `results/phase_00/vastai/sanity_forward.json` — Mistral forward 결과
- `results/phase_00/local_a100/env_check.json` — (선택) 사용자 로컬
- `notes/lmcache_pinning.md` — LMCache reference pinning (commit SHA + vLLM patch 정보)
- `patches/lmcache-vllm-cacheblend.md` — vLLM 수동 패치 원문 인용 + 적용법 (unified diff 아님, `.patch` ❌)

## 보고서 작성 가이드

`docs/reports/phase_00_setup_report.md` 작성 (Claude가 MacBook에서). 필수 섹션:

1. **요약** — 3머신 셋업 완료 여부
2. **머신별 환경 정보** — macbook / vastai / (선택) local_a100, 각각 table
3. **환경 검증 결과** — 각 머신의 `env_check.json` 핵심 (table with PASS/FAIL badge)
4. **모델 로드 확인** — vast.ai sanity_forward 결과
5. **Loong manifest 통계** — n_instances, token distribution
6. **알려진 한계 / 의심스러운 부분**
7. **다음 단계** — Step 0 진행 권장

## 다음 step 게이트

Phase 0 완료 → Step 0 진입 조건:

- [ ] MacBook `env_check.json` macbook tag로 부분 OK
- [ ] `notes/lmcache_pinning.md` + `patches/lmcache-vllm-cacheblend.md` commit됨
- [ ] 사용자 리뷰 승인

(vast.ai env_check / sanity_forward / Loong manifest는 Phase 0 게이트가 아니라 Step 0 시작 직전 Phase 0-B에서 확인)

사용자에게 다음 요청:
> "Phase 0 완료. MacBook 셋업 + LMCache pinning 완료. LMCache pinning 확정 (commit SHA ___). vast.ai 셋업은 Step 0 시작 시 자동 진행. Step 0 진행해도 될까요?"

## 솔직성 노트

- Loong jsonl schema가 가정과 다를 수 있다. 다르면 script 수정 후 보고서에 명시.
- MacBook의 env_check은 macbook tag로 일부 검증만 통과 (GPU/CUDA 없음은 정상). check_env.py 가 환경별 분기.
- vast.ai SSH alias는 Claude가 `scripts/vast_helper.py`로 인스턴스 할당 후 `~/.ssh/config`의 `Host vast` 블록에 자동 등록/갱신한다 (다른 항목 건드리지 ❌).
- 사용자 로컬 A100은 Phase 0 게이트의 일부 ❌. Step 0 시작 직전 권장.
- **인스턴스가 step별로 새로 뜨므로 Mistral 모델(~15GB)과 Loong 데이터를 매 step 다시 다운로드할 가능성이 있다.** vast.ai persistent storage / 캐시 재사용 여부는 Step 0 진입 시 결정 (지금 결정 ❌, 짚어만 둠).
