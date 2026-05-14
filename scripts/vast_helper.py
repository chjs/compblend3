#!/usr/bin/env python3
"""vast.ai 인스턴스 lifecycle helper — step별 신규 할당/destroy 자동화.

MacBook(Claude Code)에서 실행한다. vast.ai 공식 CLI(`vastai`, Python 패키지)를
subprocess로 호출하며, GPU 의존 코드는 없다 — 인스턴스 lifecycle 관리 전용.

인증: repo 루트 `.env`의 `VAST_API_KEY`를 `vastai set api-key`로 1회 등록한다
      (DECISIONS.md §13 v8 시점 확정). 시크릿 값은 stdout/log/명령 인라인에
      노출하지 않는다 (DECISIONS.md §8.2 규칙 3) — 해당 명령은 `_run(secret=True)`로
      로그에 남기지 않고, 실패 시에도 명령 본문을 출력하지 않는다.

단순 SSH 명령 트리거는 `scripts/vast_run.sh`가 담당 — 이 파일은 인스턴스
lifecycle(할당/셋업/destroy/alias)만.

관련 정책: DECISIONS.md §8.3~§8.4 (vast.ai API 사용 + 인스턴스 lifecycle).

⚠️ 첫 인스턴스 할당 시 vast.ai CLI 실제 출력 스키마(`search offers` / `create
   instance` / `show instance`의 JSON 키)를 확인하고 필요 시 파서를 조정한다.
   `INSTANCE_IMAGE`도 가용성 확인 후 고정한다 (현재 잠정값).
"""
from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

# --- 상수 (CLAUDE.md §6.3 — magic number 금지) ---
REPO_URL = "https://github.com/chjs/compblend3.git"  # public repo — clone 인증 불필요
# disk_space 필터 필수 — 없으면 disk 작은 offer가 잡혀 `--disk` 요청이 12GB로 fallback됨
# (2026-05-15 첫 할당 실패 원인). disk_space는 offer 호스트의 가용 디스크(GB).
GPU_SEARCH_QUERY = (
    "gpu_name=A100_SXM4 gpu_ram>=80 num_gpus=1 disk_space>=100 rentable=true"
)
# base image: vast.ai PyTorch 계열. 잠정값 — 가용성 확인 후 고정.
# install_vastai.sh가 torch 2.10.0+cu128을 자체 설치하므로 base의 torch 버전은 무관,
# CUDA 12.x + Python만 있으면 됨.
INSTANCE_IMAGE = "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel"
INSTANCE_DISK_GB = 100           # Mistral-7B 작업에 충분 (모델 ~15GB + torch/cuda + 캐시)
MIN_DISK_AVAIL_GB = 40           # setup 전 `/` 가용 공간 가드 — 미달 시 install 중단
POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 600

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
SSH_CONFIG_PATH = Path.home() / ".ssh" / "config"
SSH_ALIAS = "vast"

# 인스턴스로 전송할 .env 키 화이트리스트.
# VAST_API_KEY는 제외 — destroy 권한을 가진 키를 rented 인스턴스에 두지 않는다.
INSTANCE_ENV_KEYS = ("HF_TOKEN", "HF_HOME", "GITHUB_PAT", "LOONG_DATA_DIR")


def _run(
    cmd: list[str],
    *,
    secret: bool = False,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """subprocess 래퍼.

    secret=True: 명령을 로그에 출력하지 않고, 실패 시에도 명령 본문을 노출하지 않는다
                 (cmd에 시크릿이 포함될 수 있는 경우).
    """
    if not secret:
        print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    try:
        return subprocess.run(
            cmd, check=check, text=True, capture_output=True, input=input_text
        )
    except subprocess.CalledProcessError as e:
        if secret:
            raise RuntimeError(
                f"secret 명령 실패 (rc={e.returncode}) — 명령 본문은 보안상 생략"
            ) from None
        raise


def _load_env() -> dict[str, str]:
    """repo 루트 `.env`를 dict로 파싱. 값은 로그에 출력하지 않는다."""
    if not ENV_PATH.exists():
        sys.exit(f"[vast_helper] .env 없음: {ENV_PATH}")
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip()
    return env


def _ensure_api_key() -> None:
    """`VAST_API_KEY`를 vast.ai CLI에 1회 등록 (`vastai set api-key`)."""
    api_key = _load_env().get("VAST_API_KEY", "")
    if not api_key:
        sys.exit("[vast_helper] .env에 VAST_API_KEY 없음")
    # secret=True — 키가 로그/에러 메시지에 남지 않게.
    _run(["vastai", "set", "api-key", api_key], secret=True)
    print("  vastai api-key 등록 완료")


def allocate_instance() -> dict:
    """A100-SXM4 80GB 인스턴스를 검색·할당하고 SSH 접속 정보를 반환.

    동작: vastai search offers → 최저가 offer 선택 → create instance →
          actual_status=running 까지 polling → ssh_host/ssh_port 회수.

    반환: {"instance_id": int, "ssh_host": str, "ssh_port": int}
    """
    _ensure_api_key()

    # 1) offer 검색 — 최저가 선택
    print("[allocate] offer 검색")
    res = _run(["vastai", "search", "offers", GPU_SEARCH_QUERY, "--raw"])
    offers = json.loads(res.stdout)
    if not offers:
        sys.exit("[vast_helper] 조건에 맞는 offer 없음 (A100-SXM4 80GB)")
    offers.sort(key=lambda o: o.get("dph_total", float("inf")))
    offer = offers[0]
    offer_id = offer["id"]
    print(
        f"  선택 offer {offer_id}: {offer.get('gpu_name')} "
        f"${offer.get('dph_total')}/h"
    )

    # 2) 인스턴스 생성
    print("[allocate] 인스턴스 생성")
    res = _run(
        [
            "vastai", "create", "instance", str(offer_id),
            "--image", INSTANCE_IMAGE,
            "--disk", str(INSTANCE_DISK_GB),
            "--raw",
        ]
    )
    created = json.loads(res.stdout)
    instance_id = created.get("new_contract") or created.get("id")
    if not instance_id:
        sys.exit(f"[vast_helper] 인스턴스 생성 실패: {created}")
    print(f"  instance_id={instance_id}")

    # 3) running 상태까지 polling
    print(f"[allocate] running 대기 (최대 {POLL_TIMEOUT_S}s)")
    deadline = time.time() + POLL_TIMEOUT_S
    while time.time() < deadline:
        res = _run(
            ["vastai", "show", "instance", str(instance_id), "--raw"],
            check=False,
        )
        if res.returncode == 0:
            info = json.loads(res.stdout)
            if info.get("actual_status") == "running" and info.get("ssh_host"):
                ssh_host = info["ssh_host"]
                ssh_port = int(info["ssh_port"])
                print(f"  running — {ssh_host}:{ssh_port}")
                return {
                    "instance_id": int(instance_id),
                    "ssh_host": ssh_host,
                    "ssh_port": ssh_port,
                }
        time.sleep(POLL_INTERVAL_S)
    sys.exit(
        f"[vast_helper] {POLL_TIMEOUT_S}s 내 running 안 됨 (id={instance_id}) — "
        "수동 확인 후 destroy 필요"
    )


def ssh_alias_register(ssh_host: str, ssh_port: int) -> None:
    """`~/.ssh/config`의 `Host vast` 블록을 새 인스턴스 정보로 추가/갱신.

    기존 `Host vast` 블록만 교체하고 다른 Host 항목은 건드리지 않는다.
    수정 전 `~/.ssh/config.bak`로 백업한다.
    """
    block = (
        f"Host {SSH_ALIAS}\n"
        f"    HostName {ssh_host}\n"
        f"    Port {ssh_port}\n"
        f"    User root\n"
        f"    StrictHostKeyChecking no\n"
        f"    UserKnownHostsFile /dev/null\n"
        f"    ServerAliveInterval 60\n"
    )
    SSH_CONFIG_PATH.parent.mkdir(mode=0o700, exist_ok=True)
    if SSH_CONFIG_PATH.exists():
        original = SSH_CONFIG_PATH.read_text()
        (SSH_CONFIG_PATH.parent / "config.bak").write_text(original)  # 백업
    else:
        original = ""

    # 기존 `Host vast` 블록만 제거 (다음 `Host ` 또는 EOF까지). 다른 Host는 보존.
    out: list[str] = []
    skipping = False
    for line in original.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("Host "):
            skipping = stripped == f"Host {SSH_ALIAS}"
        if not skipping:
            out.append(line)

    prefix = "".join(out).rstrip()
    new_content = (prefix + "\n\n" if prefix else "") + block
    SSH_CONFIG_PATH.write_text(new_content)
    SSH_CONFIG_PATH.chmod(0o600)
    print(f"[ssh_alias] Host {SSH_ALIAS} → {ssh_host}:{ssh_port} 등록")


def _ssh_base(ssh_host: str, ssh_port: int) -> list[str]:
    """인스턴스용 ssh 명령 prefix."""
    return [
        "ssh", "-p", str(ssh_port),
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        f"root@{ssh_host}",
    ]


def setup_instance(ssh_host: str, ssh_port: int) -> None:
    """새 인스턴스에 코드/환경/시크릿을 셋업.

    1. git clone (public repo — 인증 불필요)
    2. setup/install_vastai.sh 실행 (uv venv, torch 2.10+cu128, transformers)
    3. `.env` 전송 — VAST_API_KEY 제외한 화이트리스트 키만, stdin redirect로
       전송 (시크릿 값이 argv/로그에 남지 않게).
    """
    ssh_base = _ssh_base(ssh_host, ssh_port)

    # 디스크 가드 — 직전 실패(12GB overlay에 No space left) 재발 방지.
    # 긴 install_vastai.sh 실행 전에 `/` 가용 공간을 확인한다.
    res = _run(ssh_base + ["df -BG --output=avail / | tail -1"])
    avail_gb = int(res.stdout.strip().rstrip("G"))
    if avail_gb < MIN_DISK_AVAIL_GB:
        raise RuntimeError(
            f"인스턴스 `/` 가용 공간 {avail_gb}GB < {MIN_DISK_AVAIL_GB}GB — "
            "install 중단. 인스턴스 destroy 후 disk_space 큰 offer로 재할당 필요."
        )
    print(f"[setup] 디스크 가드 통과 — `/` 가용 {avail_gb}GB")

    print("[setup] git clone")
    _run(ssh_base + [f"test -d compblend3 || git clone {REPO_URL}"])

    print("[setup] install_vastai.sh 실행 (수 분 소요)")
    _run(ssh_base + ["cd compblend3 && bash setup/install_vastai.sh"])

    print("[setup] .env 전송 (화이트리스트 키만, VAST_API_KEY 제외)")
    env = _load_env()
    lines = [f"{k}={env[k]}" for k in INSTANCE_ENV_KEYS if env.get(k)]
    lines.append("COMPBLEND_ENV_TAG=vastai")
    env_content = "\n".join(lines) + "\n"
    # 명령 본문에는 시크릿이 없음(cat 리다이렉트뿐) — input_text는 _run이 로그에 남기지 않음.
    _run(ssh_base + ["cat > compblend3/.env"], input_text=env_content)
    print("  .env 전송 완료")


def destroy_instance(instance_id: int) -> None:
    """step 완료 후 인스턴스 destroy.

    사용자 승인 게이트 ❌ (DECISIONS.md §8.4). destroy 확인은 사용자가
    vast.ai 콘솔에서 별도로 한다.
    """
    print(f"[destroy] instance {instance_id}")
    _run(["vastai", "destroy", "instance", str(instance_id)])
    print("  destroy 요청 완료 (콘솔 확인은 사용자)")


if __name__ == "__main__":
    # 단독 실행: 할당 → alias 등록 → 셋업까지. step 워크플로우에서는 import해서 사용.
    info = allocate_instance()
    ssh_alias_register(info["ssh_host"], info["ssh_port"])
    setup_instance(info["ssh_host"], info["ssh_port"])
    print(json.dumps(info, indent=2))
