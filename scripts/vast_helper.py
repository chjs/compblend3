#!/usr/bin/env python3
"""vast.ai 인스턴스 lifecycle helper — step별 신규 할당/destroy 자동화.

⚠️ Step 0 진입 직전에 구현한다. 지금은 인터페이스 명세(시그니처 + 의도된
vast.ai CLI 호출 형태)만 정의된 placeholder다. 실행 가능하지 않다.

전제:
- MacBook(Claude Code)에서 실행. vast.ai 공식 CLI(`vastai`, Python 패키지)를
  subprocess로 호출.
- `VAST_API_KEY`는 MacBook의 `.env`에 있음. CLI 인증 방식은 구현 시 확정:
  `vastai set api-key <KEY>`로 1회 저장하는 방식이 가장 확실해 보임. `--api-key`
  flag, `VAST_API_KEY` 환경변수 자동인식 여부는 구현 시 확인 필요.
- 시크릿 값은 stdout/log/명령 인라인에 노출 ❌ (DECISIONS.md §8.2 규칙 3).
- 단순 SSH 명령 트리거는 `scripts/vast_run.sh`가 담당 — 이 파일은 인스턴스
  lifecycle(할당/셋업/destroy/alias)만.

관련: DECISIONS.md §8.3~§8.4 (vast.ai API 사용 + 인스턴스 lifecycle).
"""
from __future__ import annotations


def allocate_instance() -> dict:
    """A100-SXM4 80GB 인스턴스를 검색·할당하고 SSH 접속 정보를 반환.

    의도된 vast.ai CLI 호출:
      vastai search offers 'gpu_name=A100_SXM4 gpu_ram>=80 num_gpus=1 rentable=true' --raw
      vastai create instance <offer_id> --image <pytorch image> --disk <GB> --raw
      vastai show instance <instance_id> --raw   # status=running 까지 polling
      vastai ssh-url <instance_id>               # ssh://root@host:port

    반환: {"instance_id": int, "ssh_host": str, "ssh_port": int}
    """
    raise NotImplementedError("Step 0 진입 직전 구현")


def setup_instance(ssh_host: str, ssh_port: int) -> None:
    """새 인스턴스에 코드/환경/시크릿을 셋업.

    의도된 동작:
      1. ssh <host> 'git clone https://github.com/chjs/compblend3.git'
      2. ssh <host> 'cd compblend3 && bash setup/install_vastai.sh'
      3. .env의 HF_TOKEN을 인스턴스 ~/compblend3/.env로 전송
         (시크릿 — 명령 인라인에 값 박지 않기. scp 또는 ssh stdin redirect로,
          stdout 노출 ❌.)
    """
    raise NotImplementedError("Step 0 진입 직전 구현")


def destroy_instance(instance_id: int) -> None:
    """step 완료 후 인스턴스 destroy.

    의도된 vast.ai CLI 호출:  vastai destroy instance <instance_id>

    사용자 승인 게이트 ❌ (DECISIONS.md §8.4). destroy 확인은 사용자가
    vast.ai 콘솔에서 별도로 한다.
    """
    raise NotImplementedError("Step 0 진입 직전 구현")


def ssh_alias_register(ssh_host: str, ssh_port: int) -> None:
    """~/.ssh/config의 `Host vast` 블록을 새 인스턴스 정보로 추가/갱신.

    의도된 동작:
      ~/.ssh/config에서 기존 `Host vast` 블록을 찾아 HostName/Port만 교체.
      없으면 새 블록 append. 다른 Host 항목은 건드리지 않음. 수정 전 백업 권장.

    블록 형태:
      Host vast
          HostName <ssh_host>
          Port <ssh_port>
          User root
          StrictHostKeyChecking no
          UserKnownHostsFile /dev/null
    """
    raise NotImplementedError("Step 0 진입 직전 구현")


if __name__ == "__main__":
    raise SystemExit("placeholder — Step 0 진입 직전 구현 예정")
