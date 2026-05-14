#!/usr/bin/env bash
# vast.ai SSH 트리거 helper
# 사용법:
#   bash scripts/vast_run.sh ping              # nvidia-smi 가벼운 확인
#   bash scripts/vast_run.sh sync              # git pull on vast.ai
#   bash scripts/vast_run.sh run "python tasks/step_00/run_..."   # 임의 명령
#   bash scripts/vast_run.sh tmux step_00 "python tasks/step_00/..."   # tmux로 detached
#   bash scripts/vast_run.sh tmux_log step_00   # tmux 세션 출력 확인
#   bash scripts/vast_run.sh push              # vast.ai에서 git add+commit+push
#
# 가정: ~/.ssh/config 에 `Host vast` alias 설정됨

set -euo pipefail

CMD="${1:-help}"

case "$CMD" in
    help)
        cat <<EOF
사용법:
  bash scripts/vast_run.sh ping
  bash scripts/vast_run.sh sync
  bash scripts/vast_run.sh run "<command>"
  bash scripts/vast_run.sh tmux <session_name> "<command>"
  bash scripts/vast_run.sh tmux_log <session_name>
  bash scripts/vast_run.sh push [commit_message]
EOF
        ;;

    ping)
        echo "==> vast.ai 인스턴스 ping"
        ssh vast 'echo OK && nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits | head -1'
        ;;

    sync)
        echo "==> vast.ai에서 git pull"
        ssh vast 'cd compblend3 && git pull'
        ;;

    run)
        REMOTE_CMD="${2:?command 필요}"
        echo "==> vast.ai에서 실행: $REMOTE_CMD"
        ssh vast "cd compblend3 && source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && $REMOTE_CMD"
        ;;

    tmux)
        SESSION="${2:?session_name 필요}"
        REMOTE_CMD="${3:?command 필요}"
        echo "==> vast.ai의 tmux 세션 '$SESSION' 에서 실행"
        # 기존 동명 세션이 있으면 죽이고 새로 띄움
        ssh vast "tmux kill-session -t $SESSION 2>/dev/null || true"
        ssh vast "cd compblend3 && tmux new -d -s $SESSION 'source .venv/bin/activate && export CUBLAS_WORKSPACE_CONFIG=:4096:8 && $REMOTE_CMD 2>&1 | tee log_$SESSION.txt'"
        echo "    실행 중. 로그 확인: bash scripts/vast_run.sh tmux_log $SESSION"
        ;;

    tmux_log)
        SESSION="${2:?session_name 필요}"
        echo "==> tmux 세션 '$SESSION' 출력 (마지막 50줄)"
        ssh vast "tmux capture-pane -t $SESSION -p 2>/dev/null | tail -50 || echo '세션 없음 또는 종료됨'"
        ;;

    push)
        MSG="${2:-[automated] results from vastai}"
        echo "==> vast.ai에서 git add + commit + push"
        ssh vast "cd compblend3 && git add results docs data/manifests && git diff --cached --quiet || (git commit -m '$MSG' && git push)"
        echo "==> MacBook에서 git pull"
        git pull
        ;;

    *)
        echo "Unknown command: $CMD"
        bash "$0" help
        exit 1
        ;;
esac
