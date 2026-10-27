#!/usr/bin/env bash
# Train one temporal locomotion objective; expert selection remains explicit.
set -euo pipefail

cd "$(dirname "$0")/../.."

mode="${1:-envelope}"
case "$mode" in
  envelope)
    export TASK_ID=Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Envelope
    default_lr=0.00001
    ;;
  moving)
    export TASK_ID=Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Gait-Settle
    default_lr=0.000001
    ;;
  settling)
    export TASK_ID=Agibot-X2-Ultra-Behavior-Factory-Recorded-History2-Full-Settle
    default_lr=0.000003
    ;;
  *)
    echo "Usage: PRETRAINED_CHECKPOINT=/path/to/history2.pt $0 [envelope|moving|settling]" >&2
    echo "Set FROM_SCRATCH=1 to initialize a new History2 network instead." >&2
    exit 2
    ;;
esac

if [[ "${FROM_SCRATCH:-0}" == "1" ]]; then
  unset PRETRAINED_CHECKPOINT
elif [[ -z "${PRETRAINED_CHECKPOINT:-}" ]]; then
  echo "Set PRETRAINED_CHECKPOINT to a 208-input checkpoint (or FROM_SCRATCH=1)." >&2
  exit 2
fi

export RUN_NAME="${RUN_NAME:-x2_ultra_history2_${mode}}"
export NUM_ENVS="${NUM_ENVS:-1024}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-501}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
export LEARNING_RATE="${LEARNING_RATE:-$default_lr}"
export PPO_SCHEDULE="${PPO_SCHEDULE:-fixed}"
export RESET_OPTIMIZER_ON_LOAD="${RESET_OPTIMIZER_ON_LOAD:-1}"

exec scripts/x2_ultra/train_behavior.sh
