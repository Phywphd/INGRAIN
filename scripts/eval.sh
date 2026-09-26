#!/bin/bash
# ============================================================================
# INGRAIN — evaluation on TAO, as described in the paper.
#
# Vocabulary  The full dataset taxonomy (1203 LVIS-v1 names), evaluated in
#             text chunks — one decoder pass per chunk (31 at the default
#             chunk size). The prompt is not narrowed to the categories that
#             occur in the evaluation ground truth.
# Propagation Each track's positional reference is its own predicted box from
#             the preceding observation.
# Lifecycle   A track is removed after 5 consecutive inactive frames.
#
# Usage: eval.sh <ckpt> <tag> <split> <vlist_basename> <maxvid> [filter_thresh]
# Env:   PY, NEED_FREE_MB, GPU_WAIT_MAX_S, RESUME_SNAP,
#        INGRAIN_TSA_DIM, INGRAIN_EVAL_SNAPEVERY, MEM_AGG, MEM_DEPTHS, MEM_EMA,
#        and the five architecture switches exported below.
# Protocol constants (chunk size, thresholds, miss tolerance, NMS) are fixed
# in eval/config.py.
# ============================================================================
set -uo pipefail
[ $# -ge 5 ] || { echo "Usage: eval.sh <ckpt> <tag> <split> <vlist_basename> <maxvid> [filter_thresh]" >&2; exit 2; }
CKPT="$1"; TAG="$2"; SPLIT="$3"; VLIST="$4"; MAXVID="$5"
# 6th positional wins; FILTER=<x> in the environment also works.
FILTER="${6:-${FILTER:-}}"
PY="${PY:-python}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- architecture mirror: the same five switches scripts/train.sh sets. These
#     construct modules at model build; a mismatch changes the model.
# Overridable: an ablation checkpoint must be evaluated with the same switches
# it was trained with, so `INGRAIN_HIDDEN_FUSE=0 bash scripts/eval.sh ...` has
# to work.
export INGRAIN_TRACK_CONTENT_ENCROI="${INGRAIN_TRACK_CONTENT_ENCROI:-1}"
export INGRAIN_HIDDEN_FUSE="${INGRAIN_HIDDEN_FUSE:-1}"
export INGRAIN_ENCROI_MEM_FUSE="${INGRAIN_ENCROI_MEM_FUSE:-1}"
export INGRAIN_MEMFUSE_INTERLEAVE="${INGRAIN_MEMFUSE_INTERLEAVE:-1}"
export INGRAIN_TRACK_SAMPLING_ADAPTER="${INGRAIN_TRACK_SAMPLING_ADAPTER:-1}"
export INGRAIN_TSA_DIM="${INGRAIN_TSA_DIM:-128}"
export INGRAIN_NO_PIN_MEMORY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


OUT="eval/results/${TAG}"
LOG="logs/eval_${TAG}.log"
mkdir -p "$OUT" logs

# Score threshold used for ablations (6th argument or $FILTER).
FILTER_ARG="--filter-score-thresh ${FILTER:-0.15}"
# RESUME_SNAP=<tao_track_snapshot_*.json> continues an interrupted run; the
# videos already in it are skipped and their tracks carried into the scoring.
RESUME_ARG=""; [ -n "${RESUME_SNAP:-}" ] && RESUME_ARG="--resume-snapshot $RESUME_SNAP"

# Wait for NEED_FREE_MB of free GPU memory before starting. NEED_FREE_MB=0 skips
# the check; the wait is bounded.
wait_gpu() {
  local need="${NEED_FREE_MB:-22000}"
  [ "$need" -le 0 ] && return 0
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[eval] nvidia-smi not found; skipping the free-memory check " \
         "(set NEED_FREE_MB=0 to silence this)"; return 0
  fi
  local waited=0 max="${GPU_WAIT_MAX_S:-1800}"
  while :; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    case "$FREE" in (''|*[!0-9]*)
      echo "[eval] nvidia-smi gave no usable reading ('$FREE'); skipping the check"
      return 0 ;;
    esac
    [ "$FREE" -ge "$need" ] && { echo "[eval $(date +%H:%M:%S)] GPU free ${FREE}MiB, go"; return 0; }
    if [ "$waited" -ge "$max" ]; then
      echo "[eval] only ${FREE}MiB free after ${waited}s (need ${need}MiB). Lower" \
           "NEED_FREE_MB, raise GPU_WAIT_MAX_S, or free the GPU." >&2
      exit 4
    fi
    echo "[eval $(date +%H:%M:%S)] GPU free ${FREE}MiB < ${need}MiB, wait"
    sleep 30; waited=$((waited + 30))
  done
}

wait_gpu
echo "[eval $(date +%H:%M:%S)] $TAG: $SPLIT ${MAXVID}v, full taxonomy, thr ${FILTER:-0.15}"

"$PY" -u evaluate.py \
  --tao-subset-eval --split "$SPLIT" \
  --checkpoint "$CKPT" \
  --video-list "eval/results/$VLIST" --max-videos "$MAXVID" \
  $FILTER_ARG $RESUME_ARG \
  --snapshot-every "${INGRAIN_EVAL_SNAPEVERY:-20}" \
  --trajectory-state-prop --memory-attn-residual-scale 0.1 \
  --mem-aggregation "${MEM_AGG:-attention}" \
  --memory-depths "${MEM_DEPTHS:-3}" \
  --mem-ema-alpha "${MEM_EMA:-0.9}" \
  --e2e-assoc \
  --decoupling-depth 3 \
  --track-path --staged-decoder \
  --output-dir "$OUT" > "$LOG" 2>&1
EXIT=$?
echo "[eval $(date +%H:%M:%S)] exit=$EXIT (log: $LOG)"
grep -iE "COMBINED|TETA50: INGRAIN|^Base |^Novel " "$LOG" | tail -6
# Success means a SCORE was produced, not merely that the process exited 0.
if [ $EXIT -eq 0 ] && ! grep -q '"TETA"' "$OUT/metrics.json" 2>/dev/null; then
  echo "[eval] exited 0 but $OUT/metrics.json carries no TETA score" >&2
  exit 5
fi
exit $EXIT
