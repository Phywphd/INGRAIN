#!/usr/bin/env bash
# ============================================================================
# INGRAIN — full TAO-test evaluation (all 1419 videos), paper protocol.
#
# Wraps scripts/eval.sh, so it inherits the same evaluation setup: the full
# 1203-name taxonomy, evaluated in text chunks — one decoder pass per chunk,
# 31 at the default chunk size of 40.
#
# Resumes from the newest snapshot; aborts below MIN_DISK_MB; removes the large
# intermediate files once a score exists.
#
# Env: PY, CKPT, TAG, MIN_DISK_MB, ATTEMPTS, INGRAIN_EVAL_SNAPEVERY
# ============================================================================
set -uo pipefail
PY="${PY:-python}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY

CKPT="${CKPT:-ckpt/ingrain.pth}"
TAG="${TAG:-ingrain_test}"
OUT="eval/results/${TAG}"
LOG="logs/eval_${TAG}.log"
MIN_DISK_MB="${MIN_DISK_MB:-15000}"
mkdir -p "$OUT" logs

export INGRAIN_EVAL_SNAPEVERY="${INGRAIN_EVAL_SNAPEVERY:-20}"

disk_ok() {
  local free_mb
  free_mb=$(df -Pm . | awk 'NR==2{print $4}')
  echo "[eval-retry $(date +%H:%M:%S)] disk free ${free_mb}MB; snapshots=$(ls "$OUT"/tao_track_snapshot_v*.json 2>/dev/null | wc -l) ($(du -sh "$OUT" 2>/dev/null | cut -f1))"
  [ "${free_mb:-0}" -ge "$MIN_DISK_MB" ]
}

for attempt in $(seq 1 "${ATTEMPTS:-30}"); do
  if ! disk_ok; then
    echo "[eval-retry $(date +%H:%M:%S)] disk below ${MIN_DISK_MB}MB free, aborting before the disk fills"; exit 3
  fi

  SNAP=$(ls -t "$OUT"/tao_track_snapshot_v*.json 2>/dev/null | head -1)
  if [ -n "$SNAP" ]; then
    export RESUME_SNAP="$SNAP"
    echo "[eval-retry $(date +%H:%M:%S)] attempt $attempt: RESUME from $SNAP"
  else
    unset RESUME_SNAP
    echo "[eval-retry $(date +%H:%M:%S)] attempt $attempt: FRESH start"
  fi

  bash scripts/eval.sh "$CKPT" "$TAG" test tao_test_FULL_video_list.txt 1419
  EXIT=$?

  if grep -qE "OutOfMemoryError|CUDA out of memory" "$LOG" 2>/dev/null; then
    echo "[eval-retry $(date +%H:%M:%S)] OOM on attempt $attempt -> resume from newest snapshot"; sleep 15; continue
  fi

  # Success means a SCORE exists, not merely that metrics.json is present.
  if [ $EXIT -eq 0 ] && grep -q '"TETA"' "$OUT/metrics.json" 2>/dev/null; then
    echo "[eval-retry $(date +%H:%M:%S)] ===== FULL TAO-test DONE ====="
    cat "$OUT/metrics.json"
    grep -iE "COMBINED|TETA50: INGRAIN|^Base |^Novel " "$LOG" | tail -6
    echo "[eval-retry $(date +%H:%M:%S)] hygiene: removing tao_track.json + snapshots"
    rm -f "$OUT"/tao_track.json "$OUT"/tao_track_snapshot_v*.json
    rm -rf "$OUT"/_live
    exit 0
  fi

  NOWSNAP=$(ls -t "$OUT"/tao_track_snapshot_v*.json 2>/dev/null | head -1)
  if [ -n "$NOWSNAP" ]; then
    echo "[eval-retry $(date +%H:%M:%S)] exit=$EXIT but progress exists ($NOWSNAP) -> resume"; tail -8 "$LOG"; sleep 15; continue
  fi
  echo "[eval-retry $(date +%H:%M:%S)] exit=$EXIT with NO progress (likely a config error) -> ABORT"; tail -25 "$LOG"; exit 1
done
echo "[eval-retry $(date +%H:%M:%S)] gave up after ${ATTEMPTS:-30} attempts"; exit 1
