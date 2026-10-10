#!/bin/bash
# ============================================================================
# INGRAIN — training, as described in the paper.
#
# Data      LVIS, base categories only. Annotations carry no rare-class
#           instances, and the training script additionally keeps rare class
#           names out of the negative-sampling pool, so no novel category is
#           supervised in either direction. No TAO video is used.
# Schedule  16 epochs, sequence length 4 -> 7 observations at epochs 6/10/14,
#           multi-depth trajectory memory enabled from epoch 3.
# Seed      42 (python / numpy / torch / DataLoader workers).
# Optim     AdamW, weight decay 1e-4, cosine decay, 200 warm-up steps.
#           1e-4 for the newly introduced modules, 3e-5 for the track path
#           and the fusion stage. The track path is the decoupling stage's
#           copied track layers (decoder.track_dec_*, --track-path-lr); the
#           fusion stage is decoder.layers.{3,4,5} + their regression
#           branches, shared by both paths (--lr-gd).
# Model     Staged dual-path decoder (split depth n = 3), the
#           multi-depth trajectory memory (K = 16, depths d = {0,1,2,3}),
#           observation-guided propagation (enc-ROI observation fused with the
#           decoded track state), and the track-query adapter (r = 128).
#
# Flag glossary (script name -> paper term):
#   --decoupling-depth / --track-path / --staged-decoder
#                                    staged dual-path decoder, decoupling depth 3
#   --trajectory-state-prop, --mem-* multi-depth trajectory memory (aggregation,
#                                    depths d = 1..3 + the d = 0 seed, scale 0.1)
#   --e2e-assoc                      association = track-slot continuity;
#                                    no detect-to-track matcher at any point
#   --det-pool-mode dedup, --dedup-bg-*  the duplicate-of-track background
#                                    focal term
#   --lambda-det-* / --lambda-track-*   loss weights: constant 0.1 on the
#                                    detection branch / 1.0 -> 2.0 on the track branch
#   --dn-anchor-*                    track-anchor denoise (probability, center
#                                    jitter, size jitter)
#   --motion-aug-level coordinated     smooth rotation (<= 25 deg) and
#                                    scale ramps of the motion augmentation
#   --gd-clip-norm / --clip-max-norm decoder-specific 0.1, then global 1.0
#   --lr / --lr-gd / --track-path-lr   1e-4 for the new modules /
#                                    3e-5 for the fusion stage and track path
#
# Overridable from the caller's env: PY, SEED, INIT_WEIGHTS, RESUME, OUT, LOG, BS,
# ACCUM, SAVEFRAC, MAXSTEPS, EPOCHS, WARMUP, SEQSCHED, MEMWARM, MEM_AGG,
# MEM_DEPTHS, MEM_EMA, INGRAIN_ANN_FILE, INGRAIN_H5_FILE, INGRAIN_PROMPT_MAXTOTAL.
# ============================================================================
set -uo pipefail
PY="${PY:-python}"                                     # override: PY=/path/to/python
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # repo root (portable)
OUT=${OUT:-ckpt/ingrain_train}
LOG=${LOG:-logs/train_ingrain.log}
mkdir -p "$OUT" logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAVE_FRAC_EPOCH=${SAVEFRAC:-0.04}

# --- architecture (build-time; the eval mirror sets the same five) ----------
# Overridable, so an ablation can turn a component off from the caller:
#   INGRAIN_HIDDEN_FUSE=0 bash scripts/train.sh
# OGP instance observation + its fusion with the decoded track state.
export INGRAIN_TRACK_CONTENT_ENCROI="${INGRAIN_TRACK_CONTENT_ENCROI:-1}"
export INGRAIN_HIDDEN_FUSE="${INGRAIN_HIDDEN_FUSE:-1}"
# Multi-depth trajectory memory: seed (d=0) + per-track-path depths (d=1,2,3).
export INGRAIN_ENCROI_MEM_FUSE="${INGRAIN_ENCROI_MEM_FUSE:-1}"
export INGRAIN_MEMFUSE_INTERLEAVE="${INGRAIN_MEMFUSE_INTERLEAVE:-1}"
# Track-query adapter Adapt_T, r = 128, on the track slice only.
export INGRAIN_TRACK_SAMPLING_ADAPTER="${INGRAIN_TRACK_SAMPLING_ADAPTER:-1}"
export INGRAIN_TSA_DIM="${INGRAIN_TSA_DIM:-128}"
# Hard cap on class names per training prompt (positives + sampled negatives).
export INGRAIN_PROMPT_MAXTOTAL="${INGRAIN_PROMPT_MAXTOTAL:-260}"
export INGRAIN_NO_PIN_MEMORY=1

# --- training data: LVIS, base categories only ------------------------------
# Training always uses LVISObservationSeqDataset. Point these at
# your LVIS copy; the annotation file must carry the LVIS `frequency` field so
# the training script can separate base (f/c) from novel (r).
export INGRAIN_ANN_FILE=${INGRAIN_ANN_FILE:-${INGRAIN_DATA_DIR:-data}/lvis_base_train.json}
export INGRAIN_H5_FILE=${INGRAIN_H5_FILE:-${INGRAIN_DATA_DIR:-data}/lvis_train_images.h5}

"$PY" -u train.py \
  ${INIT_WEIGHTS:+--init-weights "$INIT_WEIGHTS"} \
  ${RESUME:+--resume "$RESUME"} \
  --seq-len-schedule "${SEQSCHED:-1:4,6:5,10:6,14:7}" \
  --mem-warmup-epoch ${MEMWARM:-3} \
  --trajectory-state-prop --memory-attn-residual-scale 0.1 \
  --mem-aggregation "${MEM_AGG:-attention}" \
  --memory-depths "${MEM_DEPTHS:-3}" \
  --mem-ema-alpha "${MEM_EMA:-0.9}" \
  --decoupling-depth 3 --staged-decoder \
  --track-path --track-path-lr 3e-5 \
  --unfreeze-decoder-last-n 3 --train-track-ref-refine \
  --unfreeze-parts self_attn,cross_attn,ffn,norms \
  --e2e-assoc \
  --det-pool-mode dedup --dedup-bg-weight 1.0 --dedup-bg-warmup-steps 600 \
  --loss-cls-det-weight 0.05 \
  --lambda-det-init 0.1 --lambda-det-final 0.1 \
  --lambda-track-init 1.0 --lambda-track-final 2.0 \
  --freeze-all-but-track-path \
  --dn-anchor-denoise-p 0.4 --dn-anchor-object-scaled \
  --dn-anchor-cxy 0.12 --dn-anchor-wh 0.2 \
  --occlude-ratio 0.3 --motion-aug-strength 1.6 \
  --motion-aug-level coordinated --rot-amp 25.0 --scale-amp 0.35 \
  --gd-clip-norm 0.1 --clip-max-norm 1.0 \
  --seed ${SEED:-42} \
  --warmup-steps ${WARMUP:-200} --lr-decay cosine --lr-min 1e-5 \
  --lr 1e-4 --lr-gd 3e-5 \
  --num-frames 4 \
  --batch-size ${BS:-2} --accumulate-steps ${ACCUM:-4} --num-workers 3 \
  --epochs ${EPOCHS:-16} --diag-log \
  --max-opt-steps ${MAXSTEPS:-0} \
  --save-dir "$OUT" > "$LOG" 2>&1
echo "TRAIN_EXIT=$? (log: $LOG)"; tail -4 "$LOG"
