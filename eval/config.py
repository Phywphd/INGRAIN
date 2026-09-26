"""Evaluation configuration for INGRAIN on TAO."""

import os
import sys

# Make train.py importable so we can reuse MODEL_CFG
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_MMG = os.path.join(_ROOT, 'third_party', 'mmgroundingdino')
if _MMG not in sys.path:
    sys.path.insert(0, _MMG)

from train import MODEL_CFG, GD_CKPT  # noqa: E402,F401 — single source of truth


# === Data paths (repo-relative defaults; override any via env) ===
# Place the TAO annotations under ./data and the extracted frames at ./data/frames
# (a symlink to your real TAO frames root is fine), or point these envs elsewhere.
# $INGRAIN_DATA_DIR moves the whole set at once, matching train.py's default.
_DATA_DIR = os.environ.get('INGRAIN_DATA_DIR', os.path.join(_ROOT, 'data'))
TAO_VAL_ANN  = os.environ.get('INGRAIN_TAO_VAL_ANN',  os.path.join(_DATA_DIR, 'validation_ours_v1.json'))
TAO_TEST_ANN = os.environ.get('INGRAIN_TAO_TEST_ANN', os.path.join(_DATA_DIR, 'tao_test_burst_v1.json'))
TAO_IMG_ROOT = os.environ.get('INGRAIN_TAO_IMG_ROOT', os.path.join(_DATA_DIR, 'frames'))

# TETA scoring package. Use the copy vendored at the repo root so scoring works
# with no separate TETA install. Override with $TETA_PACKAGE_PATH.
TETA_PACKAGE_PATH = os.environ.get('TETA_PACKAGE_PATH') or _ROOT


# === Chunked text prediction ===
# Number of class names per text chunk.
CHUNKED_SIZE = 40


MAX_DETS_PER_FRAME = 300


# === Inference thresholds (global defaults; overridable) ===
# New-track threshold (cls score) — det slot must score >= this to spawn a track.
SCORE_THRESH            = 0.30
# miss_tolerance — frames a track can score low before death
MISS_TOLERANCE          = 5
FILTER_SCORE_THRESH     = 0.15   # tracks scoring below this age toward death


# === Association protocol (paper settings; fixed) ===
TOPK_CLS            = 1      # classes emitted per alive slot
IOU_THRESH          = 0.5    # IoU for inter-track NMS (and the optional newborn gate)
USE_INTER_TRACK_NMS = True   # of two active tracks with IoU > IOU_THRESH, retire the younger
USE_IOU_GATE        = False  # newborn-vs-active IoU gate (off in the paper protocol)
SOURCE_FILTER       = 'all'  # keep predictions from both det and track queries
