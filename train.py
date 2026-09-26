"""INGRAIN training entry point.

Trains the track path and the fusion stage, together with the multi-depth
trajectory memory and observation-guided propagation modules, on augmented
observation sequences built from LVIS still images. Training stays inside the
inherited query-text alignment rather than relearning it.

IMPORTANT: run from the repo root.

`scripts/train.sh` is the single source of truth for the recipe. It exports the
environment variables that construct the architecture and passes the full
flag set -- sequence-length curriculum, memory warm-up epoch, loss weights and
the per-group learning rates -- so it is what to run to reproduce the paper:

    bash scripts/train.sh                        # reproduce
    PY=/path/to/python bash scripts/train.sh     # pick an interpreter
    INGRAIN_HIDDEN_FUSE=0 bash scripts/train.sh  # ablate one component

The module-level constants below are argparse defaults for direct invocation
only; train.sh overrides every one of them explicitly. Running `train.py` by
hand therefore does NOT reproduce the paper unless you pass the same flags.

Key args (`--help` lists the full set):
    --epochs N           Total epochs
    --num-frames N       Observations per augmented sequence
    --batch-size N       Batch size; >1 uses batched backbone+encoder
    --accumulate-steps N Gradient accumulation; effective batch = bs x this
    --num-workers N      DataLoader workers
    --gd-ckpt PATH       Grounding DINO base checkpoint (default: swin-T)
    --init-weights PATH  Weights-only warm start (optimizer starts fresh)
    --resume PATH        Resume training (restores optimizer + epoch + step)
    --save-dir DIR       Checkpoint output directory
    --trajectory-state-prop  Enable trajectory-state propagation

Per-step loss numbers go to stdout (the log train.sh redirects with `>`);
there is no separate loss log file.
"""
import sys
import os
import argparse
import logging
import math
import random
import time
import fnmatch
from pathlib import Path

# --------------------------------------------------------------------------
# Path setup — must happen before any mmdet / ingrain import
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'mmgroundingdino'))
sys.path.insert(0, ROOT)

# --------------------------------------------------------------------------
# Register mmdet modules first
# --------------------------------------------------------------------------
from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)

# --------------------------------------------------------------------------
# Register INGRAIN modules
# --------------------------------------------------------------------------
import ingrain  # noqa: F401 — side-effect: registers INGRAINTracker etc.

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from mmengine.config import ConfigDict
from mmdet.registry import MODELS

from ingrain.datasets.lvis_observation_seq import LVISObservationSeqDataset, mot_collate_fn

# --------------------------------------------------------------------------
# SIGTERM graceful shutdown — avoids D-state worker orphans / dead GPU memory
# --------------------------------------------------------------------------
# SIGTERM sets a flag; the batch and epoch loops check it and break cleanly so
# DataLoader workers shut down normally. Use `kill -TERM <pid>`, not -9.
import signal as _signal
_TERM_REQUESTED = False

def _on_sigterm(signum, frame):
    global _TERM_REQUESTED
    if _TERM_REQUESTED:           # second SIGTERM → hard-exit (escape hatch)
        import os
        os._exit(130)
    _TERM_REQUESTED = True
    print(f"[SIGTERM] received signal {signum} — will stop after current step "
          f"(graceful: dataloader workers close cleanly, no D-state orphans). "
          f"Send TERM again to hard-exit.", flush=True)

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('ingrain_train')


# ==========================================================================
# Config
# ==========================================================================

# ---- Repo-relative roots (override any path via env for portability) -----
_HERE        = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR    = os.environ.get('INGRAIN_DATA_DIR', os.path.join(_HERE, 'data'))
_WEIGHTS_DIR = os.environ.get('INGRAIN_WEIGHTS_DIR', os.path.join(_HERE, 'weights'))

# ---- Data (training only; not read during inference) ---------------------
ANN_FILE     = os.environ.get('INGRAIN_ANN_FILE', os.path.join(_DATA_DIR, 'lvis_base_train.json'))
H5_FILE      = os.environ.get('INGRAIN_H5_FILE', os.path.join(_DATA_DIR, 'lvis_train_images.h5'))
CLASSES_FILE = os.environ.get('INGRAIN_CLASSES_FILE', os.path.join(_DATA_DIR, 'lvis_classes_v1.txt'))
NUM_FRAMES   = 4

# ---- GD base checkpoint (loaded at model build; override via env or --gd-ckpt)
# The public Grounding DINO Swin-T release from the OpenMMLab model zoo.
# Save it under weights/ with the name below, or point $INGRAIN_GD_CKPT /
# --gd-ckpt at wherever it already lives.
GD_CKPT = os.environ.get('INGRAIN_GD_CKPT',
                         os.path.join(_WEIGHTS_DIR, 'gd_swin_t.pth'))

# ---- Training ------------------------------------------------------------
LR               = 1e-4   # newly introduced modules (memory, propagation)
LR_GD_NATIVE     = 3e-5   # track path + fusion stage (GD-native params)
WEIGHT_DECAY     = 1e-4
SEED             = 42
EPOCHS           = 16
ACCUMULATE_STEPS = 4
CLIP_MAX_NORM    = 1.0  # global grad-clip max norm; --gd-clip-norm applies
                        # a separate, tighter clip to the GD-native weights.
AMP              = True
SAVE_DIR         = 'ckpt/ingrain_train'
SAVE_EVERY_N_EPOCHS  = 2
SAVE_FRAC_EPOCH      = float(os.environ.get('SAVE_FRAC_EPOCH', '0.1'))  # save ckpt every SAVE_FRAC_EPOCH of an epoch (env-overridable, e.g. 0.02)
LOG_EVERY_N_STEPS    = 10
NUM_WORKERS          = 4
BATCH_SIZE           = 2   # bs=1 for per-sample, bs>1 uses batched backbone+encoder
NUM_NEGATIVES        = 50  # negative classes sampled into each training
                           # prompt alongside the GT-positive classes.
PROMPT_MAX_TOTAL     = int(os.environ.get('INGRAIN_PROMPT_MAXTOTAL', 260) or 260)
                           # hard cap on class names per training prompt.

# ---- Parameter selection --------------------------------------------------
# Training is confined to the track decoding path; the patterns below are
# kept fixed, the classification head among them.
#
_GD_FREEZE_PATTERNS = [
    'backbone.*',
    'neck.*',
    'encoder.*',
    'language_model.*',
    'text_feat_map.*',
    'query_embedding.*',
    'level_embed',
    'level_embed.*',
    'memory_trans_fc.*',
    'memory_trans_norm.*',
    'dn_query_generator.*',
    # Decoder: every GD-native sublayer.
    'decoder.layers.*.self_attn.*',
    'decoder.layers.*.cross_attn.*',
    'decoder.layers.*.cross_attn_text.*',
    'decoder.layers.*.ffn.*',
    'decoder.layers.*.norms.*',
    'decoder.ref_point_head.*',
    'decoder.norm.*',
    # Bbox head kept fixed (preserves GD's cls calibration permanently)
    'bbox_head.*',
]

# The seed memory-attention is kept fixed here too; --unfreeze-parts and
# --freeze-all-but-track-path re-open the track decoding path and the newly
# introduced modules afterwards, which is where all trainable tracking lives.
FREEZE_PATTERNS = list(_GD_FREEZE_PATTERNS) + [
    'trajectory_memory.seed_mem_attn.*',
]

# ---- Model config ----------------------
MODEL_CFG = dict(
    type='INGRAINTracker',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    language_model=dict(
        type='BertModel',
        name='bert-base-uncased',
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4,
    ),
    encoder=dict(
        num_layers=6,
        num_cp=0,   # no activation checkpointing: the encoder is kept fixed
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        text_layer_cfg=dict(
            self_attn_cfg=dict(num_heads=4, embed_dims=256, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        ),
        fusion_layer_cfg=dict(
            v_dim=256, l_dim=256, embed_dim=1024, num_heads=4, init_values=1e-4,
        ),
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        num_isolation_layers=3,  # split depth n; both scripts pass
                                 # --decoupling-depth explicitly.
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0,
                               batch_first=True),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0,
                                     batch_first=True),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        post_norm_cfg=None,
    ),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20,
    ),
    bbox_head=dict(
        type='GroundingDINOHead',
        num_classes=256,
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(log_scale='auto', bias=True),
        loss_cls=dict(
            type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100),
    ),
    # INGRAIN-specific
    # Populated below from the CLI: loss_cls_det_weight, det_pool_mode,
    # dedup_bg_weight.
    track_loss_cfg=dict(),
    # Multi-depth trajectory memory: K = max_history states per track per
    # depth, aggregated by multi-head cross-attention.
    trajectory_memory_cfg=dict(
        embed_dims=256,
        max_history=16,
        ema_alpha=0.9,
    ),
    test_cfg=dict(max_per_img=300),
)


# ==========================================================================
# Model helpers
# ==========================================================================

def build_model() -> nn.Module:
    """Build INGRAINTracker from MODEL_CFG."""
    cfg = ConfigDict(MODEL_CFG)
    model = MODELS.build(cfg)
    return model


def load_gd_weights(model: nn.Module, ckpt_path: str) -> None:
    """Load GD pretrained weights into model (strict=False for new INGRAIN params)."""
    log.info(f"Loading GD weights: {os.path.basename(ckpt_path)}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model_state = model.state_dict()
    loaded = 0
    skipped_new = 0
    skipped_shape = 0
    for k in model_state:
        if k in state_dict:
            if model_state[k].shape == state_dict[k].shape:
                model_state[k] = state_dict[k]
                loaded += 1
            else:
                skipped_shape += 1
        else:
            skipped_new += 1

    model.load_state_dict(model_state)
    total = loaded + skipped_new + skipped_shape
    log.info(f"  Loaded: {loaded}/{total} ({100*loaded/total:.1f}%)")
    log.info(f"  New INGRAIN params (random init): {skipped_new}")
    if skipped_shape > 0:
        log.info(f"  Shape-mismatch skipped: {skipped_shape}")


def freeze_gd_params(model: nn.Module, patterns=None) -> int:
    """Disable grad on parameters matching the given patterns (default FREEZE_PATTERNS).

    Returns the count of affected parameters.
    """
    if patterns is None:
        patterns = FREEZE_PATTERNS
    frozen = 0
    for name, param in model.named_parameters():
        for pat in patterns:
            if fnmatch.fnmatch(name, pat):
                param.requires_grad_(False)
                frozen += 1
                break
    return frozen


def log_trainable_summary(model: nn.Module) -> None:
    """Print the trainable parameter count and the group names."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total/1e6:.2f}M total, {trainable/1e6:.2f}M trainable")
    groups = set()
    for name, p in model.named_parameters():
        if p.requires_grad:
            parts = name.split('.')
            # Show up to 3 levels
            prefix = '.'.join(parts[:min(3, len(parts))])
            groups.add(prefix)
    log.info(f"Trainable groups: {sorted(groups)}")


# ==========================================================================
# Text processing
# ==========================================================================

def load_class_names(classes_file: str) -> list:
    """Load LVIS class names from file. Returns list of 1203 names (0-indexed)."""
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def build_clip_text_prompt(gt_insts_clip: list,
                            class_names: list,
                            num_negatives: int = NUM_NEGATIVES,
                            max_total: int = PROMPT_MAX_TOTAL,
                            neg_candidates: list = None):
    """Build a per-clip text prompt with optional negative class sampling.

    The prompt carries the GT-positive class names; num_negatives adds that
    many class names sampled from the vocabulary, so each batch also scores
    classes that are absent from the image.

    Args:
        gt_insts_clip: list of per-frame gt dicts, each with a 'labels' tensor
            of global (0-indexed) class ids.
        class_names:   full list of class names (e.g. 1203 for LVIS).
        num_negatives: number of random negative classes to add on top of the
            positives. Set to 0 for positives-only prompts.
        max_total:     hard cap on the total number of classes in the prompt.
            Positives are NEVER dropped — if len(positives) already exceeds
            max_total, no negatives are added.

    Returns:
        text_prompt:     e.g. "person . car . dog ..."
        combined_labels: list of global label ids in the final prompt order
                         (positives + sampled negatives, shuffled).
        global_to_local: dict mapping global label id → local index in prompt.
            Only positives are guaranteed to be in this dict (and every GT
            label in gt_insts_clip is always a positive, so positive_maps
            look-ups never fail).
    """
    # 1) Collect positive labels — classes that actually appear in GT
    all_labels = set()
    for gi in gt_insts_clip:
        for lab in gi['labels'].tolist():
            all_labels.add(lab)
    pos_labels = sorted(all_labels)

    # 2) Sample negatives from the rest of the vocabulary
    neg_labels: list = []
    if num_negatives > 0:
        budget = max(0, max_total - len(pos_labels))
        num_neg_actual = min(num_negatives, budget)
        if num_neg_actual > 0:
            num_cls = len(class_names)
            pos_set = set(pos_labels)
            # Negatives come from the caller's restricted pool when one is
            # given (the base-only protocol supplies the base-category pool).
            _pool = neg_candidates if neg_candidates is not None \
                else range(num_cls)
            candidates = [c for c in _pool if c not in pos_set]
            if len(candidates) > num_neg_actual:
                neg_labels = random.sample(candidates, num_neg_actual)
            else:
                neg_labels = candidates

    # 3) Combine + shuffle to avoid positional bias in the prompt
    combined_labels = list(pos_labels) + list(neg_labels)
    random.shuffle(combined_labels)

    # 4) Build the local-index lookup used by precompute_clip_text_features
    global_to_local = {g: i for i, g in enumerate(combined_labels)}

    # 5) Build the prompt string
    prompt_class_names = [class_names[g] for g in combined_labels]
    text_prompt = ' . '.join(prompt_class_names)

    return text_prompt, combined_labels, global_to_local


def _restricted_neg_pool(args):
    """Negative-candidate pool for this batch, or None for full-vocab sampling.

    Holds the base-category ids (0-indexed cat_id - 1),
    so no novel class name is ever sampled as a negative.
    """
    pool = getattr(args, '_neg_pool', None) if args is not None else None
    return pool or None


def precompute_clip_text_features(model: nn.Module,
                                   text_prompt: str,
                                   gt_labels_list: list,
                                   global_to_local: dict,
                                   device: torch.device):
    """Tokenise the clip-level prompt and compute text features.

    Args:
        model:            INGRAINTracker
        text_prompt:      clip-level prompt (e.g. "person . car . dog")
        gt_labels_list:   list of per-frame label tensors (global 0-indexed)
        global_to_local:  dict mapping global label → local index
        device:           GPU device

    Returns:
        text_dict:       dict with 'embedded', 'text_token_mask', etc.
        positive_maps:   list of (N_t, max_text_len) float tensors per frame
        class_pmaps:     list of (max_text_len,) float tensors, ONE per class
                         in the prompt (local order). Used to aggregate the
                         per-token logits into one score per class.
    """
    tokenized, caption_string, tokens_positive, _ = \
        model.get_tokens_and_prompts(text_prompt, True)

    text_dict = model.language_model([caption_string])
    if model.text_feat_map is not None:
        text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])

    # Move text features to GPU
    for k, v in text_dict.items():
        if isinstance(v, torch.Tensor):
            text_dict[k] = v.to(device)

    # Build positive_maps using LOCAL label indices
    positive_maps = []
    for gt_labels in gt_labels_list:
        local_labels = [global_to_local[lab.item()] for lab in gt_labels]
        new_tp = [tokens_positive[loc] for loc in local_labels]
        _, pm = model.get_positive_map(tokenized, new_tp)
        positive_maps.append(pm.to(device).bool().float())

    # Per-class positive_maps: one (max_text_len,) map per prompt class.
    class_pmaps = []
    for cls_idx in range(len(tokens_positive)):
        _, pm = model.get_positive_map(tokenized, [tokens_positive[cls_idx]])
        class_pmaps.append(pm[0].to(device).bool().float())

    return text_dict, positive_maps, class_pmaps


def _delta_base_ckpt(args):
    """The checkpoint a delta save should point back at.

    --init-weights and --init-ckpt are both weights-only warm starts, and
    scripts/train.sh uses the former; both are read here so a delta checkpoint
    always records the base its trainable tensors are meant to merge onto.
    """
    return (getattr(args, 'init_ckpt', None)
            or getattr(args, 'init_weights', None))


def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    epoch: int,
                    step: int,
                    save_dir: str,
                    tag: str = '',
                    trainable_only: bool = False,
                    base_ckpt: str = None) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = f'epoch_{epoch}{tag}.pth'
    path = os.path.join(save_dir, fname)
    if trainable_only:
        # Delta checkpoint: ONLY trainable params, which is a small fraction
        # of the model, so a save costs megabytes instead of ~740MB.
        # 'base_ckpt' points to the full init ckpt; evaluate.py merges the two.
        keep = {n for n, q in model.named_parameters() if q.requires_grad}
        msd = {k: v for k, v in model.state_dict().items() if k in keep}
    else:
        msd = model.state_dict()
    payload = {
        'epoch': epoch,
        'step': step,
        'model': msd,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }
    if trainable_only and base_ckpt:
        payload['base_ckpt'] = os.path.abspath(base_ckpt)
    torch.save(payload, path)
    log.info(f"Checkpoint saved: {path}"
             + (f" (delta: {len(msd)} tensors)" if trainable_only else ''))


def load_checkpoint_resume(model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            scaler: GradScaler,
                            path: str) -> tuple[int, int]:
    """Load checkpoint for resuming.

    Returns (start_epoch, resumed_global_step). The resumed_global_step is the
    absolute global step counter at checkpoint save time; callers must restore
    it into the running ``global_step`` list so that LR scheduler, max-opt-steps
    stop condition, and step-based logging all continue from the saved
    position. Without this, warmup/cosine decay restarts from 0 even though
    model/optimizer state is already past warmup.
    """
    log.info(f"Resuming from: {path}")
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    # strict=False: tolerate new modules added after the ckpt was saved.
    # Newly added params keep their fresh init (zero-init heads → strict
    # no-op at start).
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
    if missing:
        log.info(f"  Missing keys (kept at init): {len(missing)}")
        for k in missing[:6]:
            log.info(f"    + {k}")
        if len(missing) > 6:
            log.info(f"    ... and {len(missing) - 6} more")
    if unexpected:
        log.info(f"  Unexpected keys (ignored): {len(unexpected)}")
        for k in unexpected[:6]:
            log.info(f"    - {k}")
    # Try to load optimizer state, but skip if param-group structure changed
    # (new modules → new optimizer groups → state_dict mismatch).
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        log.info("  Optimizer state restored")
    except (ValueError, RuntimeError, KeyError) as e:
        log.warning(f"  Optimizer state NOT restored ({type(e).__name__}: "
                    f"{str(e)[:80]}); fresh optimizer state for new params.")
    scaler.load_state_dict(ckpt['scaler'])
    start_epoch = ckpt['epoch'] + 1
    resumed_step = int(ckpt.get('step', 0))
    log.info(f"  Resumed: epoch {ckpt['epoch']}, step {resumed_step}")
    return start_epoch, resumed_step


# ==========================================================================
# Training loop
# ==========================================================================


def _compute_loss_weights(epoch: int, total_epochs: int, args,
                            opt_step: int = 0,
                            steps_per_epoch: int = 1) -> dict:
    """Annealed det/track loss weights for the training schedule."""
    half = max(total_epochs // 2, 1)
    progress = min(max(epoch - 1, 0) / half, 1.0)
    return {
        'det': args.lambda_det_init + (args.lambda_det_final - args.lambda_det_init) * progress,
        'track_emit': (args.lambda_track_init
                       + (args.lambda_track_final
                          - args.lambda_track_init) * progress),
        # --e2e-drop-cls-track removes loss_cls_track from the backward; the
        # released recipe leaves it OFF, so the track slice keeps its per-frame
        # class supervision. Ablation only.
        'e2e_drop_cls_track': bool(getattr(args, 'e2e_drop_cls_track', False)),
    }


_DET_LOSS_KEYS = (
    'loss_cls_det', 'loss_box_det',
    'enc_loss_cls', 'enc_loss_box',
)
_TRACK_EMIT_LOSS_KEYS = (
    # Q_tr box/cls emission losses. Setting this weight to zero keeps the
    # association losses alive on their own.
    'loss_cls_track', 'loss_box_track',
)


def _apply_loss_weights(losses: dict, weights: dict) -> torch.Tensor:
    """Aggregate frame_losses dict into a single scalar with annealed
    weights. Unknown keys default to weight 1.0.
    """
    total = None
    for k, v in losses.items():
        if not (isinstance(v, torch.Tensor) and v.requires_grad):
            continue
        if k == 'loss_cls_track' and weights.get('e2e_drop_cls_track', False):
            # Class carried from birth — do not supervise track-slice cls.
            continue
        if k in _DET_LOSS_KEYS:
            w = weights['det']
        elif k in _TRACK_EMIT_LOSS_KEYS:
            w = weights['track_emit']
        else:
            w = 1.0
        weighted = w * v
        total = weighted if total is None else (total + weighted)
    return total


def _compute_lr_factor(opt_step: int,
                        warmup_steps: int,
                        total_opt_steps: int,
                        decay: str,
                        lr_min_ratio: float = 0.1) -> float:
    """Linear warmup + (optionally) cosine decay multiplier on base LR.

    Args:
        opt_step: current optimizer step (0-indexed; called BEFORE step).
        warmup_steps: linear ramp from 0 → 1 over these many steps.
        total_opt_steps: total opt steps in the run (for cosine).
        decay: 'none' or 'cosine'.
        lr_min_ratio: minimum LR as fraction of base (cosine floor).
    """
    if warmup_steps > 0 and opt_step < warmup_steps:
        return float(opt_step) / float(max(warmup_steps, 1))
    if decay == 'cosine' and total_opt_steps > warmup_steps:
        progress = (opt_step - warmup_steps) / max(
            total_opt_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress))
    return 1.0


def train_one_epoch(model: nn.Module,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    scaler: GradScaler,
                    epoch: int,
                    class_names: list,
                    device: torch.device,
                    global_step_ref: list,
                    save_dir: str = SAVE_DIR,
                    num_negatives: int = NUM_NEGATIVES,
                    clip_max_norm: float = CLIP_MAX_NORM,
                    args=None,
                    total_epochs: int = 1,
                    accumulate_steps: int = 8,
                    base_lrs: list = None,
                    total_opt_steps: int = 0) -> bool:
    """Train for one epoch.

    Args:
        class_names:     list of 1203 LVIS class names (0-indexed)
        global_step_ref: single-element list holding the global step counter
                         (mutable so the caller sees updates after the call).
        num_negatives:   number of random negative classes to sample into
                         each batch-level prompt (0 = positives only).
        clip_max_norm:   gradient clipping max norm.
    """
    model.train()
    optimizer.zero_grad()

    total_steps_in_epoch = len(dataloader)
    frac_save_interval = max(1, int(total_steps_in_epoch * SAVE_FRAC_EPOCH))

    step_in_epoch = 0
    t0 = time.time()
    reached_max_opt_steps = False
    max_opt_steps = int(getattr(args, 'max_opt_steps', 0) or 0)

    for data_dict in dataloader:
        if _TERM_REQUESTED:
            log.info("[SIGTERM] stopping epoch — exiting batch loop cleanly "
                     "(dataloader workers will close via __del__).")
            break
        if (max_opt_steps > 0
                and global_step_ref[0] // max(accumulate_steps, 1)
                >= max_opt_steps):
            reached_max_opt_steps = True
            break

        is_batched = isinstance(data_dict, dict) and 'batch_size' in data_dict

        if not is_batched:
            # ── bs=1 path (original) ──────────────────────────────
            imgs = data_dict['imgs']
            gt_insts = data_dict['gt_instances']
            img_shape = data_dict['img_shape']
            num_frames = len(imgs)

            imgs_gpu = [img.unsqueeze(0).to(device) for img in imgs]

            text_prompt, clip_labels, global_to_local = \
                build_clip_text_prompt(
                    gt_insts, class_names, num_negatives=num_negatives,
                    max_total=PROMPT_MAX_TOTAL,
                    neg_candidates=_restricted_neg_pool(args))
            all_labels = [gi['labels'] for gi in gt_insts]
            with torch.no_grad():
                text_dict, positive_maps, prompt_class_pmaps = \
                    precompute_clip_text_features(
                        model, text_prompt, all_labels, global_to_local, device)

            gt_instances_for_model = []
            for t in range(num_frames):
                boxes = gt_insts[t]['boxes'].to(device)
                labels = gt_insts[t]['labels'].to(device)
                obj_ids = gt_insts[t]['obj_ids'].to(device)
                pm = positive_maps[t]
                img_h, img_w = imgs_gpu[t].shape[-2:]
                bx1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
                by1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
                bx2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
                by2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
                boxes_xyxy = torch.stack([bx1, by1, bx2, by2], dim=-1)
                gt_instances_for_model.append({
                    'boxes': boxes, 'boxes_xyxy': boxes_xyxy,
                    'labels': labels,
                    'obj_ids': obj_ids, 'positive_maps': pm,
                })

            text_token_mask = text_dict['text_token_mask']
            data_for_model = {
                'imgs': imgs_gpu,
                'gt_instances': gt_instances_for_model,
                'text_dict': text_dict,
                'text_token_mask': text_token_mask,
                'img_shape': img_shape,
                'prompt_class_pmaps': prompt_class_pmaps,
            }
            forward_fn = model.forward_train_mot

        else:
            # ── bs>1 path (batched backbone+encoder) ──────────────
            bs = data_dict['batch_size']
            num_frames = len(data_dict['imgs'])
            img_shapes = data_dict['img_shapes']

            # Move batched images to device
            imgs_gpu = [frame_batch.to(device) for frame_batch in data_dict['imgs']]

            # Build union text prompt across ALL samples in the batch
            all_gt_flat = []
            for i in range(bs):
                for t in range(num_frames):
                    all_gt_flat.append(data_dict['gt_instances'][i][t])
            text_prompt, clip_labels, global_to_local = \
                build_clip_text_prompt(
                    all_gt_flat, class_names, num_negatives=num_negatives,
                    max_total=PROMPT_MAX_TOTAL,
                    neg_candidates=_restricted_neg_pool(args))

            # Collect all per-sample per-frame labels for positive_map
            all_labels_flat = [gi['labels'] for gi in all_gt_flat]
            with torch.no_grad():
                text_dict, positive_maps_flat, prompt_class_pmaps = \
                    precompute_clip_text_features(
                        model, text_prompt, all_labels_flat,
                        global_to_local, device)

            # Reshape positive_maps: flat → [sample_i][frame_t]
            idx = 0
            gt_all = []  # bs x num_frames x dict
            for i in range(bs):
                sample_frames = []
                for t in range(num_frames):
                    gi = data_dict['gt_instances'][i][t]
                    boxes = gi['boxes'].to(device)
                    labels = gi['labels'].to(device)
                    obj_ids = gi['obj_ids'].to(device)
                    pm = positive_maps_flat[idx]
                    idx += 1
                    img_h, img_w = imgs_gpu[t].shape[-2:]
                    bx1 = (boxes[:, 0] - boxes[:, 2] / 2) * img_w
                    by1 = (boxes[:, 1] - boxes[:, 3] / 2) * img_h
                    bx2 = (boxes[:, 0] + boxes[:, 2] / 2) * img_w
                    by2 = (boxes[:, 1] + boxes[:, 3] / 2) * img_h
                    boxes_xyxy = torch.stack([bx1, by1, bx2, by2], dim=-1)
                    sample_frames.append({
                        'boxes': boxes, 'boxes_xyxy': boxes_xyxy,
                        'labels': labels,
                        'obj_ids': obj_ids, 'positive_maps': pm,
                    })
                gt_all.append(sample_frames)

            text_token_mask = text_dict['text_token_mask']

            data_for_model = {
                'imgs': imgs_gpu,
                'gt_instances': gt_all,
                'text_dict': text_dict,
                'text_token_mask': text_token_mask,
                'img_shapes': img_shapes,
                'batch_size': bs,
                'prompt_class_pmaps': prompt_class_pmaps,
            }
            forward_fn = model.forward_train_mot_batched

        # Sync model._global_step before forward so any step-aware feature
        # (curricula, future schedulers) sees the right step.
        if hasattr(model, '_global_step'):
            model._global_step = int(global_step_ref[0])

        # ---- Forward + loss ----
        with autocast('cuda', dtype=torch.float16, enabled=AMP):
            losses = forward_fn(data_for_model)

            if not losses:
                log.warning(f"Empty loss dict at step {global_step_ref[0]}")
                continue

            # Apply the annealed det/track weights when args is provided.
            if args is not None and total_epochs > 0:
                _opt_step_now = step_in_epoch // max(accumulate_steps, 1)
                _opt_steps_per_epoch = max(
                    int(total_steps_in_epoch / max(accumulate_steps, 1)),
                    1)
                _w = _compute_loss_weights(
                    epoch, total_epochs, args,
                    opt_step=_opt_step_now,
                    steps_per_epoch=_opt_steps_per_epoch)
                if hasattr(model, 'criterion') and model.criterion is not None:
                    # Ramp dedup_bg_weight 0->target over warmup, keyed off the
                    # global optimizer step (like the LR schedule).
                    _mbw_warm = int(getattr(args, 'dedup_bg_warmup_steps', 0) or 0)
                    if _mbw_warm > 0:
                        _mbw_tgt = float(getattr(args, 'dedup_bg_weight', 0.0) or 0.0)
                        _mbw_step = global_step_ref[0] // max(accumulate_steps, 1)
                        _mbw_frac = min(1.0, _mbw_step / _mbw_warm)
                        model.criterion.dedup_bg_weight = _mbw_tgt * _mbw_frac
                total_loss = _apply_loss_weights(losses, _w)
            else:
                total_loss = sum(v for v in losses.values()
                                 if isinstance(v, torch.Tensor)
                                 and v.requires_grad)
            if not isinstance(total_loss, torch.Tensor):
                log.warning(f"No differentiable losses at step {global_step_ref[0]}")
                continue
            total_loss_scaled = total_loss / accumulate_steps

        scaler.scale(total_loss_scaled).backward()

        step_in_epoch += 1
        global_step_ref[0] += 1

        # ---- Gradient accumulation step ----
        if step_in_epoch % accumulate_steps == 0:
            opt_step_global = global_step_ref[0] // accumulate_steps
            # Linear warmup + optional cosine decay.
            if args is not None and base_lrs is not None:
                lr_factor = _compute_lr_factor(
                    opt_step=opt_step_global,
                    warmup_steps=args.warmup_steps,
                    total_opt_steps=total_opt_steps,
                    decay=args.lr_decay,
                    lr_min_ratio=(args.lr_min / max(args.lr, 1e-12)),
                )
                for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                    pg['lr'] = base_lr * lr_factor
            scaler.unscale_(optimizer)
            # Tight per-group clip on the trainable GD decoder weights
            # (standard GD finetune clips 0.1; the loose global 1.0 stays for
            # fresh-init INGRAIN modules which need bigger steps). Cached list.
            _gdc = (float(getattr(args, 'gd_clip_norm', 0.0) or 0.0)
                    if args is not None else 0.0)
            if _gdc > 0:
                _gdp = getattr(model, '_gd_clip_params', None)
                if _gdp is None:
                    _gdp = [p for n, p in model.named_parameters()
                            if p.requires_grad
                            and (n.startswith('decoder.layers.')
                                 or n.startswith('decoder.track_dec_'))]
                    model._gd_clip_params = _gdp
                if _gdp:
                    clip_grad_norm_(_gdp, _gdc)
            grad_norm = clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                clip_max_norm)
            # Per-module grad norms. Must read grads BEFORE zero_grad;
            # a zero norm means the module received no gradient.
            if getattr(model, '_diag', None) is not None:
                from ingrain.models.diag import module_grad_norm as _mgn
                # reg_branches grad norm (last-2 trainable), covering the
                # trainable iterative box relocation. Expected to be nonzero.
                model._diag.add('grad_reg', _mgn(getattr(
                    getattr(model, 'bbox_head', None), 'reg_branches', None)))
                # The track path's trainable reg (track_dec_reg) is the
                # track-box relocator (the main reg_branches is kept fixed, so its
                # norm reads 0); it is trained through the next-frame
                # loss_box_track.
                model._diag.add('grad_copyreg', _mgn(getattr(
                    getattr(model, 'decoder', None), 'track_dec_reg', None)))
                # Gradient reaching the det↔track fusion stage
                # (trainable last-layer self_attn).
                try:
                    _dec_sa = model.decoder.layers[-1].self_attn
                except (AttributeError, IndexError, TypeError):
                    _dec_sa = None
                model._diag.add('grad_dec_sa', _mgn(_dec_sa))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            opt_step = step_in_epoch // accumulate_steps
            if opt_step % LOG_EVERY_N_STEPS == 0:
                elapsed = time.time() - t0
                primary_losses = {
                    k: f'{v.item():.4f}'
                    for k, v in losses.items()
                    if isinstance(v, torch.Tensor) and not k.startswith('d')
                }
                memory_attn_str = ''
                weight_norm_str = ''
                try:
                    tmem = getattr(model, 'trajectory_memory', None)
                    if tmem is not None:
                        stats = getattr(
                            tmem, '_last_memory_attention_stats', None)
                        if stats:
                            memory_attn_str = (
                                f" | mem_attn: n={stats['n']}"
                                f" Δ={stats['delta_mean']:.4f}"
                                f" rel={stats['rel_delta_mean']:.4f}"
                                f" mem_fill={stats['memory_bank_filled']}"
                                f" tok/trk={stats['mean_mem_tokens_per_track']:.2f}")
                        # Weight-norm diag (seed_mem_attn out_proj) so we
                        # can confirm the zero-init residual paths are actually
                        # leaving zero. Cheap per-step probe.
                        ma = getattr(tmem, 'seed_mem_attn', None)
                        wn_parts = []
                        if ma is not None and hasattr(ma, 'attn'):
                            wn_parts.append(
                                f"ma={ma.attn.out_proj.weight.norm().item():.3f}")
                        # Interleaved memory attention (depths 0/1/2): per-depth out_proj
                        # norm so we can watch each depth activate independently.
                        _ilma = getattr(tmem, 'depth_mem_attn', None)
                        if _ilma is not None and len(_ilma) > 0:
                            _iln = "/".join(
                                f"{m.attn.out_proj.weight.norm().item():.3f}"
                                for m in _ilma if hasattr(m, 'attn'))
                            if _iln:
                                wn_parts.append(f"ilma={_iln}")
                        if wn_parts:
                            weight_norm_str = (
                                " | out_proj.W.norm: " + " ".join(wn_parts))
                except Exception:
                    pass
                log.info(
                    f"Epoch {epoch:3d} | "
                    f"opt_step {opt_step:5d} | "
                    f"gstep {global_step_ref[0]:6d} | "
                    f"loss {total_loss.item():.4f} | "
                    f"gnorm {grad_norm:.3f} | "
                    f"t {elapsed:.1f}s"
                    f"{memory_attn_str}{weight_norm_str} | "
                    f"{primary_losses}"
                )
                # Per-module diagnostic line (mean over the log interval).
                if getattr(model, '_diag', None) is not None:
                    log.info(model._diag.format_line(opt_step))
                    model._diag.reset()
                t0 = time.time()

            if max_opt_steps > 0 and opt_step_global >= max_opt_steps:
                log.info(f"Reached --max-opt-steps={max_opt_steps}; "
                         "stopping after current optimizer step.")
                reached_max_opt_steps = True
                break

        # Fractional epoch checkpoint (every SAVE_FRAC_EPOCH of the epoch)
        if step_in_epoch > 0 and step_in_epoch % frac_save_interval == 0:
            frac = step_in_epoch / total_steps_in_epoch
            save_checkpoint(model, optimizer, scaler, epoch,
                            global_step_ref[0], save_dir,
                            tag=f'_frac{frac:.3f}',
                            trainable_only=bool(getattr(
                                args, 'save_trainable_only', False)),
                            base_ckpt=_delta_base_ckpt(args))

    # Flush any leftover gradients at epoch end
    remainder = step_in_epoch % accumulate_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        _gdc2 = (float(getattr(args, 'gd_clip_norm', 0.0) or 0.0)
                 if args is not None else 0.0)
        if _gdc2 > 0 and getattr(model, '_gd_clip_params', None):
            clip_grad_norm_(model._gd_clip_params, _gdc2)
        clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return reached_max_opt_steps


# ==========================================================================
# Main
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='INGRAIN training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--init-weights', type=str, default=None,
                        help='Weights-ONLY warm start: load model state_dict '
                             '(strict=False) from this ckpt; optimizer/'
                             'scaler/epoch/step start fresh (vs --resume '
                             'which restores all four).')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed for python/numpy/torch and the '
                             'DataLoader workers (default: %(default)s).')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--lr', type=float, default=LR,
                        help=f'Learning rate for INGRAIN-new params (default: {LR})')
    parser.add_argument('--lr-gd', type=float, default=LR_GD_NATIVE,
                        help=f'Learning rate for the trainable GD-native params (default: {LR_GD_NATIVE})')
    parser.add_argument('--track-path', action='store_true',
                        help='Build the track path of the decoupling stage: a '
                             'SEPARATE, FULLY TRAINABLE copy of the detection-path '
                             'decoder layers, dedicated to the TRACK query (the '
                             'paper\'s "copied track path"). The detection path '
                             'is kept fixed (preserves '
                             'detection + open-vocabulary alignment); track is '
                             're-decoded through the track path (track reads det + '
                             'trainable ref_point_head + deformable + reg; '
                             'cross_attn_text + cls head are kept as pretrained). Built '
                             'AFTER parameter selection so it warm-starts from the loaded '
                             'decoder weights and stays trainable.')
    parser.add_argument('--track-path-lr', type=float, default=2e-5,
                        help='Dedicated LR for the track-path params of the '
                             'decoupling stage (deformable; keep gentle, NOT lr=1e-4).')
    parser.add_argument('--staged-decoder', action='store_true',
                        help='Stage the decoder: give the track path ONLY the first '
                             'n=--decoupling-depth layers (the decoupling stage, '
                             'where the detection path stays fixed); the remaining '
                             '(num_layers-n) main decoder layers are shared by the '
                             'detection and track paths (the '
                             'FUSION STAGE, det↔track interaction). Requires '
                             '--decoupling-depth n and --unfreeze-decoder-last-n '
                             '(num_layers-n) to make the fusion stage trainable.')
    parser.add_argument('--track-path-parts', type=str, default='',
                        help='Narrow selection for the track path: '
                             'comma-list of {self_attn,cross_attn,ffn,norms,rph,reg} '
                             'kept TRAINABLE on the track path; the rest stay FIXED '
                             '(cross_attn_text always kept as pretrained). Empty = the whole '
                             'track path is trainable. E.g. "self_attn" = train only '
                             "the track path's self-attn (track reads det), keep "
                             'the image cross-attention, reg and rph fixed.')
    parser.add_argument('--unfreeze-decoder-last-n', type=int, default=0,
                        help='After parameter selection, re-enable grad on the '
                             'LAST N decoder layers (e.g. 2 → layers 4,5). They '
                             'route to the GD-native param group (--lr-gd, '
                             '3e-5 in the released recipe). '
                             'bbox_head stays fixed. 0 = the whole decoder stays fixed.')
    parser.add_argument('--unfreeze-track-text-ca', action='store_true',
                        help='Release the track path\'s text cross-attention '
                             '(kept as pretrained by default). Ablation only.')
    parser.add_argument('--unfreeze-cls-branches', action='store_true',
                        help='Release the region-text contrastive head '
                             '(bbox_head.cls_branches, kept fixed in the reported '
                             'recipe). Ablation only.')
    parser.add_argument('--unfreeze-parts', type=str,
                        default='self_attn',
                        help='Which GD-native decoder-layer components to '
                             'unfreeze with --unfreeze-decoder-last-n '
                             '(comma-sep). DEFAULT self_attn = the det↔track '
                             'FUSION STAGE ONLY. Adapting the deformable image '
                             'cross-attention, FFN and norms as well must be '
                             'requested explicitly; the released recipe does so '
                             'with --unfreeze-parts self_attn,cross_attn,ffn,norms.')
    parser.add_argument('--num-workers', type=int, default=NUM_WORKERS,
                        help=f'DataLoader num_workers (default: {NUM_WORKERS})')
    parser.add_argument('--save-dir', type=str, default=SAVE_DIR,
                        help=f'Checkpoint directory (default: {SAVE_DIR})')
    parser.add_argument('--num-frames', type=int, default=NUM_FRAMES,
                        help=f'Observations per augmented sequence (default: {NUM_FRAMES})')
    parser.add_argument('--init-ckpt', type=str, default=None,
                        help='INGRAIN checkpoint to init model weights from (no optimizer restore)')
    parser.add_argument('--gd-ckpt', type=str, default=GD_CKPT,
                        help=f'Grounding DINO pretrained checkpoint to load as the '
                             f'base (always loaded, even when --init-ckpt is passed). '
                             f'Default: {os.path.basename(GD_CKPT)}')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE}). >1 uses batched backbone+encoder')
    parser.add_argument('--decoupling-depth', type=int, default=None,
                        help='Number of decoder layers (from layer 0) where det↔track '
                             'self-attention is blocked. Range: 0-6. '
                             'Default: use MODEL_CFG value (3).')
    parser.add_argument('--clip-max-norm', type=float, default=CLIP_MAX_NORM,
                        help=f'Gradient clipping max norm (default: {CLIP_MAX_NORM}). '
                             f'The released recipe uses {CLIP_MAX_NORM}.')
    parser.add_argument('--iso-directional', action='store_true',
                        help='Directional isolation: det+DN never read track in '
                             'ANY layer (det slice stays protected) while '
                             'TRACK CAN READ DET, including in the fusion '
                             'stage, which this therefore makes '
                             'unidirectional. Eval MUST pass the same flag.')
    parser.add_argument('--gd-clip-norm', type=float, default=0.0,
                        help='Separate tighter grad clip for the trainable '
                             'GD-native decoder weights (decoder.layers.*), '
                             'applied BEFORE the global --clip-max-norm. '
                             '0 = off; the released recipe uses 0.1.')
    parser.add_argument('--loss-cls-det-weight', type=float, default=1.0,
                        help='Weight on det Hungarian-matched cls/bbox '
                             'losses. Default: 1.0 (without this, '
                             'det query path is unsupervised and only frame-0 '
                             'detection is learned).')
    parser.add_argument('--trajectory-state-prop', action='store_true',
                        help='Enable trajectory-state propagation: track '
                             'queries are updated by multi-head attention over '
                             'the multi-depth trajectory memory before they are '
                             'decoded. Both scripts/train.sh and scripts/eval.sh '
                             'pass this; the two must agree.')
    parser.add_argument('--memory-attn-residual-scale', type=float, default=0.1,
                        help='Residual scale for trajectory memory attention.')
    parser.add_argument('--memory-attn-dropout', type=float, default=0.0,
                        help='Dropout for trajectory memory attention.')
    # ── end-to-end association ──────────────────────────────────────────────
    parser.add_argument('--e2e-assoc', action='store_true',
                        help='End-to-end association: the propagated '
                             'track-query content is shaped in-graph by the '
                             'box/cls losses, so slot continuity IS the '
                             'association and no matcher is needed.')
    parser.add_argument('--e2e-drop-cls-track', action='store_true',
                        help='Drop loss_cls_track from the backward '
                             'under e2e (track class then comes ONLY from eval '
                             'birth-carry). OFF by default = KEEP cls_track (the '
                             'track query stays class-aware). Ablation only.')
    parser.add_argument('--save-trainable-only', action='store_true',
                        help='Save delta checkpoints containing ONLY trainable '
                             'params + a base_ckpt pointer (eval merges).')
    parser.add_argument('--dn-anchor-denoise-p', type=float, default=0.0,
                        help='Prob of replacing a matched track row '
                             'carry-forward anchor with a jittered GT box. '
                             '0=off; the released recipe uses 0.4. Requires '
                             '--train-track-ref-refine and isolation<6.')
    parser.add_argument('--dn-anchor-cxy', type=float, default=0.12,
                        help='DN anchor jitter: cx,cy shift U(-x,x). FRAME-fraction '
                             'unless --dn-anchor-object-scaled (then OBJECT-w/h frac).')
    parser.add_argument('--dn-anchor-wh', type=float, default=0.2,
                        help='DN anchor jitter: w,h scale *(1±x).')
    parser.add_argument('--dn-anchor-object-scaled', action='store_true',
                        help='Scale the cx,cy jitter by the '
                             "object's own w/h (|dx|<=cxy*w) instead of a "
                             'fraction of the frame. The released recipe passes '
                             'this and keeps the default --dn-anchor-cxy 0.12.')
    parser.add_argument('--dedup-bg-warmup-steps', type=int, default=0,
                        help='Linearly ramp dedup_bg_weight from 0 '
                             'to its target over the first N opt-steps. '
                             '0 = constant. The released recipe uses 600.')
    parser.add_argument('--occlude-ratio', type=float, default=0.4,
                        help='Fraction of objects given a hidden-frame '
                             'schedule. The released recipe uses 0.3.')
    parser.add_argument('--freeze-all-but-track-path', action='store_true',
                        help='Final word on parameter selection: everything stays fixed except the '
                             'track decoding path selected by '
                             '--unfreeze-decoder-last-n / --unfreeze-parts and '
                             'the newly introduced modules (trajectory memory, '
                             'track adapters). Overrides the earlier selection '
                             'passes, so the trainable summary is the truth.')
    parser.add_argument('--diag-log', action='store_true',
                        help='Enable the per-module diagnostic line: '
                             'track-query box IoU percentiles, '
                             'recover-missed, duplicate-fire, track counts, '
                             'matched-det score, and per-group grad norms.')
    parser.add_argument('--det-pool-mode', choices=['remaining', 'all', 'dedup'],
                        default='remaining',
                        help='GT pool used by det-query Hungarian loss. '
                             'remaining = newborn/untracked only; all = all '
                             'visible GTs; dedup = remaining + explicit strong '
                             'focal-negative on det queries overlapping a '
                             'tracked GT.')
    parser.add_argument('--dedup-bg-weight', type=float, default=2.0,
                        help='Weight of the explicit focal-negative on '
                             'duplicate-of-track det queries (normalized by '
                             'num_pos_det, i.e. the same scale as the main det '
                             'cls loss).')
    # Annealed loss weights.
    parser.add_argument('--lambda-det-init', type=float, default=1.0,
                        help='Detection-branch loss weight at the start of '
                             'training; anneals to --lambda-det-final over '
                             'the first half of training. The released '
                             'recipe holds it constant at 0.1.')
    parser.add_argument('--lambda-det-final', type=float, default=0.5,
                        help='Final detection-branch loss weight (released '
                             'recipe: 0.1, i.e. constant).')
    parser.add_argument('--lambda-track-init', type=float, default=1.0,
                        help='Q_tr bbox/cls emission loss weight at the start '
                             'of training; ramps to --lambda-track-final. '
                             'The released recipe uses 1.0 -> 2.0.')
    parser.add_argument('--lambda-track-final', type=float, default=2.0,
                        help='Final Q_tr bbox/cls emission loss weight '
                             '(released recipe: 2.0).')
    # ---- LR schedule + accumulation (perf tuning) ----
    parser.add_argument('--accumulate-steps', type=int, default=None,
                        help='Override ACCUMULATE_STEPS. effective_batch = '
                             'batch_size * accumulate_steps. Default: '
                             f'{ACCUMULATE_STEPS}.')
    parser.add_argument('--warmup-steps', type=int, default=0,
                        help='Linear LR warmup over the first N opt-steps. '
                             'The released recipe uses 200.')
    parser.add_argument('--lr-decay', choices=['none', 'cosine'],
                        default='none',
                        help='LR decay schedule after warmup. cosine decays '
                             'to lr_min over the remaining opt-steps.')
    parser.add_argument('--lr-min', type=float, default=2e-5,
                        help='Cosine floor, interpreted relative to --lr: every '
                             'group decays to (lr-min / lr) x its own initial '
                             'LR. The released recipe passes 1e-5 with '
                             '--lr 1e-4, i.e. a 0.1x floor for all groups.')
    # ----- track-reference refinement and curriculum -----
    parser.add_argument('--train-track-ref-refine', action='store_true',
                        help='Relocation training: train the last-N '
                             '(=--unfreeze-decoder-last-n) decoder reg_branches '
                             'AND keep gradient on the TRACK slice of the '
                             'inter-layer reference in the fusion stage, so '
                             'loss_box_track trains iterative track-query '
                             'relocation. cls '
                             'heads stay fixed (open-vocab preserved).')
    parser.add_argument('--motion-aug-strength', type=float, default=1.0,
                        help='Scales the coordinated-translation '
                             'magnitude applied per clip (base ±5%% of '
                             'short side → ±5%%*strength). 1.0=default; '
                             '1.2=+20%% motion (still '
                             'in deformable refine range, pushes harder).')
    parser.add_argument('--motion-aug-level', type=str, default='basic',
                        choices=['basic', 'coordinated'],
                        help="Clip motion mode. 'basic': rigid linear "
                             "translation. 'coordinated': smooth "
                             "translation trajectory + the smooth rotation/"
                             "scale ramps below.")
    parser.add_argument('--rot-amp', type=float, default=0.0,
                        help='coordinated mode: max TOTAL rotation over a clip in '
                             'degrees, smoothly ramped 0 -> ±θ (per-clip draw '
                             'up to this amplitude). 0 = off.')
    parser.add_argument('--scale-amp', type=float, default=0.0,
                        help='coordinated mode: max TOTAL size change over a clip, '
                             'smoothly ramped 1 -> 1±a (per-clip draw up to '
                             'this amplitude). 0 = off.')
    # --- run control, prompt construction, memory aggregation --------------
    parser.add_argument('--max-opt-steps', type=int, default=0,
                        help='Debug/diagnostic: stop training after this many '
                             'optimizer steps. 0 = disabled.')
    parser.add_argument('--mem-aggregation', type=str, default='attention',
                        choices=['attention', 'mean', 'ema'],
                        help='How the K stored trajectory states are collapsed '
                             'into one context vector. The modes share '
                             'normalisation, residual scale, FFN and a '
                             'zero-init exit, so only the aggregator differs. '
                             'Default attention = the full model.')
    parser.add_argument('--mem-ema-alpha', type=float, default=0.9,
                        help='Decay for --mem-aggregation ema: a state of age '
                             'k gets weight alpha^k, newest first. Ignored by '
                             'the other modes.')
    parser.add_argument('--memory-depths', type=int, default=3,
                        help='Number of track-path memory injection points, '
                             'i.e. depths d = 1..N; the seed is d = 0. Default '
                             '3 gives d = {0,1,2,3}, one after each '
                             'decoupling-stage track layer. 0 gives d = {0}. N = 4 '
                             'adds '
                             'an injection after the FIRST fusion-stage layer '
                             '(d = 4); beyond that there is no call site.')
    parser.add_argument('--track-ref-grad', action='store_true',
                        help='Let the track slice\'s reference_points carry '
                             'gradient through the fusion stage, so the box '
                             'head learns to predict an offset from the '
                             'propagated reference. Unlike '
                             '--train-track-ref-refine this touches NO '
                             'detection parameter — use it when the pretrained '
                             'detector must stay fixed.')
    parser.add_argument('--seq-len-schedule', type=str, default='',
                        help='Epoch-level sequence-length curriculum, '
                             '"epoch:observations" pairs, e.g. '
                             '"1:4,6:5,10:6,14:7" = 4 observations from epoch 1, '
                             '5 from epoch 6, 6 from epoch 10, 7 from epoch 14. '
                             'Steps at epoch boundaries and holds; overrides '
                             '--num-frames. Empty = fixed --num-frames.')
    parser.add_argument('--mem-warmup-epoch', type=int, default=0,
                        help='Enable the multi-depth trajectory memory only '
                             'from this epoch onward (1-indexed); the track '
                             'path first learns single-step propagation. '
                             'Gates the seed (d=0) and per-depth (d=1,2,3) '
                             'memory attention together. 0 = no warm-up.')
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    """Seed every RNG the training path draws from."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own deterministic stream.

    torch seeds workers from the loader's generator, so deriving the python
    and numpy seeds from it keeps the augmentations reproducible without
    making every worker draw the same sequence.
    """
    s = torch.initial_seed() % 2 ** 32
    random.seed(s)
    np.random.seed(s)


def main():
    args = parse_args()

    _seed_everything(args.seed)
    log.info(f"Seed: {args.seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # -----------------------------------------------------------------------
    # 1. Build model
    # -----------------------------------------------------------------------
    if args.trajectory_state_prop:
        if MODEL_CFG.get('trajectory_memory_cfg') is None:
            MODEL_CFG['trajectory_memory_cfg'] = dict()
        tmcfg = MODEL_CFG['trajectory_memory_cfg']
        tmcfg.update(dict(
            enable_memory_attention=True,
            memory_attention_num_heads=8,
            memory_attention_dropout=args.memory_attn_dropout,
            memory_attention_residual_scale=args.memory_attn_residual_scale,
            memory_aggregation=getattr(args, 'mem_aggregation', 'attention'),
            interleave_depths=int(getattr(args, 'memory_depths', 3)),
            ema_alpha=float(getattr(args, 'mem_ema_alpha', 0.9)),
        ))
        log.info("[trajectory-state propagation] enabled")
        log.info(
            f"  trajectory state: memory attention "
            f"(residual_scale={args.memory_attn_residual_scale}, "
            f"dropout={args.memory_attn_dropout})")

    if args.loss_cls_det_weight > 0.0:
        MODEL_CFG['track_loss_cfg']['loss_cls_det_weight'] = (
            args.loss_cls_det_weight)
        log.info(f"  loss_cls_det_weight = {args.loss_cls_det_weight}")
    MODEL_CFG['track_loss_cfg']['det_pool_mode'] = args.det_pool_mode
    log.info(f"  det_pool_mode = {args.det_pool_mode}")
    MODEL_CFG['track_loss_cfg']['dedup_bg_weight'] = args.dedup_bg_weight
    if args.det_pool_mode == 'dedup':
        log.info(f"  dedup_bg_weight = {args.dedup_bg_weight} "
                 f"(normalized by num_pos_det)")

    # End-to-end association wiring. The DN-anchor hyperparameters below are
    # part of this block, so they are configured exactly when --e2e-assoc is on
    # and nothing else gates them.
    if bool(getattr(args, 'e2e_assoc', False)):
        MODEL_CFG['e2e_assoc'] = True
        log.info("  E2E-ASSOC ON: the propagated track query is shaped "
                 "in-graph by the box/cls losses; association is read off "
                 "slot continuity and no matcher is run.")
        MODEL_CFG['dn_anchor_denoise_p'] = float(
            getattr(args, 'dn_anchor_denoise_p', 0.0) or 0.0)
        MODEL_CFG['dn_anchor_cxy'] = float(getattr(args, 'dn_anchor_cxy', 0.12))
        MODEL_CFG['dn_anchor_wh'] = float(getattr(args, 'dn_anchor_wh', 0.2))
        MODEL_CFG['dn_anchor_object_scaled'] = bool(
            getattr(args, 'dn_anchor_object_scaled', False))
        if MODEL_CFG['dn_anchor_denoise_p'] > 0:
            log.info(f"  DN anchor denoise p={MODEL_CFG['dn_anchor_denoise_p']} "
                     f"(cxy±{MODEL_CFG['dn_anchor_cxy']}, wh×1±{MODEL_CFG['dn_anchor_wh']}, "
                     f"object_scaled={MODEL_CFG['dn_anchor_object_scaled']})")
            # What DN denoising actually needs is un-detached track
            # reference_points, which either flag provides:
            # --train-track-ref-refine (and additionally trains
            # reg_branches) or --track-ref-grad (gradient only).
            if not (bool(getattr(args, 'train_track_ref_refine', False))
                    or bool(getattr(args, 'track_ref_grad', False))):
                log.warning("  DN anchor denoise needs --track-ref-grad "
                            "(or --train-track-ref-refine) — without one "
                            "the carried ref is detached and this is a "
                            "no-op")

    log.info("Building INGRAINTracker ...")
    model = build_model()

    # -----------------------------------------------------------------------
    # 2. Load GD pretrained weights (base)
    # -----------------------------------------------------------------------
    if not os.path.isfile(args.gd_ckpt):
        raise FileNotFoundError(
            f"GD checkpoint not found: {args.gd_ckpt}\n"
            f"Pass a valid path via --gd-ckpt, or drop the file at the default "
            f"location ({GD_CKPT})."
        )
    load_gd_weights(model, args.gd_ckpt)

    # Load INGRAIN checkpoint if provided — overlays on top of GD weights
    # (weights only, no optimizer restore)
    if args.init_ckpt:
        log.info(f"Loading INGRAIN init weights: {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        ingrain_sd = ckpt.get('model', ckpt)
        _m0, _u0 = model.load_state_dict(ingrain_sd, strict=False)
        log.info("  INGRAIN weights loaded (optimizer starts fresh): "
                 f"missing={len(_m0)} unexpected={len(_u0)}")
        # The track path is built further below, so it does not exist yet and
        # a checkpoint's decoder.track_dec_* tensors have nowhere to land
        # here. --init-weights is applied after the build and does carry them.
        _tdc0 = sum(1 for k in _u0 if k.startswith('decoder.track_dec_'))
        if _tdc0:
            log.warning("  %d decoder.track_dec_* tensors were NOT applied "
                        "(the track path is built later) — use "
                        "--init-weights to warm-start the track path", _tdc0)
    else:
        log.info("  No --init-ckpt passed → PURE GD start "
                 "(INGRAIN-new params: random init)")

    # Override content-isolation layer count if specified via CLI
    num_dec_layers = model.decoder.num_layers
    if args.decoupling_depth is not None:
        if not (0 <= args.decoupling_depth <= num_dec_layers):
            raise ValueError(
                f"--decoupling-depth must be in [0, {num_dec_layers}], "
                f"got {args.decoupling_depth}")
        model.decoder.num_isolation_layers = args.decoupling_depth
        log.info(f"  decoder.num_isolation_layers overridden: "
                 f"{args.decoupling_depth}/{num_dec_layers} layers "
                 f"(det↔track blocked in first {args.decoupling_depth} layers)")
    else:
        log.info(f"  decoder.num_isolation_layers = "
                 f"{model.decoder.num_isolation_layers}/{num_dec_layers} "
                 f"(MODEL_CFG default)")
    if getattr(args, 'iso_directional', False):
        model.decoder.iso_directional = True
        log.info("  DIRECTIONAL isolation: track READS det in all "
                 "layers; det+DN never read track.")
    # Guard: under e2e-assoc + duplicate suppression (det_pool_mode=dedup,
    # dedup_bg_weight>0) the LAST layers MUST allow det↔track self-attention so
    # det queries can SEE existing tracks and learn to predict background on
    # already-tracked objects (loss_dedup_bg). FULL isolation (==num_layers)
    # makes dedup_bg an incoherent objective (identical det queries asked to both
    # fire and say-bg with no distinguishing input). So require isolation <
    # num_layers. Det-slice calibration is protected by loss_cls_det (active
    # supervision), NOT by full isolation.
    if bool(getattr(args, 'e2e_assoc', False)):
        _iso = int(model.decoder.num_isolation_layers)
        _dedup = (str(getattr(args, 'det_pool_mode', '')) == 'dedup'
                 and float(getattr(args, 'dedup_bg_weight', 0.0)) > 0)
        if _dedup and _iso >= num_dec_layers:
            raise SystemExit(
                f"--e2e-assoc with duplicate suppression needs isolation < {num_dec_layers} "
                f"(fusion-stage layers open for det↔track) but got {_iso}. Full "
                f"isolation breaks loss_dedup_bg. Use --decoupling-depth 4.")
        log.info(f"  e2e-assoc: {num_dec_layers - _iso} fusion-stage layer(s) open for "
                 f"det↔track duplicate suppression (isolation={_iso}); det cls protected by "
                 f"loss_cls_det. cls_track dropped: "
                 f"{bool(getattr(args, 'e2e_drop_cls_track', False))}.")

    # Log training config
    log.info("  === INGRAIN Training ===")

    # Log effective negative sampling
    log.info(f"  prompt negative-class sampling: {NUM_NEGATIVES} "
             f"classes/batch (cap {PROMPT_MAX_TOTAL} names)")
    log.info(f"  grad clip_max_norm: {args.clip_max_norm}")

    # -----------------------------------------------------------------------
    # 3. Keep the GD pipeline and the seed memory fixed; --unfreeze-parts and
    # --freeze-all-but-track-path re-open the track decoding path below.
    # -----------------------------------------------------------------------
    frozen = freeze_gd_params(model, patterns=FREEZE_PATTERNS)
    log.info(f"Parameter selection applied to {frozen} tensors")

    # Re-enable grad on the LAST N decoder layers AFTER the selection pass.
    # They match GD_NATIVE_PATTERNS → routed to the GD-native optimizer group
    # at lr_gd (3e-5 vs 1e-4 for the newly introduced modules, in the released
    # recipe). bbox_head stays fixed, so GD's cls calibration is preserved at
    # the head; only the decoder hidden adapts.
    # default empty so the --freeze-all-but-track-path block below can
    # safely reference them ("x".startswith(()) is False) to AVOID re-disabling
    # the explicitly-requested decoder layers (the det↔track fusion stage).
    prefixes = ()
    gd_dec_parts = ()
    n_unfreeze = int(getattr(args, 'unfreeze_decoder_last_n', 0) or 0)
    if n_unfreeze > 0 and hasattr(model, 'decoder') and \
            hasattr(model.decoder, 'layers'):
        n_dec = len(model.decoder.layers)
        unfreeze_ids = list(range(max(0, n_dec - n_unfreeze), n_dec))
        prefixes = tuple(f'decoder.layers.{i}.' for i in unfreeze_ids)
        # Positive-match: re-enable ONLY the GD-native decoder components
        # (self_attn / cross_attn / cross_attn_text / ffn / norms). This
        # excludes the track-query adapter inside the layer, which would
        # otherwise route to INGRAIN-new at full LR instead of the GD-native rate.
        gd_dec_parts = tuple(
            s.strip() for s in
            getattr(args, 'unfreeze_parts',
                    'self_attn,cross_attn,ffn,norms').split(',')
            if s.strip())
        n_un = 0
        for name, p in model.named_parameters():
            # Boundary match '.{part}.' — a bare substring test would let
            # 'cross_attn' swallow 'cross_attn_text' (the query→text reading
            # path = recognition-adjacent, must stay fixed unless explicitly
            # listed as its own part).
            if name.startswith(prefixes) and any(
                    f'.{s}.' in name for s in gd_dec_parts):
                p.requires_grad_(True)
                n_un += 1
        log.info(f"E2E: decoder layers {unfreeze_ids} GD-native "
                 f"parts {gd_dec_parts} ({n_un} param tensors) train at "
                 f"GD-native LR {args.lr_gd}")
        # ref_point_head is NOT a per-layer component (it's
        # decoder.ref_point_head.*) so the per-layer prefix loop above misses it.
        # It recomputes query_pos from the (evolving) reference every layer
        # (see the decoder's forward) = the relocalization
        # knob; leaving it fixed caps the track query's ability to relocalize.
        # Positively re-enable it when 'ref_point_head' is listed in
        # --unfreeze-parts (routes to GD-native LR via GD_NATIVE_PATTERNS).
        if 'ref_point_head' in gd_dec_parts and \
                hasattr(model.decoder, 'ref_point_head'):
            _nrph = 0
            for name, p in model.named_parameters():
                if name.startswith('decoder.ref_point_head.'):
                    p.requires_grad_(True)
                    _nrph += 1
            log.info(f"  decoder.ref_point_head.* "
                     f"({_nrph} tensors) → GD-native LR {args.lr_gd} "
                     f"(relocalization query_pos head)")
    # The final word on parameter selection: it runs after the passes above and
    # overrides them, so log_trainable_summary reports the outcome of every
    # selection pass. (The track path is built later and logs its own
    # trainable count.) Everything not named below stays fixed.
    if getattr(args, 'freeze_all_but_track_path', False):
        n_tim = 0
        n_decunf = 0
        for name, p in model.named_parameters():
            # INGRAIN_ENCROI_MEM_FUSE=1: re-enable the seed memory-attention so
            # it can learn (FREEZE_PATTERNS otherwise locks it at its
            # zero-init no-op). Paired with the memory fusion in detector.py.
            # This switch alone: the seed refines whatever content
            # _propagate_tracks propagates — the enc-ROI observation under
            # INGRAIN_TRACK_CONTENT_ENCROI=1, the decoded track state under =0 —
            # so it is called, and does receive gradient, either way.
            if (os.environ.get('INGRAIN_ENCROI_MEM_FUSE') == '1'
                    and name.startswith('trajectory_memory.seed_mem_attn.')):
                p.requires_grad_(True)
                n_tim += 1
            # INGRAIN_MEMFUSE_INTERLEAVE=1: re-enable the 3 per-depth
            # interleaved memory-attention modules (applied after track-path
            # layers 0/1/2). zero-init out_proj => safe start. Independent of
            # memory_attention above (both flags together = 4-point injection).
            elif (os.environ.get('INGRAIN_MEMFUSE_INTERLEAVE') == '1'
                  and name.startswith('trajectory_memory.depth_mem_attn.')):
                p.requires_grad_(True)
                n_tim += 1
            # Track-only decoder adapters (zero-init) — new
            elif name.startswith(prefixes) and any(
                    f'.{s}.' in name for s in gd_dec_parts):
                # preserve the explicit --unfreeze-decoder-last-n
                # selection (e.g. last-N self_attn = the det↔track fusion
                # stage). Without this the blanket selection below would clobber
                # it → the fusion stage would have ZERO trainable params (no-op).
                p.requires_grad_(True)
                n_decunf += 1
            elif (name.startswith('decoder.ref_point_head.')
                  and 'ref_point_head' in gd_dec_parts):
                # Preserve the explicit --unfreeze-parts ref_point_head
                # selection (the relocalization query_pos head; NOT a per-layer
                # module, so the prefixes test above misses it). Without this
                # the blanket selection below re-locks it.
                p.requires_grad_(True)
                n_decunf += 1
            else:
                p.requires_grad_(False)
        log.info(f"  TRACK-PATH-ONLY: trainable = "
                 f"decoder_last_n ({n_decunf}) + traj_mem_attn ({n_tim})")

    # Track-ref refinement. Set AFTER all parameter-selection
    # logic so it survives --freeze-all-but-track-path. Re-enable ONLY the last-N
    # decoder reg_branches (NOT cls_branches → open-vocab + score calibration
    # preserved) + flip the decoder flag so the track-slice inter-layer ref keeps
    # gradient in the fusion stage. reg_branches route to lr_gd via 'bbox_head.*'.
    # --track-ref-grad: the decoder half of --train-track-ref-refine WITHOUT
    # unfreezing any detection parameter. A track's box is predicted as an
    # offset from its carried positional reference, so gradient has to flow
    # through the track slice's reference_points in the fusion stage; the
    # regression branches do not have to be trainable for that. Use this when
    # the detector is meant to stay fixed.
    if (getattr(args, 'track_ref_grad', False)
            and not getattr(args, 'train_track_ref_refine', False)):
        if hasattr(model, 'decoder'):
            model.decoder.train_track_ref_refine = True
            log.info("  track-ref-grad ON — track-slice reference_points carry "
                     "gradient through the fusion stage (layer-wise box "
                     "refinement); the detection parameters are unaffected.")
    if getattr(args, 'train_track_ref_refine', False):
        n_dec = len(model.decoder.layers) if hasattr(model, 'decoder') else 0
        n_unf = int(getattr(args, 'unfreeze_decoder_last_n', 0) or 0) or 2
        reg_ids = set(range(max(0, n_dec - n_unf), n_dec))
        n_reg = 0
        for name, p in model.named_parameters():
            if name.startswith('bbox_head.reg_branches.'):
                try:
                    lid = int(name.split('bbox_head.reg_branches.')[1].split('.')[0])
                except (ValueError, IndexError):
                    continue
                if lid in reg_ids:
                    p.requires_grad_(True)
                    n_reg += 1
        if hasattr(model, 'decoder'):
            model.decoder.train_track_ref_refine = True
        log.info(f"  train-track-ref-refine ON — reg_branches "
                 f"layers {sorted(reg_ids)} ({n_reg} params → lr_gd) + track-slice "
                 "ref gradient in the fusion stage; cls_branches are left as "
                 "pretrained (open-vocab preserved).")
    # Must run after every selection pass above, which is what makes it an override.
    if getattr(args, 'unfreeze_cls_branches', False):
        _n_cls = 0
        for name, p in model.named_parameters():
            if name.startswith('bbox_head.cls_branches.'):
                p.requires_grad_(True)
                _n_cls += 1
        log.info(f"  ABLATION: released the region-text contrastive head "
                 f"({_n_cls} params in bbox_head.cls_branches → lr_gd). The "
                 f"default configuration leaves them as pretrained.")
    log_trainable_summary(model)

    model = model.to(device)
    model.train()

    # Per-module diagnostics: attach a DiagAccumulator so the per-frame
    # _collect_diag + per-opt-step grad-norms aggregate into one log line.
    if getattr(args, 'diag_log', False):
        from ingrain.models.diag import DiagAccumulator
        model.set_diag(DiagAccumulator())
        log.info("  diag-log ON (track-query box IoU, recover-missed, "
                 "duplicate-fire, track counts, matched-det score, "
                 "grad-norms).")

    # 4. Build dataset + dataloader
    # -----------------------------------------------------------------------
    log.info("Building LVISObservationSeqDataset ...")
    dataset = LVISObservationSeqDataset(
        ann_file=ANN_FILE,
        h5_file=H5_FILE,
        classes_file=CLASSES_FILE,
        num_frames=args.num_frames,
        motion_aug_strength=getattr(args, 'motion_aug_strength', 1.0),
        motion_aug_level=getattr(args, 'motion_aug_level', 'basic'),
        rot_amp=getattr(args, 'rot_amp', 0.0),
        scale_amp=getattr(args, 'scale_amp', 0.0),
        occlude_ratio_early=getattr(args, 'occlude_ratio', 0.4),
        seed=args.seed,
    )
    _loader_gen = torch.Generator()
    _loader_gen.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        generator=_loader_gen, worker_init_fn=_seed_worker,
        num_workers=args.num_workers,
        collate_fn=mot_collate_fn,
        # RAM is usually the bottleneck here and pinning's payoff is marginal,
        # so INGRAIN_NO_PIN_MEMORY=1 turns it off (scripts/eval.sh sets it).
        pin_memory=(args.batch_size <= 8
                    and os.environ.get('INGRAIN_NO_PIN_MEMORY') != '1'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    log.info(f"Dataset: {len(dataset)} samples, "
             f"{len(dataloader)} batches/epoch")

    # -----------------------------------------------------------------------
    # 5. Load class names (for clip-level prompt building)
    # -----------------------------------------------------------------------
    class_names = load_class_names(CLASSES_FILE)
    log.info(f"Loaded {len(class_names)} LVIS class names")
    # -----------------------------------------------------------------------
    # 5a-bis. BASE-ONLY vocabulary (open-vocabulary protocol, always on).
    #
    # The open-vocabulary protocol trains on base categories only. Restricting
    # the *annotations* is not sufficient: with the full vocabulary in the
    # prompt, every rare (novel) class name is still sampled as a NEGATIVE, so
    # the model is explicitly trained to push novel-category scores down — the
    # opposite of what the novel split is supposed to measure. This restricts
    # the negative-sampling pool to base classes, reading the frequency field
    # of the training annotation file itself.
    #
    # Positives are never dropped by build_clip_text_prompt, so a stray rare
    # GT instance (there should be none) still trains correctly rather than
    # corrupting the prompt.
    # -----------------------------------------------------------------------
    import json as _json
    with open(ANN_FILE) as _f:
        _bo_json = _json.load(_f)
    _bo_cats = _bo_json.get('categories', [])
    _bo_anns = _bo_json.get('annotations', [])
    _freqs = {c.get('frequency') for c in _bo_cats}
    if not _freqs & {'f', 'c', 'r'}:
        raise RuntimeError(
            f'{ANN_FILE} has no LVIS `frequency` field on '
            f'its categories, so base and novel cannot be separated.')
    _base_pool = sorted(c['id'] - 1 for c in _bo_cats
                        if c.get('frequency') in ('f', 'c')
                        and 1 <= c['id'] <= len(class_names))
    _n_rare = sum(1 for c in _bo_cats if c.get('frequency') == 'r')
    # The negative pool is only half the protocol. Positives come from the
    # annotation file unfiltered, so an annotation file that still contains
    # rare-category boxes trains on the novel split while this block reports
    # that novel names are excluded. Refuse the file instead.
    _rare_ids = {c['id'] for c in _bo_cats if c.get('frequency') == 'r'}
    _n_rare_ann = sum(1 for a in _bo_anns if a.get('category_id') in _rare_ids)
    if _n_rare_ann:
        raise RuntimeError(
            f'{ANN_FILE} carries {_n_rare_ann} '
            f'annotations of rare (novel) categories. Training on them '
            f'would put the novel split in the training set and invalidate '
            f'the novel-category evaluation. Use an annotation file with '
            f'base (f/c) instances only.')
    args._neg_pool = _base_pool
    log.info(f"BASE-ONLY VOCAB: negatives sampled from "
             f"{len(_base_pool)} base (f+c) class names; {_n_rare} rare "
             f"(novel) names in {os.path.basename(ANN_FILE)} are EXCLUDED "
             f"from every training prompt, and the file carries no "
             f"rare-category instances.")

    # -----------------------------------------------------------------------
    # 6. Optimiser (only trainable params)
    # -----------------------------------------------------------------------
    # Split trainable params into INGRAIN-new (high LR) vs GD-native (low LR).
    #
    # Build the decoupling stage's SEPARATE trainable track path AFTER all
    # parameter selection above, so configure_track_decoder_copy's requires_grad
    # is final (the blanket selection never sees these params). Warm-starts from
    # the loaded decoder weights. Its params (decoder.track_dec_*) route to
    # their own LR group below; the detection path stays on decoder.layers.
    if getattr(args, 'track_path', False):
        _rb = getattr(getattr(model, 'bbox_head', None), 'reg_branches', None)
        if _rb is None:
            raise RuntimeError('--track-path: model.bbox_head.reg_branches '
                               'not found (cannot build the track path)')
        # The track path spans the first n=num_isolation_layers layers (the
        # decoupling stage); the remaining main layers (which MUST be trainable
        # via --unfreeze-decoder-last-n) form the det↔track fusion stage.
        _nsplit = None
        if getattr(args, 'staged_decoder', False):
            _ndl = len(model.decoder.layers)
            _nsplit = int(model.decoder.num_isolation_layers)
            if not (1 <= _nsplit < _ndl):
                raise RuntimeError(
                    f'--staged-decoder needs --decoupling-depth in '
                    f'[1, {_ndl-1}] (the split count); got {_nsplit}.')
            _nmerge = _ndl - _nsplit
            _ufn = int(getattr(args, 'unfreeze_decoder_last_n', 0) or 0)
            if _ufn != _nmerge:
                log.warning(
                    '  --unfreeze-decoder-last-n=%d != fusion-stage depth '
                    '%d (num_layers %d - split depth %d). The fusion-stage '
                    'det↔track layers will NOT all be trainable unless these '
                    'match.', _ufn, _nmerge, _ndl, _nsplit)
            log.info('  [stages] decoupling=%d | fusion=%d '
                     '(det↔track, trained)', _nsplit, _nmerge)
        _cparts = [s.strip() for s in
                   str(getattr(args, 'track_path_parts', '') or '').split(',')
                   if s.strip()] or None
        _ntdc = model.decoder.configure_track_decoder_copy(
            _rb, n_split_layers=_nsplit, copy_parts=_cparts,
            release_text_ca=bool(getattr(args, 'unfreeze_track_text_ca', False)))
        log.info("  decoupling-stage track path: %.2fM trainable params added "
                 "(after parameter selection; lr=%g)",
                 _ntdc / 1e6, float(getattr(args, 'track_path_lr', 2e-5)))
        # RESTORE the detection path to PURE GD — det must be the untouched
        # detector, not a drifted one. load_gd_weights writes every GD-native
        # key (decoder.layers/ref_point_head/cls/reg/backbone/encoder) back to
        # the pure-GD base ckpt; the track path (decoder.track_dec_*) and the
        # INGRAIN modules are NOT in the GD ckpt so they are left alone.
        # requires_grad untouched (the detection path stays fixed).
        # NOTE ON ORDER: --init-weights and --resume are both applied later
        # (step 7), so a warm-started run reaches the track path and the
        # trainable fusion-stage layers through that checkpoint's own
        # decoder.track_dec_* / decoder.layers.{n..} keys, which supersede
        # this restore — that is the intended warm-start semantics. The
        # decoupling-stage detection layers are never trained, so they stay
        # pure GD either way. Without a warm start (the released recipe) the
        # restore below is what the run trains from.
        if os.path.isfile(args.gd_ckpt):
            log.info("  det=PURE GD: restoring detection-path GD-native "
                     "weights from %s (track path + INGRAIN modules untouched)",
                     os.path.basename(args.gd_ckpt))
            load_gd_weights(model, args.gd_ckpt)
        else:
            log.warning("  gd_ckpt missing (%s) — the detection path is NOT "
                        "restored to pure GD", args.gd_ckpt)

    # GD-native layers: the full decoder + bbox_head, trained at lr_gd
    # (slower than INGRAIN-new) so GD's pretrained detection quality is
    # preserved while the model gradually adapts for tracking.
    # INGRAIN-new (lr) catches only the newly introduced modules.
    GD_NATIVE_PATTERNS = [
        'decoder.layers.*.self_attn.*',
        'decoder.layers.*.norms.*',
        'decoder.layers.*.cross_attn_text.*',
        'decoder.layers.*.cross_attn.*',
        'decoder.layers.*.ffn.*',
        'decoder.ref_point_head.*',
        'decoder.norm.*',
        'bbox_head.*',
        # The track-side adapters spliced into the fusion-stage layers bill
        # at --lr-gd; their copies inside decoder.track_dec_layers are
        # diverted above and bill at --track-path-lr.
        'decoder.layers.*.track_sampling_adapter.*',
    ]
    # enc_roi_proj is a NEW module → --freeze-all-but-track-path would have
    # excluded it. Re-enable so it trains (lands in ingrain_new param group @ --lr).
    if (os.environ.get('INGRAIN_TRACK_CONTENT_ENCROI') == '1'
            and getattr(model, 'enc_roi_proj', None) is not None):
        for _p in model.enc_roi_proj.parameters():
            _p.requires_grad_(True)
        log.info("enc_roi_proj TRAINABLE (encoder-ROI content path)")
    # hidden_fuse is a NEW module → same re-enable so it trains.
    if getattr(model, 'hidden_fuse', None) is not None:
        for _p in model.hidden_fuse.parameters():
            _p.requires_grad_(True)
        log.info("hidden_fuse TRAINABLE (concat-MLP: enc-ROI + decoder hidden, "
                 "zero-init = enc-ROI)")
    # track_sampling_adapter: NEW per-GD-decoder-layer track-slice bottleneck
    # (env INGRAIN_TRACK_SAMPLING_ADAPTER=1). Lives on each IngrainDecoderLayer, so
    # walk modules and re-enable grad. zero-init => identity at start.
    _tsa_params = 0
    for _m in model.modules():
        _tsa = getattr(_m, 'track_sampling_adapter', None)
        if _tsa is not None:
            for _p in _tsa.parameters():
                _p.requires_grad_(True)
                _tsa_params += _p.numel()
    if _tsa_params > 0:
        log.info(f"track_sampling_adapter TRAINABLE ({_tsa_params/1e6:.3f}M: "
                 "per-layer track-slice adapter before deformable sampling, "
                 "zero-init)")
    ingrain_new_params = []
    gd_native_params = []
    ingrain_new_names = []
    gd_native_names = []
    # The decoupling stage's separate trainable track-path params get their OWN
    # LR group (NOT the INGRAIN-new lr, NOT lr_gd which is for the pretrained GD layers).
    tdc_params, tdc_names = [], []
    _tdc_lr = float(getattr(args, 'track_path_lr', 2e-5) or 2e-5)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Route the SEPARATE track path (decoder.track_dec_*) to
        # its own gentle LR group BEFORE any other classification (its layers
        # contain cross_attn.sampling_offsets / reg / etc. that would otherwise be
        # caught by the gd-native branch below).
        if name.startswith('decoder.track_dec_'):
            tdc_params.append(p)
            tdc_names.append(name)
            continue
        is_gd = any(
            fnmatch.fnmatch(name, pat) for pat in GD_NATIVE_PATTERNS)
        if is_gd:
            gd_native_params.append(p)
            gd_native_names.append(name)
        else:
            ingrain_new_params.append(p)
            ingrain_new_names.append(name)
    gd_native_lr = args.lr_gd
    log.info(f"Optimizer param groups: INGRAIN-new {len(ingrain_new_params)} params (lr={args.lr}), "
             f"GD-native {len(gd_native_params)} params (lr={gd_native_lr})")
    # Print a compact sanity breakdown so LR regressions are easy to spot
    def _summarize(names):
        buckets = {}
        for n in names:
            key = '.'.join(n.split('.')[:2])
            buckets[key] = buckets.get(key, 0) + 1
        return ', '.join(f"{k}:{v}" for k, v in sorted(buckets.items()))
    log.info(f"  INGRAIN-new groups: {_summarize(ingrain_new_names)}")
    log.info(f"  GD-native groups: {_summarize(gd_native_names)}")
    _opt_groups = [
        {'params': ingrain_new_params, 'lr': args.lr},
        {'params': gd_native_params, 'lr': gd_native_lr},
    ]
    if tdc_params:  # separate trainable track path, dedicated LR
        _opt_groups.append({'params': tdc_params, 'lr': _tdc_lr})
        log.info(f"  TRACK-PATH group: {len(tdc_params)} params "
                 f"(lr={_tdc_lr:g}): {_summarize(tdc_names)}")
    optimizer = torch.optim.AdamW(_opt_groups, weight_decay=WEIGHT_DECAY)
    _nominal_lrs = [pg['lr'] for pg in optimizer.param_groups]
    scaler = GradScaler(enabled=AMP)

    # -----------------------------------------------------------------------
    # 7. Optionally resume
    # -----------------------------------------------------------------------
    start_epoch = 1
    global_step = [0]   # list wrapper so train_one_epoch can mutate it
    if getattr(args, 'init_weights', None):
        # Weights-ONLY warm start: fresh optimizer/scaler/schedule/
        # step, new run identity. Unlike --resume, epoch stays 1 so all
        # frac-keyed anneals/curricula re-walk over the new data. Must run
        # and BEFORE training; GD rows overwrite still-pretrained values.
        _ck = torch.load(args.init_weights, map_location='cpu',
                         weights_only=False)
        _missing, _unexpected = model.load_state_dict(
            _ck.get('model', _ck), strict=False)
        log.info(f"INIT-WEIGHTS from {args.init_weights} "
                 f"(step {_ck.get('step', '?')}): "
                 f"missing={len(_missing)} unexpected={len(_unexpected)}")
        for k in list(_missing)[:5]:
            log.info(f"    + missing (fresh init): {k}")
        del _ck
    if args.resume:
        start_epoch, resumed_step = load_checkpoint_resume(
            model, optimizer, scaler, args.resume)
        # Restore the global step counter so LR scheduler / max-opt-steps
        # continue from the resumed position. Without this, warmup
        # restarts from 0 and --max-opt-steps becomes relative-not-absolute.
        global_step[0] = resumed_step
        log.info(f"  Restored global_step = {resumed_step}")

    # -----------------------------------------------------------------------
    # 8. Training loop
    # -----------------------------------------------------------------------
    accumulate_steps = (args.accumulate_steps
                        if args.accumulate_steps is not None
                        else ACCUMULATE_STEPS)
    # total_opt_steps is the denominator for cosine decay. When --max-opt-steps
    # is given, it is the **absolute** total opt step target the user wants to
    # reach; the cosine schedule should span from 0 → that target. When not
    # given, use the FULL-RUN epoch budget regardless of start_epoch, so a
    # resumed run keeps the same cosine span as an uninterrupted one (the
    # restored absolute global_step then lands at the right point on it).
    if args.max_opt_steps and args.max_opt_steps > 0:
        total_opt_steps = int(args.max_opt_steps)
    else:
        total_opt_steps = (
            len(dataloader) * args.epochs
        ) // max(accumulate_steps, 1)
    # Nominal per-group LRs captured at optimizer construction — NOT read back
    # from the param groups here, because a resumed optimizer state restores
    # the cosine-decayed lr of the save point and would double-decay.
    base_lrs = _nominal_lrs
    log.info(f"Starting: epochs {start_epoch}–{args.epochs}, "
             f"lr={args.lr}, accumulate={accumulate_steps} "
             f"(eff_batch={args.batch_size * accumulate_steps}), "
             f"warmup={args.warmup_steps}, decay={args.lr_decay}, "
             f"total_opt_steps={total_opt_steps}, amp={AMP}")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Install SIGTERM handler so `kill -TERM <pid>` stops cleanly (no D-state
    # worker orphans, no dead GPU memory). NOTE: `kill -9` skips this handler
    # and will leave orphans — always TERM first, KILL only as last resort.
    _signal.signal(_signal.SIGTERM, _on_sigterm)
    log.info(f"SIGTERM handler installed: `kill -TERM {os.getpid()}` "
             f"stops gracefully (clean workers + CUDA release).")

    # Parse the epoch-level schedules once, before the loop.
    _seq_schedule = []
    _seq_raw = str(getattr(args, 'seq_len_schedule', '') or '')
    if _seq_raw.strip():
        _seq_schedule = sorted(
            (int(_p.split(':')[0]), int(_p.split(':')[1]))
            for _p in _seq_raw.split(',') if _p.strip())
        log.info(f"[seq-len-schedule] {_seq_schedule} "
                 f"(epoch -> observations per clip; overrides --num-frames)")
    _mem_warmup = int(getattr(args, 'mem_warmup_epoch', 0) or 0)
    if _mem_warmup > 0:
        log.info(f"[mem-warmup] trajectory memory disabled until epoch "
                 f"{_mem_warmup}")

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        if _TERM_REQUESTED:
            log.info("[SIGTERM] stopping training between epochs.")
            break
        epoch_t0 = time.time()
        log.info(f"=== Epoch {epoch}/{args.epochs} ===")

        # ── epoch-level sequence-length curriculum (--seq-len-schedule) ──
        # "1:4,6:5,10:6,14:7" = 4 observations from epoch 1, 5 from epoch 6, ...
        # Steps at epoch boundaries and holds.
        if _seq_schedule:
            _tgt = _seq_schedule[0][1]
            for _ep, _nf in _seq_schedule:
                if epoch >= _ep:
                    _tgt = _nf
            _ds_seq = getattr(dataloader, 'dataset', None)
            if (_ds_seq is not None and hasattr(_ds_seq, 'set_num_frames')
                    and _ds_seq.num_frames != _tgt):
                _ds_seq.set_num_frames(_tgt)
                log.info(f"[seq-len-schedule] epoch {epoch} -> "
                         f"{_tgt} observations per clip")

        # ── trajectory-memory warm-up (--mem-warmup-epoch) ──
        # Memory is disabled until the given epoch, so the track path first
        # learns single-step propagation and only then gains history. Gates the
        # seed (d=0) and per-depth (d=1,2,3) memory attention together.
        if _mem_warmup > 0:
            _mem_on = epoch >= _mem_warmup
            _core = model.module if hasattr(model, 'module') else model
            if getattr(_core, 'memory_enabled', True) != _mem_on:
                _core.memory_enabled = _mem_on
                log.info(f"[mem-warmup] epoch {epoch}: trajectory memory "
                         f"{'ENABLED' if _mem_on else 'disabled'} "
                         f"(warm-up until epoch {_mem_warmup})")

        reached_max_opt_steps = train_one_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            class_names=class_names,
            device=device,
            global_step_ref=global_step,
            save_dir=args.save_dir,
            num_negatives=NUM_NEGATIVES,
            clip_max_norm=args.clip_max_norm,
            args=args,
            total_epochs=args.epochs,
            accumulate_steps=accumulate_steps,
            base_lrs=base_lrs,
            total_opt_steps=total_opt_steps,
        )

        elapsed = time.time() - epoch_t0
        last_epoch = epoch
        log.info(f"Epoch {epoch} done in {elapsed/60:.1f} min "
                 f"(global step {global_step[0]})")

        if epoch % SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(model, optimizer, scaler, epoch,
                            global_step[0], args.save_dir,
                    trainable_only=bool(getattr(args, 'save_trainable_only', False)),
                    base_ckpt=_delta_base_ckpt(args))

        if reached_max_opt_steps:
            log.info("Stopping training loop because --max-opt-steps was hit.")
            break

    # Final checkpoint
    final_epoch = max(last_epoch, start_epoch)
    save_checkpoint(model, optimizer, scaler, final_epoch,
                    global_step[0], args.save_dir, tag='_final',
                    trainable_only=bool(getattr(args, 'save_trainable_only', False)),
                    base_ckpt=_delta_base_ckpt(args))
    log.info("Training complete.")


if __name__ == '__main__':
    main()
