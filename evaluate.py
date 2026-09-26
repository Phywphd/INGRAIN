"""INGRAIN inference and TAO subset evaluation.

Pipeline per frame:
  1. The detector produces det and track-query predictions, the track queries
     decoded along the track path and the fusion stage.
  2. The trajectory-state modules update track queries before decoding.
  3. The per-frame runner converts chunked text logits into class scores.
  4. Runtime tracking assigns IDs from score and miss tolerance.
  5. Active tracks are propagated to the next frame.

Usage (TAO evaluation — see scripts/eval.sh for the full invocation):
    python evaluate.py \
        --tao-subset-eval --split val \
        --checkpoint /path/to/ckpt.pth \
        --video-list eval/results/val_FULL988_video_list.txt --max-videos 988 \
        --filter-score-thresh 0.15 \
        --output-dir /path/to/output

Protocol constants (chunk size, thresholds, miss tolerance, NMS) are fixed in
eval/config.py.
"""
import argparse
import copy
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

# Path setup — must happen before any mmdet / ingrain import
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'third_party', 'mmgroundingdino'))
sys.path.insert(0, ROOT)

from mmdet.utils import register_all_modules
register_all_modules(init_default_scope=True)
import ingrain  # noqa: F401 — registers INGRAIN modules with mmdet registry

import torch

from mmengine.config import ConfigDict
from mmdet.registry import MODELS

from eval.ingrain_runner import IngrainRunner
from eval.config import (
    TAO_VAL_ANN, TAO_TEST_ANN, TAO_IMG_ROOT,
    SCORE_THRESH, MISS_TOLERANCE, FILTER_SCORE_THRESH,
    MAX_DETS_PER_FRAME, CHUNKED_SIZE, TOPK_CLS,
    IOU_THRESH, USE_INTER_TRACK_NMS, USE_IOU_GATE, SOURCE_FILTER,
)
from eval.evaluate_tao import run_teta_eval
from eval.result_formatter import ResultFormatter
from eval.tao_dataset import TAODataset

# ============================================================================
# Model / text loading
# ============================================================================

def load_model(ckpt_path: str,
               gd_ckpt_path: str = None,
               trajectory_state_prop: bool = False,
               memory_attn_residual_scale: float = 0.1,
               memory_attn_dropout: float = 0.0,
               mem_aggregation: str = 'attention',
               mem_interleave_depths: int = 3,
               mem_ema_alpha: float = 0.9,
               isolation_layers: int | None = None):
    """Build an INGRAINTracker for TAO/video inference."""
    # Import train.py for MODEL_CFG and default GD_CKPT (single source of truth)
    import train as _train
    gd_ckpt_path = gd_ckpt_path or _train.GD_CKPT

    cfg = copy.deepcopy(_train.MODEL_CFG)
    if cfg.get('trajectory_memory_cfg') is None:
        cfg['trajectory_memory_cfg'] = {}
    if trajectory_state_prop:
        cfg['trajectory_memory_cfg'].update(dict(
            enable_memory_attention=True,
            memory_attention_residual_scale=float(memory_attn_residual_scale),
            memory_attention_dropout=float(memory_attn_dropout),
            # Must match training: the aggregator decides which submodules the
            # memory attention builds (attn.* vs agg_out.*) and the depth count
            # decides how many exist. A mismatch loads a different module set
            # and the checkpoint's memory weights are dropped.
            memory_aggregation=str(mem_aggregation),
            interleave_depths=int(mem_interleave_depths),
            ema_alpha=float(mem_ema_alpha),
        ))
        print("  trajectory-state propagation: enabled "
              f"(memory_attn_scale={memory_attn_residual_scale})")
    if isolation_layers is not None:
        cfg['decoder']['num_isolation_layers'] = int(isolation_layers)
        print(f"  decoder.num_isolation_layers = {isolation_layers}")
    model = MODELS.build(ConfigDict(cfg))

    # Load GD base weights
    gd = torch.load(gd_ckpt_path, map_location='cpu', weights_only=False)
    sd = gd.get('state_dict', gd.get('model', gd))
    ms = model.state_dict()
    loaded = 0
    for k in ms:
        if k in sd and ms[k].shape == sd[k].shape:
            ms[k] = sd[k]
            loaded += 1
    model.load_state_dict(ms)
    print(f"  GD base: {loaded} tensors loaded from the base checkpoint; "
          f"the remaining {len(ms) - loaded} are INGRAIN modules")

    import builtins as _b
    # The decoupling stage's separate track path must exist BEFORE the overlay so its
    # trained decoder.track_dec_* weights load (configured-after = a fresh copy of
    # the GD base = the track path misses its trained weights = train/eval skew). Built
    # from the GD-base decoder here; the overlay below replaces track_dec_* with the
    # trained ckpt weights.
    if bool(getattr(_b, '_INGRAIN_TRACK_PATH', False)):
        _rb_e = getattr(getattr(model, 'bbox_head', None), 'reg_branches', None)
        if _rb_e is not None:
            # Build the track path with the SAME split depth as training
            # (num_isolation_layers) so track_dec_* key count + forward path match.
            _nsplit_e = None
            if bool(getattr(_b, '_INGRAIN_STAGED_DECODER', False)):
                _nsplit_e = int(model.decoder.num_isolation_layers)
            _ntdc_e = model.decoder.configure_track_decoder_copy(
                _rb_e, n_split_layers=_nsplit_e)
            _smtag = (f'split depth n={_nsplit_e}' if _nsplit_e is not None
                      else 'no fusion stage')
            print(f'  TRACK PATH configured pre-overlay ({_smtag}): '
                  f'{_ntdc_e/1e6:.2f}M params (det via the detection path, '
                  f'track via the track path)')

    # Overlay INGRAIN checkpoint (strict=False so new params are kept)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ingrain_sd = ckpt.get('model', ckpt)
    # Delta checkpoint (saves ONLY trainable params + a pointer to the full
    # base ckpt): merge base first, then overlay delta.
    if isinstance(ckpt, dict) and ckpt.get('base_ckpt'):
        _base = torch.load(ckpt['base_ckpt'], map_location='cpu',
                           weights_only=False)
        _base_sd = dict(_base.get('model', _base))
        _base_sd.update(ingrain_sd)
        print(f"  Delta ckpt: {len(ingrain_sd)} trainable tensors overlaid on "
              f"base {ckpt['base_ckpt']}")
        ingrain_sd = _base_sd
    missing, unexpected = model.load_state_dict(ingrain_sd, strict=False)
    if missing or unexpected:
        print(f"  INGRAIN overlay: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys")
        if unexpected:
            print(f"  Unexpected sample: {unexpected[:8]}")
        if missing:
            print(f"  Missing sample: {missing[:8]}")
    # Zero-init modules built from env (OGP fusion gate, track-query adapter)
    # degrade to exact no-ops when the ckpt has no trained weights for them.
    # The run still produces numbers, so the mismatch is otherwise silent:
    # "component enabled" and "component trained" are not the same claim, and
    # only the latter supports a reported number. Built-without-ckpt-weights ABORTS
    # (below); the opposite direction only warns, because turning a component
    # off is how an ablation row is produced and its weights are then meant to
    # be discarded.
    # (env vars actually consulted, human label for the message, description)
    _zero_init_env = {
        'hidden_fuse.': (('INGRAIN_HIDDEN_FUSE',), 'INGRAIN_HIDDEN_FUSE',
                         'observation-guided propagation fusion gate'),
        'track_sampling_adapter.': (('INGRAIN_TRACK_SAMPLING_ADAPTER',),
                                    'INGRAIN_TRACK_SAMPLING_ADAPTER',
                                    'track-query sampling adapter'),
        # An ablation checkpoint evaluated under the full architecture (or
        # the reverse) differs in exactly these modules, and that difference IS
        # the ablated variable — the one thing that must never be silent.
        'enc_roi_proj.': (('INGRAIN_TRACK_CONTENT_ENCROI',),
                          'INGRAIN_TRACK_CONTENT_ENCROI',
                          'enc-ROI observation projection'),
        'seed_mem_attn.': (('INGRAIN_ENCROI_MEM_FUSE',),
                           'INGRAIN_ENCROI_MEM_FUSE',
                           'trajectory memory at depth d=0 (seed)'),
        'depth_mem_attn.': (('INGRAIN_MEMFUSE_INTERLEAVE',),
                            'INGRAIN_MEMFUSE_INTERLEAVE / '
                            '--memory-depths / --mem-aggregation',
                            'per-depth trajectory memory'),
        # The decoupling-stage track path is the single largest trainable
        # block; evaluating without it silently runs track queries through the
        # detection path instead.
        'decoder.track_dec_': ((),
                               '--track-path / '
                               '--staged-decoder',
                               'decoupling-stage track path'),
    }
    _absent = []
    for _pat, (_vars, _env, _desc) in _zero_init_env.items():
        _n = sum(1 for k in missing if _pat in k)
        if _n:
            _absent.append(f"{_desc} ({_n} tensors, {_env})")
        # The opposite direction: the checkpoint carries weights the model did
        # not build, so they are dropped on the floor. That is what an ablation
        # row looks like, so it warns rather than aborts. Skipped when the same
        # module is ALSO missing: both lists then hold the same tensors under
        # two different key names, the module was in fact built, and naming an
        # env switch as the cause sends the reader after the wrong thing.
        _m = 0 if _n else sum(1 for k in unexpected if _pat in k)
        if _m:
            # Name a variable as the cause only after CHECKING it. Blaming an
            # env var that is in fact set sends the reader chasing a phantom.
            _unset = [v for v in _vars if os.environ.get(v) != '1']
            _why = (f"{'/'.join(_unset)} is not set"
                    if _unset else
                    "the current flags did not build it (the env switches are "
                    "set, so check --memory-depths / the CLI flags)")
            print(f"  !! WARNING: the checkpoint contains trained weights for "
                  f"the {_desc} ({_m} tensors), but {_why}, so the module was "
                  f"NOT built and those weights are DISCARDED. Evaluate under "
                  f"the configuration the checkpoint was trained with ({_env}).")
    if _absent:
        raise SystemExit(
            "eval/ckpt MISMATCH — these modules were built but the checkpoint "
            "has no weights for them, so they are zero-init and act as exact "
            "no-ops:\n"
            + '\n'.join(f"    {_u}" for _u in _absent)
            + "\nThe run would report the numbers of a model WITHOUT them. "
              "Evaluate under the configuration this checkpoint was trained "
              "with, or re-key the checkpoint to match.")

    # Construction-independent check. The two memory modules are NOT built from
    # their env switches — seed_mem_attn comes from --trajectory-state-prop and depth_mem_attn
    # is built unconditionally with count = --memory-depths — so with a
    # switch off the trained weights load cleanly, produce no missing/unexpected
    # key, and are then never called. The scan above cannot see that by
    # construction, so read the environment directly. This is the direction that
    # reports the ablated model's numbers under the full model's name.
    _tmem = getattr(model, 'trajectory_memory', None)
    if _tmem is not None:
        # The seed memory refines whatever content is propagated, so it is
        # called whenever INGRAIN_ENCROI_MEM_FUSE is on — with the enc-ROI
        # observation under INGRAIN_TRACK_CONTENT_ENCROI=1 and with the decoded
        # track state under =0. It sits outside the enc-ROI branch on both
        # paths, so the memory depth set is the same either way. Mirrors the
        # training-side gate in train.py.
        _seed_off = [_v for _v in ('INGRAIN_ENCROI_MEM_FUSE',)
                     if os.environ.get(_v) != '1']
        if getattr(_tmem, 'seed_mem_attn', None) is not None and _seed_off:
            print("  !! WARNING: the trajectory memory at depth d=0 (seed) is "
                  f"BUILT and its weights loaded, but "
                  f"{' and '.join(_seed_off)} "
                  f"{'is' if len(_seed_off) == 1 else 'are'} not set, so it is "
                  "never called. These are the numbers of a model WITHOUT the "
                  "seed memory.")
        _nd = len(getattr(_tmem, 'depth_mem_attn', None) or [])
        if _nd and os.environ.get('INGRAIN_MEMFUSE_INTERLEAVE') != '1':
            print(f"  !! WARNING: the per-depth trajectory memory ({_nd} "
                  f"depths) is BUILT and its weights loaded, but "
                  f"INGRAIN_MEMFUSE_INTERLEAVE is not set, so it is never "
                  f"called. These are the numbers of a model WITHOUT the "
                  f"multi-depth memory.")
    return model




# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='INGRAIN inference and evaluation')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='INGRAIN model checkpoint (.pth)')
    p.add_argument('--checkpoints', type=str, nargs='+', default=None,
                   help='Evaluate multiple checkpoints in TAO subset mode.')
    p.add_argument('--tao-subset-eval', action='store_true',
                   help='Run TETA on the selected TAO videos. Required: this '
                        'is the only mode.')
    p.add_argument('--split', choices=['val', 'test'], default='val',
                   help='TAO split for --tao-subset-eval.')
    p.add_argument('--video-names', type=str, nargs='+', default=None,
                   help='TAO video names, e.g. val/BDD/b2cbf6d8-732b47be.')
    p.add_argument('--video-dirs', type=str, nargs='+', default=None,
                   help='Frame directories; converted to TAO video names.')
    p.add_argument('--video-list', type=str, default=None,
                   help='Text file with one TAO video name or frame dir per line.')
    p.add_argument('--tao-ann', type=str, default=None,
                   help='Override master GT ann json for --tao-subset-eval '
                        '(default: TAO_VAL_ANN/TAO_TEST_ANN from eval.config). '
                        'Use to point at a custom TAO-schema GT (e.g. exported '
                        'LVIS observation sequences).')
    p.add_argument('--tao-img-root', type=str, default=None,
                   help='Override image root prefix for --tao-subset-eval '
                        '(default: TAO_IMG_ROOT). file_name fields in the GT '
                        'are joined onto this; pass an empty string and use '
                        'absolute file_name paths for arbitrary frame dirs.')
    p.add_argument('--no-majority-vote', action='store_true',
                   help='Disable ResultFormatter per-track majority vote.')
    p.add_argument('--snapshot-every', type=int, default=0,
                   help='Write intermediate tao_track snapshots every N videos '
                        'in subset eval. 0 disables snapshots.')
    p.add_argument('--mem-aggregation', type=str, default='attention',
                   choices=['attention', 'mean', 'ema'],
                   help='Memory aggregator; MUST match the checkpoint\'s '
                        'training setting.')
    p.add_argument('--mem-ema-alpha', type=float, default=0.9,
                   help='Decay for --mem-aggregation ema.')
    p.add_argument('--memory-depths', type=int, default=3,
                   help='Track-path memory injection points d = 1..N (the seed '
                        'is d = 0); MUST match training.')
    p.add_argument('--resume-snapshot', type=str, default=None,
                   help='Continue an interrupted subset eval from a '
                        'tao_track_snapshot_*.json written by --snapshot-every. '
                        'Videos already present in the snapshot are skipped and '
                        'their tracks are carried into the final scoring, so '
                        'the result covers the whole list. Default None = '
                        'fresh run.')
    p.add_argument('--live-score-every', type=int, default=0,
                   help='Every N videos, run TETA on the videos done SO FAR '
                        '(GT restricted to them) and append a running score to '
                        '<output_dir>/live_scores.tsv — peek the rough score '
                        'mid-eval. 0 disables. Isolated: a failure never breaks '
                        'the real eval.')
    p.add_argument('--max-videos', type=int, default=None,
                   help='Limit selected TAO videos after filtering.')
    p.add_argument('--gd-ckpt', type=str, default=None,
                   help='Override GD base checkpoint path (default: train.GD_CKPT)')
    p.add_argument('--filter-score-thresh', type=float,
                   default=FILTER_SCORE_THRESH,
                   help='Per-frame threshold; tracks '
                        'below this accumulate disappear_time.')
    p.add_argument('--class-list-file', type=str, default=None,
                   help='Explicit class vocabulary as JSON [[name,id],...].')
    p.add_argument('--trajectory-state-prop', action='store_true',
                   help='Enable trajectory-state propagation: track queries '
                        'are updated by multi-head attention over the '
                        'multi-depth trajectory memory before decoding. Must '
                        'match what the checkpoint was trained with.')
    p.add_argument('--memory-attn-residual-scale', type=float, default=0.1,
                   help='Residual scale for trajectory memory attention.')
    p.add_argument('--memory-attn-dropout', type=float, default=0.0,
                   help='Dropout for trajectory memory attention.')
    p.add_argument('--track-path', action='store_true',
                   help='Build the decoupling stage\'s SEPARATE track path — a '
                        'copy of the detection-path decoder layers (det via the '
                        'detection path, track via the copy). MUST match '
                        'training so the ckpt decoder.track_dec_* '
                        'weights load (else track re-runs the detection path = '
                        'train/eval skew).')
    p.add_argument('--staged-decoder', action='store_true',
                   help='Stage the decoder: the track path spans ONLY the first '
                        'n=--decoupling-depth layers (the decoupling stage); the '
                        'remaining main layers are the shared det↔track fusion '
                        'stage. MUST match training (else the track path has the '
                        'wrong depth and ckpt keys mismatch).')
    p.add_argument('--iso-directional', action='store_true',
                   help='Directional isolation: track READS det in the '
                        'isolated layers, det+DN never read track. MUST '
                        'match training.')
    p.add_argument('--decoupling-depth', type=int, default=None,
                   help='Override decoder.num_isolation_layers (the split '
                        'depth n; MODEL_CFG uses 3). MUST match training.')
    # ── Identity association ─────────────────────────────────────────────
    p.add_argument('--e2e-assoc', action='store_true',
                   help='End-to-end association at eval: the trained propagated '
                        'track-query content carries identity; forces slot-'
                        'continuity (no cosine/Hungarian matcher). Must match '
                        'the training --e2e-assoc flag.')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: eval/results/subset_eval; '
                        'scripts/eval.sh passes eval/results/<tag>).')
    return p.parse_args()




def _checkpoint_tag(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    parent = path.parent.name
    return f'{parent}_{path.stem}' if parent else path.stem


def _selected_checkpoints(args):
    ckpts = []
    if args.checkpoints:
        ckpts.extend(args.checkpoints)
    if args.checkpoint:
        ckpts.append(args.checkpoint)
    # Preserve order while removing duplicates.
    deduped = []
    seen = set()
    for ckpt in ckpts:
        if ckpt not in seen:
            deduped.append(ckpt)
            seen.add(ckpt)
    return deduped


def _tao_video_name_from_input(value: str) -> str:
    """Accept either a TAO video name or a frame directory path."""
    value = value.strip()
    if not value:
        return value
    normalized = value.replace('\\', '/')
    for split in ('val', 'test'):
        marker = f'/{split}/'
        if marker in normalized:
            return f'{split}/{normalized.split(marker, 1)[1]}'
        if normalized.startswith(f'{split}/'):
            return normalized
    parts = Path(value).parts
    for idx, part in enumerate(parts):
        if part in {'val', 'test'} and idx + 2 < len(parts):
            return '/'.join(parts[idx:idx + 3])
    return normalized


def _collect_requested_video_names(args):
    values = []
    if args.video_names:
        values.extend(args.video_names)
    if args.video_dirs:
        values.extend(args.video_dirs)
    if args.video_list:
        with open(args.video_list) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    values.append(line)
    names = [_tao_video_name_from_input(v) for v in values]
    names = [n for n in names if n]
    deduped = []
    seen = set()
    for name in names:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def _match_tao_video_ids(data: dict, requested_names):
    if not requested_names:
        raise ValueError('TAO subset eval requires --video-names, --video-dirs, or --video-list')
    by_name = {v.get('name', ''): v for v in data['videos']}
    selected = []
    for req in requested_names:
        if req in by_name:
            selected.append(by_name[req])
            continue
        suffix_matches = [v for v in data['videos']
                          if v.get('name', '').endswith(req)]
        if len(suffix_matches) == 1:
            selected.append(suffix_matches[0])
        elif len(suffix_matches) > 1:
            raise ValueError(f'Ambiguous TAO video selector {req!r}: '
                             f'{[v.get("name") for v in suffix_matches[:8]]}')
        else:
            raise ValueError(f'TAO video not found: {req}')
    deduped = []
    seen = set()
    for video in selected:
        if video['id'] not in seen:
            deduped.append(video)
            seen.add(video['id'])
    return deduped


def _write_tao_subset_ann(ann_file: str, requested_names, out_path: str,
                          max_videos: int = None):
    with open(ann_file) as f:
        data = json.load(f)
    selected_videos = _match_tao_video_ids(data, requested_names)
    if max_videos is not None:
        selected_videos = selected_videos[:max_videos]
    video_ids = {v['id'] for v in selected_videos}
    images = [img for img in data['images'] if img['video_id'] in video_ids]
    image_ids = {img['id'] for img in images}
    annotations = [ann for ann in data.get('annotations', [])
                   if ann.get('video_id') in video_ids
                   or ann.get('image_id') in image_ids]
    subset = dict(data)
    subset['videos'] = selected_videos
    subset['images'] = images
    subset['annotations'] = annotations
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(subset, f)
    print(f'Wrote TAO subset GT: {out_path}')
    print(f'  videos={len(selected_videos)} frames={len(images)} '
          f'annotations={len(annotations)}')
    for video in selected_videos:
        print(f"  video_id={video['id']} name={video.get('name')}")
    return out_path


def _snapshot_identity(ckpt_path: str, args) -> dict:
    """Identify the run a snapshot came from.

    Path alone is too weak — a retrained checkpoint often keeps its filename —
    so the size and mtime go in too. Hashing the file is not worth it: these
    checkpoints are hundreds of megabytes and this runs once per snapshot.
    """
    try:
        st = os.stat(ckpt_path)
        size, mtime = st.st_size, int(st.st_mtime)
    except OSError:
        size, mtime = None, None
    return {'ckpt': os.path.abspath(ckpt_path), 'ckpt_size': size,
            'ckpt_mtime': mtime, 'split': args.split,
            'video_list': args.video_list}


def _check_snapshot_identity(snap_path: str, ckpt_path: str, args) -> None:
    """Refuse to resume a snapshot that another run produced.

    `eval_test_full.sh` picks the newest snapshot by TAG alone, so re-running a
    tag with a different checkpoint would otherwise splice two models' tracks
    into one score that looks perfectly normal.
    """
    meta_path = snap_path + '.meta.json'
    if not os.path.exists(meta_path):
        print(f'[resume] {os.path.basename(snap_path)} predates identity '
              f'tracking; cannot verify it came from this checkpoint')
        return
    with open(meta_path) as f:
        was = json.load(f)
    now = _snapshot_identity(ckpt_path, args)
    diff = [k for k in ('ckpt', 'ckpt_size', 'ckpt_mtime', 'split')
            if was.get(k) != now.get(k)]
    if diff:
        raise SystemExit(
            f'Refusing to resume: {os.path.basename(snap_path)} was written by '
            f'a different run (differs in {", ".join(diff)}).\n'
            f'  snapshot: {was}\n  this run: {now}\n'
            f'Use a different --output-dir/TAG, or delete the stale snapshots.')


def _extract_teta_metrics(output_dir: str):
    results_path = os.path.join(output_dir, 'INGRAIN', 'teta_summary_results.pth')
    metrics = {}
    if not os.path.exists(results_path):
        return metrics
    with open(results_path, 'rb') as f:
        teta_res = pickle.load(f)
    avg = None
    if 'COMBINED_SEQ' in teta_res:
        avg = teta_res['COMBINED_SEQ'].get('average')
    if avg is None:
        avg = teta_res.get('average')
    if avg and 'TETA' in avg and 50 in avg['TETA']:
        row = avg['TETA'][50]
        if len(row) >= 4:
            metrics = {
                'TETA': float(row[0]),
                'LocA': float(row[1]),
                'AssocA': float(row[2]),
                'ClsA': float(row[3]),
            }
    return metrics


def _write_compare_summary(rows, output_dir: str):
    json_path = os.path.join(output_dir, 'compare_summary.json')
    csv_path = os.path.join(output_dir, 'compare_summary.csv')
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)
    fieldnames = [
        'ckpt', 'tag', 'TETA', 'LocA', 'AssocA', 'ClsA', 'output_dir',
        'score_thresh',
        'filter_score_thresh',
        'miss_tolerance',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f'Wrote comparison summary: {json_path}')
    print(f'Wrote comparison CSV: {csv_path}')
    scored = [r for r in rows if 'TETA' in r]
    if scored:
        best = max(scored, key=lambda r: r['TETA'])
        print(f"Best TETA: {best['TETA']:.3f}  {best['ckpt']}")


def _build_runner_for_dataset(model, dataset, args):
    class_names, cat_ids = dataset.get_class_names_and_cat_ids()
    # Explicit class list as JSON [[name, id], ...]. Default None = the full
    # vocabulary above.
    _clf = getattr(args, 'class_list_file', None)
    if _clf:
        import json as _json
        _pairs = _json.load(open(_clf))
        class_names = [p[0] for p in _pairs]
        cat_ids = [int(p[1]) for p in _pairs]
        print(f'  class-list-file: {len(class_names)} classes from {_clf} '
              f'(GT-free explicit vocabulary)')
    # Isolation eval mirrors (must match training flags)
    if getattr(args, 'iso_directional', False):
        model.decoder.iso_directional = True
        print('  DIRECTIONAL isolation ON (track reads det)')
    # Protocol constants come from eval/config.py — fixed paper settings.
    _runner = IngrainRunner(
        model, class_names, cat_ids,
        chunked_size=CHUNKED_SIZE,
        score_thresh=SCORE_THRESH,
        filter_score_thresh=args.filter_score_thresh,
        miss_tolerance=MISS_TOLERANCE,
        max_dets=MAX_DETS_PER_FRAME,
        topk_cls=TOPK_CLS,
        source_filter=SOURCE_FILTER,
        e2e_assoc=bool(getattr(args, 'e2e_assoc', False)),
        use_iou_gate=USE_IOU_GATE,
        iou_thresh=IOU_THRESH,
        use_inter_track_nms=USE_INTER_TRACK_NMS,
    )
    return _runner


def _run_tao_subset_for_checkpoint(args, ckpt_path: str, subset_ann_file: str,
                                   output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    img_root = (args.tao_img_root if getattr(args, 'tao_img_root', None) is not None
                else TAO_IMG_ROOT)
    dataset = TAODataset(subset_ann_file, img_root)
    class_names, _ = dataset.get_class_names_and_cat_ids()
    print('\n' + '=' * 80)
    print(f'Evaluating checkpoint: {ckpt_path}')
    print(f'Output dir: {output_dir}')
    print(f'TAO subset: videos={dataset.num_videos()} frames={dataset.num_frames()} '
          f'classes={len(class_names)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(
        ckpt_path,
        gd_ckpt_path=args.gd_ckpt,
        trajectory_state_prop=bool(args.trajectory_state_prop),
        memory_attn_residual_scale=float(args.memory_attn_residual_scale),
        mem_aggregation=str(getattr(args, 'mem_aggregation', 'attention')),
        mem_interleave_depths=int(getattr(args, 'memory_depths', 3)),
        mem_ema_alpha=float(getattr(args, 'mem_ema_alpha', 0.9)),
        memory_attn_dropout=float(args.memory_attn_dropout),
        isolation_layers=args.decoupling_depth)
    model = model.to(device).eval()

    runner = _build_runner_for_dataset(model, dataset, args)
    formatter = ResultFormatter()
    # ── resume from a snapshot of an interrupted run (--resume-snapshot) ──
    # Pre-seed the formatter with the tracks already produced, and remember
    # which videos they came from so the loop can skip them. Track ids continue
    # past the highest one seen, so the merged output has no collisions.
    # No flag => empty set => behaviour is identical to a fresh run.
    _resume_done_vids: set = set()
    _resume_path = getattr(args, 'resume_snapshot', None)
    if _resume_path:
        _check_snapshot_identity(_resume_path, ckpt_path, args)
        with open(_resume_path) as _rf:
            _resume_tracks = json.load(_rf)
        _resume_done_vids = {t['video_id'] for t in _resume_tracks}
        formatter.results = list(_resume_tracks)
        formatter.global_offset = (
            max(t['track_id'] for t in _resume_tracks) + 1
            if _resume_tracks else 0)
        print(f'[resume] {len(_resume_tracks)} tracks from {_resume_path}; '
              f'skipping {len(_resume_done_vids)} completed videos, '
              f'track-id offset {formatter.global_offset}')
    print(f'Runner: score={runner.score_thresh} '
          f'filter={runner.tracker.filter_score_thresh} '
          f'miss={runner.tracker.miss_tolerance}')

    track_json = os.path.join(output_dir, 'tao_track.json')
    start = time.time()
    total_dets = 0
    video_count = 0

    def write_snapshot(tag):
        if args.snapshot_every <= 0:
            return
        snap_path = os.path.join(output_dir, f'tao_track_snapshot_{tag}.json')
        formatter.finalize(
            snap_path, use_majority_vote=not args.no_majority_vote)
        # Sidecar identifying the run, so a later --resume-snapshot can refuse
        # to continue a different checkpoint's tracks. Kept beside the snapshot
        # rather than inside it: the snapshot is a bare list of detections and
        # the partial scorer reads it as one.
        with open(snap_path + '.meta.json', 'w') as _mf:
            json.dump(_snapshot_identity(ckpt_path, args), _mf, indent=2)

    _live_path = os.path.join(output_dir, 'live_scores.tsv')
    _done_vids = []

    def write_live_score(n_done):
        """Run TETA on the videos done SO FAR (GT restricted to them) and
        append a running score to live_scores.tsv. Fully isolated — any
        failure is swallowed so it can NEVER break the real eval."""
        if args.live_score_every <= 0:
            return
        try:
            import shutil
            tmp = os.path.join(output_dir, '_live')
            if os.path.isdir(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(tmp, exist_ok=True)
            # partial predictions = everything tracked so far
            ptrack = os.path.join(tmp, 'tao_track.json')
            formatter.finalize(ptrack, use_majority_vote=not args.no_majority_vote)
            # partial GT = subset_gt restricted to done video_ids
            with open(subset_ann_file) as _f:
                _gt = json.load(_f)
            _vset = set(_done_vids)
            _gt2 = dict(_gt)
            _gt2['videos'] = [v for v in _gt['videos'] if v['id'] in _vset]
            _imgs = [im for im in _gt['images'] if im['video_id'] in _vset]
            _gt2['images'] = _imgs
            _iset = {im['id'] for im in _imgs}
            _gt2['annotations'] = [a for a in _gt.get('annotations', [])
                                   if a.get('video_id') in _vset
                                   or a.get('image_id') in _iset]
            pgt = os.path.join(tmp, 'subset_gt.json')
            with open(pgt, 'w') as _f:
                json.dump(_gt2, _f)
            run_teta_eval(pgt, ptrack, tmp, dataset)
            m = _extract_teta_metrics(tmp)
            line = (f"{n_done}\t{m.get('TETA', float('nan')):.3f}\t"
                    f"{m.get('LocA', float('nan')):.3f}\t"
                    f"{m.get('AssocA', float('nan')):.3f}\t"
                    f"{m.get('ClsA', float('nan')):.3f}")
            _hdr = not os.path.exists(_live_path)
            with open(_live_path, 'a') as _f:
                if _hdr:
                    _f.write("n_videos\tTETA\tLocA\tAssocA\tClsA\n")
                _f.write(line + "\n")
            print(f"  [live-score] after {n_done} videos: {line}")
        except Exception as _e:
            print(f"  [live-score] skipped (n={n_done}): {_e}")

    for video_info, frames in dataset.iter_videos():
        if video_info['id'] in _resume_done_vids:
            video_count += 1
            _done_vids.append(video_info['id'])
            print(f"  video {video_count}/{dataset.num_videos()}: "
                  f"{video_info.get('name')} — already in the resume "
                  f"snapshot, skipped")
            continue
        video_count += 1
        _done_vids.append(video_info['id'])
        ds_name = dataset.get_dataset_name(video_info['id'])

        runner.on_video_start()
        print(f"  video {video_count}/{dataset.num_videos()}: "
              f"{video_info.get('name')} frames={len(frames)} ds={ds_name} "
              f"score={runner.tracker.score_thresh}")
        for frame in frames:
            img_path = os.path.join(img_root, frame['file_name'])
            (boxes, scores, cats, ids,
             sources) = runner.process_frame(img_path)
            formatter.add_frame(
                boxes, scores, cats, ids,
                frame['id'], video_info['id'],
                sources=sources)
            total_dets += len(boxes)
        formatter.on_video_end(runner.next_id)
        # Release cached GPU memory between videos to cap the eval's peak
        # working set.
        torch.cuda.empty_cache()
        if args.snapshot_every > 0 and video_count % args.snapshot_every == 0:
            write_snapshot(f'v{video_count:04d}')
        if args.live_score_every > 0 and video_count % args.live_score_every == 0:
            write_live_score(video_count)

    elapsed = time.time() - start
    print(f'Inference done: {elapsed:.1f}s, total detections={total_dets}')
    formatter.finalize(track_json, use_majority_vote=not args.no_majority_vote)

    # Both splits are scored locally. The base/novel open-vocabulary breakdown is
    # val-specific, so on test a failure inside it must not cost us the COMBINED
    # line and the metrics; on val the same failure is a real error and propagates.
    try:
        run_teta_eval(subset_ann_file, track_json, output_dir, dataset)
    except Exception as _teta_e:
        if args.split == 'val':
            raise
        print(f'[test] base/novel split print skipped: {_teta_e!r}')

    metrics = _extract_teta_metrics(output_dir)
    metrics_path = os.path.join(output_dir, 'metrics.json')
    # Record what produced this number. Without the split, the video list and
    # the video count, a one-video probe and a full-split run leave products
    # that look identical afterwards; without the five build-time switches the
    # architecture that was evaluated cannot be recovered from the output.
    row = {
        'ckpt': ckpt_path,
        'tag': _checkpoint_tag(ckpt_path),
        'output_dir': output_dir,
        'split': args.split,
        'video_list': args.video_list,
        'max_videos': args.max_videos,
        'num_videos_scored': dataset.num_videos(),
        'chunked_size': CHUNKED_SIZE,
        'score_thresh': SCORE_THRESH,
        'filter_score_thresh': args.filter_score_thresh,
        'miss_tolerance': MISS_TOLERANCE,
        'arch_switches': {k: os.environ.get(k)
                          for k in ('INGRAIN_TRACK_CONTENT_ENCROI',
                                    'INGRAIN_HIDDEN_FUSE',
                                    'INGRAIN_ENCROI_MEM_FUSE',
                                    'INGRAIN_MEMFUSE_INTERLEAVE',
                                    'INGRAIN_TRACK_SAMPLING_ADAPTER')},
        **metrics,
    }
    with open(metrics_path, 'w') as f:
        json.dump(row, f, indent=2)
    print(f'Wrote metrics: {metrics_path}')
    del model
    torch.cuda.empty_cache()
    return row


def run_tao_subset_eval(args):
    ckpts = _selected_checkpoints(args)
    if not ckpts:
        raise ValueError('--tao-subset-eval requires --checkpoint or --checkpoints')
    requested_names = _collect_requested_video_names(args)
    if args.output_dir is None:
        args.output_dir = 'eval/results/subset_eval'
    os.makedirs(args.output_dir, exist_ok=True)

    ann_file = (args.tao_ann if getattr(args, 'tao_ann', None)
                else (TAO_VAL_ANN if args.split == 'val' else TAO_TEST_ANN))
    subset_ann_file = os.path.join(args.output_dir, 'subset_gt.json')
    _write_tao_subset_ann(
        ann_file, requested_names, subset_ann_file, max_videos=args.max_videos)

    rows = []
    multi = len(ckpts) > 1
    for ckpt in ckpts:
        tag = _checkpoint_tag(ckpt)
        ckpt_output_dir = (os.path.join(args.output_dir, tag)
                           if multi else args.output_dir)
        row = _run_tao_subset_for_checkpoint(
            args, ckpt, subset_ann_file, ckpt_output_dir)
        rows.append(row)
    _write_compare_summary(rows, args.output_dir)


def main():
    args = parse_args()
    import builtins as _b
    # Build the decoupling stage's separate track path at eval so the ckpt's trained
    # decoder.track_dec_* weights load (else track re-runs the detection path = skew).
    _b._INGRAIN_TRACK_PATH = bool(getattr(args, 'track_path', False))
    # The track path spans only the first num_isolation_layers (the decoupling
    # stage); MUST match training.
    _b._INGRAIN_STAGED_DECODER = bool(
        getattr(args, 'staged_decoder', False))
    # Optional hard GPU-memory cap (env GPU_MEM_FRAC = fraction of total device
    # memory). Lets an early eval run concurrently with a training process
    # without being able to OOM it: if this process exceeds the cap it OOMs
    # itself, never the trainer. No-op when the env var is unset.
    import os as _os
    _gmf = _os.environ.get('GPU_MEM_FRAC')
    if _gmf and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(_gmf), 0)
        print(f'[eval] GPU_MEM_FRAC cap = {_gmf} of device-0 total memory')
    if not args.tao_subset_eval:
        raise SystemExit(
            'evaluate.py evaluates on the TAO val and test splits. Pass '
            '--tao-subset-eval; scripts/eval.sh does.')
    run_tao_subset_eval(args)


if __name__ == '__main__':
    main()
