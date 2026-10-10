#!/usr/bin/env python
"""Score a (partial) TAO tracking output with TETA, restricted to whatever
videos the prediction covers — so you can score a mid-run
`tao_track_snapshot_vNNNN.json` without waiting for the full 1419-video test set.

The federated TETA mean is computed over exactly the classes present in the
subset GT, identically to the full-evaluation scorer.

Portable: TETA is imported from the vendored `teta/` (see eval/config.py), so a
fresh copy of this package scores with no separate TETA install. The scoring itself
needs only numpy + scipy, but importing `eval.config` re-exports train.py's
model config, so torch/mmdet/mmcv have to be installed too.

Usage
-----
  # score one snapshot
  python scripts/score_partial_snapshot.py \
      --pred eval/results/.../tao_track_snapshot_v0080.json \
      --gt   /path/to/tao_test_annotations.json \
      --out  eval/results/_snap80_score

  # score a second set of predictions on the SAME videos, for a like-for-like
  # comparison
  python scripts/score_partial_snapshot.py \
      --pred .../tao_track_snapshot_v0080.json \
      --gt   /path/to/tao_test_annotations.json \
      --out  eval/results/_snap80_compare \
      --compare /path/to/other/tao_track.json --compare-name OTHER

Notes
-----
* --pred / --compare are TAO-format lists of dets: each item has at least
  image_id, category_id, bbox, score, track_id (video_id optional; derived
  from GT when absent).
* --gt is the FULL TAO test/val annotation json (videos/images/annotations/
  categories). The subset is auto-carved from the videos the --pred covers.
* Both prediction sets are filtered to the SAME subset images before scoring,
  so the comparison is like-for-like.
"""
import os, sys, json, pickle, argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from eval.config import TETA_PACKAGE_PATH  # noqa: E402
sys.path.insert(0, TETA_PACKAGE_PATH)
import teta  # noqa: E402

COLS = ['TETA', 'LocA', 'AssocA', 'ClsA']


def _load(p):
    with open(p) as f:
        return json.load(f)


def build_subset_gt(full_gt, video_ids):
    """Carve a TETA-scorable GT restricted to `video_ids` (keep full category
    list so federated scoring is unchanged)."""
    sub = {'info': full_gt.get('info', {}),
           'licenses': full_gt.get('licenses', []),
           'categories': full_gt['categories']}
    sub['videos'] = [v for v in full_gt['videos'] if v['id'] in video_ids]
    keep_img = set()
    sub['images'] = []
    for im in full_gt['images']:
        if im.get('video_id') in video_ids:
            sub['images'].append(im)
            keep_img.add(im['id'])
    sub['annotations'] = [a for a in full_gt['annotations']
                          if a['image_id'] in keep_img]
    sub['tracks'] = [t for t in full_gt.get('tracks', [])
                     if t.get('video_id') in video_ids]
    return sub, keep_img


def score(pred_records, keep_img, gt_path, out_dir, tracker_name):
    """Filter tracker dets to the subset images and run TETA. Returns the
    federated overall [TETA, LocA, AssocA, ClsA] and the #classes scored."""
    filt = [d for d in pred_records if d['image_id'] in keep_img]
    cov = len(set(d['image_id'] for d in filt))
    fj = os.path.join(out_dir, f'trk_{tracker_name}.json')
    with open(fj, 'w') as f:
        json.dump(filt, f)

    ec = teta.config.get_default_eval_config()
    ec['PRINT_ONLY_COMBINED'] = True
    ec['DISPLAY_LESS_PROGRESS'] = True
    ec['OUTPUT_TEM_RAW_DATA'] = True
    ec['OUTPUT_PER_SEQ_RES'] = False
    ec['NUM_PARALLEL_CORES'] = int(os.environ.get('INGRAIN_TETA_CORES', '4'))
    dc = teta.config.get_default_dataset_config()
    dc['TRACKERS_TO_EVAL'] = [tracker_name]
    dc['GT_FOLDER'] = gt_path
    dc['OUTPUT_FOLDER'] = out_dir
    dc['TRACKER_SUB_FOLDER'] = fj
    teta.Evaluator(ec).evaluate([teta.datasets.TAO(dc)], [teta.metrics.TETA()])

    res = pickle.load(open(os.path.join(out_dir, tracker_name,
                                        'teta_summary_results.pth'), 'rb'))
    if 'COMBINED_SEQ' in res:
        res = res['COMBINED_SEQ']
    # 'average' is TETA's own pseudo-class row, not a real category.
    vecs = [np.array(res[k]['TETA'][50][:4]).astype(float)
            for k in res if k != 'average']
    ov = np.mean(np.stack(vecs), axis=0)
    print(f"  [{tracker_name}] dets={len(filt)} "
          f"covered_imgs={cov}/{len(keep_img)} classes={len(vecs)}")
    return ov, len(vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True,
                    help='tracking output to score (tao_track / snapshot json)')
    ap.add_argument('--gt', required=True,
                    help='FULL TAO test/val annotation json')
    ap.add_argument('--out', required=True, help='output dir for scoring artifacts')
    ap.add_argument('--compare', default=None,
                    help='optional 2nd tracker json to score on the same videos')
    ap.add_argument('--compare-name', default='OTHER')
    ap.add_argument('--name', default='PRED', help='display name for --pred')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pred = _load(args.pred)
    full_gt = _load(args.gt)
    name_by_vid = {v['id']: v['name'] for v in full_gt['videos']}

    # subset = videos the prediction covers (by video_id if present, else via GT
    # image->video map)
    img2vid = {im['id']: im.get('video_id') for im in full_gt['images']}
    vids = set()
    for d in pred:
        vids.add(d.get('video_id', img2vid.get(d['image_id'])))
    vids.discard(None)

    # domain composition (TAO video name = split/DOMAIN/clip)
    from collections import Counter
    dom = Counter()
    for vid in vids:
        parts = name_by_vid.get(vid, '').split('/')
        dom[parts[1] if len(parts) > 1 else (parts[0] or '?')] += 1
    print(f"subset: {len(vids)} videos  domains={dict(dom)}")

    sub_gt, keep_img = build_subset_gt(full_gt, vids)
    gt_path = os.path.join(args.out, 'subset_gt.json')
    with open(gt_path, 'w') as f:
        json.dump(sub_gt, f)
    print(f"subset GT: videos={len(sub_gt['videos'])} images={len(keep_img)} "
          f"anns={len(sub_gt['annotations'])} tracks={len(sub_gt['tracks'])} "
          f"cats={len(sub_gt['categories'])}\n")

    print(f"--- scoring {args.name} ---")
    a, na = score(pred, keep_img, gt_path, args.out, args.name)

    b = None
    if args.compare:
        print(f"--- scoring {args.compare_name} (same videos) ---")
        comp = _load(args.compare)
        b, nb = score(comp, keep_img, gt_path, args.out, args.compare_name)

    print("\n========== RESULT ("
          f"{len(vids)}v, federated over {na} classes) ==========")
    hdr = "".join(f"{c:>9s}" for c in COLS)
    print(f"{'':<13s}{hdr}")
    print(f"{args.name:<13s}" + "".join(f"{x:9.3f}" for x in a))
    if b is not None:
        print(f"{args.compare_name:<13s}" + "".join(f"{x:9.3f}" for x in b))
        print(f"{'Δ(P-C)':<13s}" + "".join(f"{a[i]-b[i]:+9.3f}" for i in range(4)))


if __name__ == '__main__':
    main()
