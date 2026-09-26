#!/usr/bin/env python3
"""Track your own video with INGRAIN.

Give a video (or a directory of frames) and the category words you care
about; INGRAIN tracks them — including categories absent from the training
annotations — and writes an annotated copy.

  python demo_video.py --video your.mp4 \
      --words "person . lawn_mower . dog" \
      --checkpoint ckpt/ingrain.pth

The model is built exactly as in scripts/eval.sh (same architecture switches,
propagation and lifecycle); only the vocabulary is yours. With 40 words or
fewer the whole prompt fits one text chunk, so tracking costs one decoder pass
per frame. The default birth/death thresholds (0.30 / 0.15) are the same
values TAO benchmark scoring uses (scripts/eval.sh).
"""
import argparse
import os
import sys
import tempfile

# Architecture switches, before any model code is imported. Same five as
# scripts/train.sh / scripts/eval.sh; setdefault so an ablation checkpoint can
# still be driven with e.g. INGRAIN_HIDDEN_FUSE=0.
os.environ.setdefault('INGRAIN_TRACK_CONTENT_ENCROI', '1')
os.environ.setdefault('INGRAIN_HIDDEN_FUSE', '1')
os.environ.setdefault('INGRAIN_ENCROI_MEM_FUSE', '1')
os.environ.setdefault('INGRAIN_MEMFUSE_INTERLEAVE', '1')
os.environ.setdefault('INGRAIN_TRACK_SAMPLING_ADAPTER', '1')
os.environ.setdefault('INGRAIN_TSA_DIM', '128')
os.environ.setdefault('INGRAIN_NO_PIN_MEMORY', '1')


def parse_words(spec: str):
    """'person . lawn_mower . dog' -> ['person', 'lawn_mower', 'dog']"""
    words = [w.strip() for w in spec.replace(',', '.').split('.')]
    words = [w for w in words if w]
    seen, out = set(), []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def id_color(track_id: int):
    """Stable, well-separated BGR colour per track id (golden-angle hue)."""
    import colorsys
    hue = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_tracks(frame, boxes, scores, cats, ids, class_names):
    import cv2
    for (x1, y1, x2, y2), score, cat, tid in zip(boxes, scores, cats, ids):
        color = id_color(int(tid))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, color, 2)
        label = f'{class_names[int(cat) - 1]} #{int(tid)} {score:.2f}'
        (tw, th), base = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = p1[1] - 4 if p1[1] - th - base - 4 >= 0 else p1[1] + th + base + 4
        cv2.rectangle(frame, (p1[0], ty - th - base),
                      (p1[0] + tw + 2, ty + base), color, -1)
        cv2.putText(frame, label, (p1[0] + 1, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def iter_video_frames(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f'cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, fps
    finally:
        cap.release()


def iter_dir_frames(path, fps):
    import cv2
    exts = ('.jpg', '.jpeg', '.png')
    names = sorted(n for n in os.listdir(path)
                   if n.lower().endswith(exts))
    if not names:
        raise SystemExit(f'no image frames under {path}')
    for n in names:
        frame = cv2.imread(os.path.join(path, n))
        if frame is not None:
            yield frame, fps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video', help='input video file')
    src.add_argument('--frames', help='directory of image frames '
                                      '(sorted by filename)')
    ap.add_argument('--words', required=True,
                    help="category words, ' . '-separated, e.g. "
                         "'person . lawn_mower . dog'")
    ap.add_argument('--checkpoint', default='ckpt/ingrain.pth',
                    help='trained INGRAIN checkpoint')
    ap.add_argument('--gd-ckpt', default=None,
                    help='Grounding DINO base weights '
                         '(default: the path train.py resolves)')
    ap.add_argument('--out', default=None,
                    help='output video path '
                         '(default: <input stem>_tracked.mp4)')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='output fps when --frames is used '
                         '(--video keeps the source fps)')
    ap.add_argument('--score-thresh', type=float, default=0.30,
                    help='birth threshold: a detection must score this high '
                         'to start a track')
    ap.add_argument('--filter-score-thresh', type=float, default=0.15,
                    help='death threshold: below this a track ages toward '
                         'removal')
    ap.add_argument('--miss-tolerance', type=int, default=5,
                    help='frames a track may stay below the death threshold')
    ap.add_argument('--max-dets', type=int, default=100,
                    help='max tracks reported per frame')
    ap.add_argument('--device', default=None,
                    help='cuda | cpu (default: cuda if available)')
    args = ap.parse_args()

    words = parse_words(args.words)
    if not words:
        raise SystemExit('--words parsed to an empty list')

    import cv2
    import torch

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print('WARNING: no CUDA device; this will be very slow.',
              file=sys.stderr)

    # The decoupling-stage track path must be configured before the overlay,
    # exactly as evaluate.py does for the TAO protocol.
    import builtins as _b
    _b._INGRAIN_TRACK_PATH = True
    _b._INGRAIN_STAGED_DECODER = True

    from evaluate import load_model
    from eval.ingrain_runner import IngrainRunner

    print(f'words ({len(words)}): {" . ".join(words)}')
    model = load_model(
        args.checkpoint,
        gd_ckpt_path=args.gd_ckpt,
        trajectory_state_prop=True,
        memory_attn_residual_scale=0.1,
        mem_aggregation='attention',
        mem_interleave_depths=3,
        mem_ema_alpha=0.9,
        isolation_layers=3)
    model = model.to(device).eval()

    runner = IngrainRunner(
        model, words, list(range(1, len(words) + 1)),
        chunked_size=40,
        score_thresh=args.score_thresh,
        filter_score_thresh=args.filter_score_thresh,
        miss_tolerance=args.miss_tolerance,
        max_dets=args.max_dets,
        topk_cls=1,
        e2e_assoc=True,
        use_inter_track_nms=True,
        iou_thresh=0.5)
    runner.on_video_start()

    if args.video:
        frames = iter_video_frames(args.video)
        stem = os.path.splitext(args.video)[0]
    else:
        frames = iter_dir_frames(args.frames, args.fps)
        stem = os.path.normpath(args.frames)
    out_path = args.out or f'{stem}_tracked.mp4'

    writer = None
    n_frames = 0
    all_ids = set()
    with tempfile.TemporaryDirectory(prefix='ingrain_demo_') as tmp:
        # IngrainRunner consumes image paths; one scratch file, rewritten per
        # frame, keeps the disk footprint at a single frame.
        scratch = os.path.join(tmp, 'frame.jpg')
        for frame, fps in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f'cannot open writer for {out_path}')
            cv2.imwrite(scratch, frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            boxes, scores, cats, ids, _sources = runner.process_frame(scratch)
            all_ids.update(int(i) for i in ids)
            writer.write(draw_tracks(frame, boxes, scores, cats, ids, words))
            n_frames += 1
            if n_frames % 20 == 0:
                print(f'  frame {n_frames}: {len(ids)} active tracks')
    if writer is None:
        raise SystemExit('no readable frames in the input')
    writer.release()
    print(f'{n_frames} frames, {len(all_ids)} tracks -> {out_path}')


if __name__ == '__main__':
    main()
