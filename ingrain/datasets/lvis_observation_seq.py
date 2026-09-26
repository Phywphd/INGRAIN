"""LVISObservationSeqDataset: loads LVIS images and builds augmented
observation sequences for MOT training.

Data files:
  lvis_base_train.json            — COCO-style JSON
  lvis_train_images.h5            — HDF5 keyed by bare filename, values are raw JPEG bytes
  lvis_classes_v1.txt             — 1203 class names, one per line
"""
import json
import math
import multiprocessing as mp
import random
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.utils.data


# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)   # RGB
_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


# ---------------------------------------------------------------------------
# Augmentation helpers (operate on numpy uint8 RGB images + xywh pixel boxes)
# ---------------------------------------------------------------------------

def _resize_image_and_boxes(img: np.ndarray,
                             boxes_xywh: np.ndarray,
                             target_short: int,
                             max_long: int = 1333
                             ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Resize *img* so its shorter side == *target_short* (cap long side at *max_long*).

    Returns:
        resized_img, resized_boxes_xywh (pixel coords), scale_factor
    """
    h, w = img.shape[:2]
    scale = min(target_short / min(h, w), max_long / max(h, w))
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if len(boxes_xywh):
        boxes_xywh = boxes_xywh * scale
    return img, boxes_xywh, scale


def _hflip_image_and_boxes(img: np.ndarray,
                            boxes_xywh: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal flip image and xywh pixel boxes."""
    img = img[:, ::-1, :].copy()
    if len(boxes_xywh):
        h, w = img.shape[:2]
        boxes_out = boxes_xywh.copy()
        boxes_out[:, 0] = w - boxes_xywh[:, 0] - boxes_xywh[:, 2]
        return img, boxes_out
    return img, boxes_xywh


def _color_jitter(img: np.ndarray,
                  brightness: float = 0.15,
                  saturation: float = 0.15,
                  hue: float = 0.05
                  ) -> np.ndarray:
    """Small colour jitter in HSV space (independent per frame)."""
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Hue shift
    img_hsv[:, :, 0] += random.uniform(-hue * 180, hue * 180)
    img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0], 0, 180)
    # Saturation
    img_hsv[:, :, 1] *= random.uniform(1 - saturation, 1 + saturation)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    # Value (brightness)
    img_hsv[:, :, 2] *= random.uniform(1 - brightness, 1 + brightness)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _translate_image_and_boxes(img: np.ndarray,
                                boxes_xywh: np.ndarray,
                                dx: float, dy: float
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Translate image by (dx, dy) pixels using warpAffine and shift boxes."""
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, dx],
                  [0.0, 1.0, dy]], dtype=np.float32)
    img = cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(int(_MEAN[0]), int(_MEAN[1]), int(_MEAN[2])))
    if len(boxes_xywh):
        boxes_out = boxes_xywh.copy()
        boxes_out[:, 0] += dx
        boxes_out[:, 1] += dy
        return img, boxes_out
    return img, boxes_xywh


_TRAJ_DIRS = [[0, 1], [1, 1], [1, 0], [1, -1],
                   [0, -1], [-1, -1], [-1, 0], [-1, 1]]


def _coordinated_trajectory(num_frames: int,
                     max_translate=(0.05, 0.10, 0.12, 0.15),
                     strength: float = 1.0) -> list:
    """Per-frame normalized (fx, fy) offsets along a random
    cardinal-direction path (turning point only when nf>4). strength
    scales the translation magnitude."""
    idx = min(max(num_frames - 2, 0), len(max_translate) - 1)
    mtr = max_translate[idx] * float(strength)
    tx_max = random.uniform(mtr / 3, mtr)
    ty_max = random.uniform(mtr / 3, mtr)

    def _walk(n, d, coords):
        out, c = [], list(coords)
        for _ in range(n):
            c = [c[0] + d[0], c[1] + d[1]]
            out.append(c)
        return out

    if num_frames > 4:
        turning = random.randint(1, num_frames - 1)
        t0, t1 = random.randint(0, 7), random.randint(0, 7)
        pre = _walk(turning, _TRAJ_DIRS[t0], [0, 0])
        path = pre + _walk(num_frames - turning, _TRAJ_DIRS[t1], pre[-1])
    else:
        path = _walk(num_frames, _TRAJ_DIRS[random.randint(0, 7)], [0, 0])

    arr = np.asarray(path, dtype=np.float32)
    xmin, xmax = float(arr[:, 0].min()), float(arr[:, 0].max())
    ymin, ymax = float(arr[:, 1].min()), float(arr[:, 1].max())
    xs = (xmax - xmin) / tx_max / 2 if tx_max > 0 else 0
    ys = (ymax - ymin) / ty_max / 2 if ty_max > 0 else 0
    xc, yc = (xmax + xmin) / 2, (ymax + ymin) / 2
    return [[(p[0] - xc) / xs if xs != 0 else 0.0,
             (p[1] - yc) / ys if ys != 0 else 0.0] for p in path]


def _coordinated_affine_frame(img: np.ndarray, boxes_xywh: np.ndarray,
                       ori_area: np.ndarray, fx: float, fy: float,
                       ratio_threshold: float = 0.3,
                       border_val=(114, 114, 114),
                       scale_fixed=None, rot_fixed=None):
    """Coordinated-trajectory affine for one frame: rotation about the image
    centre and scale about the origin, from the SMOOTH per-frame trajectory
    values (rot_fixed degrees, scale_fixed factor; None = identity), plus
    translation (fx, fy) from the trajectory. warpPerspective the image,
    warp box corners, then keep boxes with visible-area ratio > 0.3 AND drop
    degenerate boxes (>1px, area-frac>0.01, aspect<20). Returns
    (img, boxes_xywh_kept, keep_mask)."""
    h, w = img.shape[:2]
    rot = float(rot_fixed) if rot_fixed is not None else 0.0
    scale = float(scale_fixed) if scale_fixed is not None else 1.0
    rad = np.radians(rot)
    cx, cy = w / 2.0, h / 2.0
    R = np.array([[np.cos(rad), -np.sin(rad), 0.],
                  [np.sin(rad), np.cos(rad), 0.], [0., 0., 1.]], np.float32)
    T0 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float32)
    T1 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float32)
    rot_m = T1 @ R @ T0
    sc_m = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
    tr_m = np.array([[1, 0, fx * w], [0, 1, fy * h], [0, 0, 1]], np.float32)
    warp = (tr_m @ rot_m @ sc_m).astype(np.float32)
    img = cv2.warpPerspective(img, warp, (w, h), borderValue=border_val)
    if not len(boxes_xywh):
        return img, boxes_xywh, np.zeros(0, dtype=bool)
    b = boxes_xywh
    xyxy = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
    n = len(xyxy)
    xs = xyxy[:, [0, 0, 2, 2]].reshape(n * 4)
    ys = xyxy[:, [1, 3, 3, 1]].reshape(n * 4)
    pts = np.vstack([xs, ys, np.ones_like(xs)])
    wp = warp @ pts
    wp = wp[:2] / wp[2]
    wxs = wp[0].reshape(n, 4); wys = wp[1].reshape(n, 4)
    warp_xyxy = np.vstack((wxs.min(1), wys.min(1), wxs.max(1), wys.max(1))).T
    warp_xyxy[:, [0, 2]] = warp_xyxy[:, [0, 2]].clip(0, w)
    warp_xyxy[:, [1, 3]] = warp_xyxy[:, [1, 3]].clip(0, h)
    ww = warp_xyxy[:, 2] - warp_xyxy[:, 0]
    wh = warp_xyxy[:, 3] - warp_xyxy[:, 1]
    # find_inside_bboxes_v2: retained semantic area > 30% of pre-affine area
    cur_area = ww * wh / (scale ** 2)
    keep = (cur_area / (ori_area + 1e-16)) > ratio_threshold
    # filter_gt_bboxes: >1px, area-frac>0.01 vs scaled-original, aspect<20
    osc = xyxy * scale
    ow = osc[:, 2] - osc[:, 0]; oh = osc[:, 3] - osc[:, 1]
    aspect = np.maximum(ww / (wh + 1e-16), wh / (ww + 1e-16))
    keep = keep & (ww > 1) & (wh > 1) & \
        ((ww * wh / (ow * oh + 1e-16)) > 0.01) & (aspect < 20)
    out = np.stack([warp_xyxy[:, 0], warp_xyxy[:, 1], ww, wh], 1).astype(np.float32)
    return img, out[keep], keep


def _occlude_objects(img: np.ndarray,
                     boxes_xywh: np.ndarray,
                     occlude_mask: np.ndarray,
                     fill_colors=None,
                     ) -> np.ndarray:
    """Visually occlude selected objects by filling their bbox with a random color.

    Args:
        img:           (H, W, 3) uint8 RGB image, MODIFIED in-place
        boxes_xywh:    (N, 4) pixel [x, y, w, h]
        occlude_mask:  (N,) bool — True = occlude this object
        fill_colors:   list of (R,G,B) tuples to sample from

    Returns:
        img with occluded regions filled
    """
    if fill_colors is None:
        fill_colors = [
            (90, 100, 110), (110, 90, 75), (100, 85, 110),
            (90, 120, 100), (90, 75, 110), (100, 120, 95),
            (80, 80, 80), (120, 120, 120),
        ]
    for i in range(len(boxes_xywh)):
        if not occlude_mask[i]:
            continue
        x, y, w, h = boxes_xywh[i].astype(int)
        x2, y2 = x + w, y + h
        x = max(0, x); y = max(0, y)
        x2 = min(img.shape[1], x2); y2 = min(img.shape[0], y2)
        if x2 > x and y2 > y:
            color = fill_colors[random.randint(0, len(fill_colors) - 1)]
            img[y:y2, x:x2] = color
    return img


def _normalize_and_pad(img: np.ndarray
                       ) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Normalize with ImageNet stats, pad to 32-divisor, return CHW tensor."""
    img = img.astype(np.float32)
    img = (img - _MEAN) / _STD
    h, w = img.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img,
                     ((0, pad_h), (0, pad_w), (0, 0)),
                     mode='constant',
                     constant_values=0.0)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return tensor, (h, w)   # real (unpadded) shape


def _filter_boxes(boxes_xywh: np.ndarray,
                  img_h: int, img_w: int,
                  min_area: float = 1.0,
                  max_outside_frac: float = 0.9
                  ) -> np.ndarray:
    """Return boolean keep-mask for valid boxes (in xywh pixel coords).

    Removes boxes:
      - with area < min_area
      - that are more than *max_outside_frac* fraction outside the image
    """
    if not len(boxes_xywh):
        return np.zeros(0, dtype=bool)

    x, y, bw, bh = (boxes_xywh[:, i] for i in range(4))
    area = bw * bh
    area_ok = area >= min_area

    # Intersection with image
    ix1 = np.clip(x, 0, img_w)
    iy1 = np.clip(y, 0, img_h)
    ix2 = np.clip(x + bw, 0, img_w)
    iy2 = np.clip(y + bh, 0, img_h)
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    # Fraction outside = 1 - inter/area
    frac_outside = 1.0 - inter / (area + 1e-6)
    outside_ok = frac_outside <= max_outside_frac

    return area_ok & outside_ok


def _xywh_pixel_to_cxcywh_norm(boxes_xywh: np.ndarray,
                                img_h: int, img_w: int
                                ) -> np.ndarray:
    """Convert pixel [x,y,w,h] → normalized [cx,cy,w,h] in [0,1]."""
    if not len(boxes_xywh):
        return boxes_xywh.copy()
    out = boxes_xywh.copy().astype(np.float32)
    out[:, 0] = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0) / img_w   # cx
    out[:, 1] = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0) / img_h   # cy
    out[:, 2] = boxes_xywh[:, 2] / img_w                                # w
    out[:, 3] = boxes_xywh[:, 3] / img_h                                # h
    return out


# ---------------------------------------------------------------------------
# Repeat-factor sampling
# ---------------------------------------------------------------------------

def _compute_repeat_factors(image_annots: List[List[dict]],
                             num_images: int,
                             repeat_thr: float = 0.006
                             ) -> List[float]:
    """Compute per-image repeat factors for low-frequency-category oversampling.

    For each category c:
        freq(c) = |images containing c| / |total images|
        r(c)    = max(1, sqrt(repeat_thr / freq(c)))

    For each image i:
        rf(i)   = max over categories in image of r(c)

    Returns list of length num_images with repeat factors >= 1.
    """
    # Count images per category
    cat_image_count: Dict[int, int] = {}
    for annots in image_annots:
        cats_in_img = set(a['category_id'] for a in annots)
        for c in cats_in_img:
            cat_image_count[c] = cat_image_count.get(c, 0) + 1

    # Per-category repeat factor
    cat_rf: Dict[int, float] = {}
    for c, cnt in cat_image_count.items():
        freq = cnt / num_images
        cat_rf[c] = max(1.0, math.sqrt(repeat_thr / freq))

    # Per-image repeat factor
    rf_list = []
    for annots in image_annots:
        cats = set(a['category_id'] for a in annots)
        if cats:
            rf = max(cat_rf.get(c, 1.0) for c in cats)
        else:
            rf = 1.0
        rf_list.append(rf)
    return rf_list


def _build_repeat_indices(rf_list: List[float]) -> List[int]:
    """Turn per-image repeat factors into a flat index list."""
    indices = []
    for idx, rf in enumerate(rf_list):
        # Integer part always, fractional part by coin flip
        rep = int(math.floor(rf))
        if random.random() < (rf - rep):
            rep += 1
        indices.extend([idx] * rep)
    return indices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LVISObservationSeqDataset(torch.utils.data.Dataset):
    """Builds augmented observation sequences from LVIS still images for MOT
    training.

    Each item contains *num_frames* independent augmentations of the same
    image, forming a short observation sequence.  Track IDs (obj_ids) are the
    same across all observations for matching objects, and are taken directly
    from the annotation 'id' field.

    Args:
        ann_file (str):     Path to COCO-style JSON (lvis_base_train.json).
        h5_file (str):      Path to HDF5 file with JPEG image bytes.
        classes_file (str): Path to text file with one class name per line.
        num_frames (int):   Number of observations per sequence.  Default: 2.
        max_size (int):     Maximum long-side.  Default: 1333.
        repeat_thr (float): Repeat-factor threshold for low-frequency classes.
                            Default: 0.006.
        scales (list):      Short-side scale candidates.  Default: [640..800].
        seed (int):         Random seed for reproducibility of repeat indices.
    """

    _SCALES = [640, 672, 704, 736, 768, 800]

    def __init__(self,
                 ann_file: str,
                 h5_file: str,
                 classes_file: str,
                 num_frames: int = 2,
                 max_size: int = 1333,
                 repeat_thr: float = 0.006,
                 scales: Optional[List[int]] = None,
                 seed: int = 42,
                 occlude_ratio_early: float = 0.4,
                 motion_aug_strength: float = 1.0,
                 motion_aug_level: str = 'basic',
                 rot_amp: float = 0.0,
                 scale_amp: float = 0.0,
                 ):
        # Clip motion. 'basic': rigid whole-frame linear translation, so the
        # relative geometry between objects is preserved across the clip.
        # 'coordinated': coordinated translation trajectory plus SMOOTH
        # per-clip rotation (0 -> ±rot_amp degrees) and scale
        # (1 -> 1±scale_amp) ramps — coordinated ramps, never per-frame
        # jitter, so the displacement stays learnable. Shared Value so the
        # strength curriculum (train.py) can ramp it mid-epoch across forked
        # workers (mirrors _num_frames_val).
        self._motion_aug_strength_val = mp.Value('d', float(motion_aug_strength), lock=False)
        self.motion_aug_level = str(motion_aug_level)
        self.rot_amp = float(rot_amp)
        self.scale_amp = float(scale_amp)
        self.ann_file = ann_file
        self.h5_file = h5_file
        self.classes_file = classes_file
        # num_frames lives in a shared-memory Value so the sequence-length
        # curriculum can update it and forked DataLoader workers see the change.
        # lock=False: an int read/write is atomic; we only need eventual
        # visibility (the exact boundary batch may straddle, handled by the
        # collate's min-frame truncation).
        self._num_frames_val = mp.Value('i', int(num_frames), lock=False)
        self.max_size = max_size
        self.repeat_thr = repeat_thr
        self.scales = scales if scales is not None else self._SCALES
        # Fraction of objects hidden per clip: some enter late, some are
        # temporarily occluded mid-clip.
        self.occlude_ratio_early = occlude_ratio_early

        # H5 handle opened lazily in worker processes
        self._h5: Optional[h5py.File] = None

        # ---- Load annotations ----
        print(f"[LVISObservationSeqDataset] Loading annotations from {ann_file} ...")
        with open(ann_file, 'r') as f:
            data = json.load(f)

        self.images: List[dict] = data['images']   # list of {id, file_name, ...}
        annotations: List[dict] = data['annotations']

        # Build image_id → annotations mapping
        self._img2anns: Dict[int, List[dict]] = {}
        for img in self.images:
            self._img2anns[img['id']] = []
        for ann in annotations:
            img_id = ann['image_id']
            if img_id in self._img2anns:
                self._img2anns[img_id].append(ann)

        # ---- Load class names ----
        with open(classes_file, 'r') as f:
            self.class_names: List[str] = [line.strip() for line in f
                                           if line.strip()]
        # category_id is 1-indexed → 0-indexed
        # category_id 1 → index 0, etc.

        # ---- Compute repeat factors ----
        print("[LVISObservationSeqDataset] Computing repeat factors ...")
        image_annots = [self._img2anns[img['id']] for img in self.images]
        rf_list = _compute_repeat_factors(image_annots, len(self.images),
                                          repeat_thr=self.repeat_thr)

        rng_state = random.getstate()
        random.seed(seed)
        self.repeat_indices: List[int] = _build_repeat_indices(rf_list)
        random.setstate(rng_state)

        print(f"[LVISObservationSeqDataset] "
              f"{len(self.images)} images → {len(self.repeat_indices)} samples "
              f"(after repeat-factor sampling)")

    # -----------------------------------------------------------------------
    # H5 lazy open
    # -----------------------------------------------------------------------

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, 'r', swmr=True)
        return self._h5

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------

    def _load_image(self, file_name: str) -> np.ndarray:
        """Load image as HWC RGB numpy array from H5 JPEG bytes."""
        # Key is the bare filename, e.g. "000000391895.jpg"
        key = file_name.split('/')[-1]
        h5 = self._get_h5()
        jpeg_bytes = np.frombuffer(h5[key][()], dtype=np.uint8)
        img_bgr = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image: {key}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------------------------------------------------
    # Dataset protocol
    # -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.repeat_indices)

    @property
    def num_frames(self) -> int:
        """Observations per sequence. Backed by a shared Value so the frame-
        curriculum (train.py) can change it mid-epoch across forked workers."""
        return int(self._num_frames_val.value)

    def set_num_frames(self, n: int) -> None:
        self._num_frames_val.value = int(n)

    @property
    def motion_aug_strength(self) -> float:
        """Translate magnitude scale. Shared Value so the trainer can
        adjust it across forked workers."""
        return float(self._motion_aug_strength_val.value)

    def set_motion_aug_strength(self, s: float) -> None:
        self._motion_aug_strength_val.value = float(s)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.repeat_indices[idx]
        img_info = self.images[real_idx]
        img_id = img_info['id']
        # LVIS uses coco_url instead of file_name
        if 'file_name' in img_info:
            file_name = img_info['file_name']
        else:
            import os as _os
            file_name = _os.path.basename(img_info.get('coco_url', f'{img_id:012d}.jpg'))
        annots = self._img2anns.get(img_id, [])

        # ---- Load image ----
        img_rgb = self._load_image(file_name)
        orig_h, orig_w = img_rgb.shape[:2]

        # ---- Build GT arrays (pixel xywh + category_id + obj_id) ----
        # Filter out any annotation with zero-area bbox
        valid_annots = [a for a in annots
                        if a['bbox'][2] > 0 and a['bbox'][3] > 0]
        if valid_annots:
            gt_boxes_xywh = np.array([a['bbox'] for a in valid_annots],
                                     dtype=np.float32)          # (N,4) pixel xywh
            gt_cat_ids = np.array([a['category_id'] for a in valid_annots],
                                  dtype=np.int64)               # 1-indexed
            gt_obj_ids = np.array([a['id'] for a in valid_annots],
                                  dtype=np.int64)
        else:
            gt_boxes_xywh = np.zeros((0, 4), dtype=np.float32)
            gt_cat_ids = np.zeros(0, dtype=np.int64)
            gt_obj_ids = np.zeros(0, dtype=np.int64)

        # 0-indexed labels
        gt_labels_0idx = gt_cat_ids - 1   # shape (N,)

        # ---- Choose shared scale ----
        target_short = random.choice(self.scales)

        # ---- Per-frame translation offsets (smooth trajectory) ----
        # Base translation: random ±5% of image short side, scaled by
        # motion_aug_strength (1.0 = ±5%; larger values scale proportionally)
        trans_range = min(orig_h, orig_w) * 0.05 * float(getattr(self, 'motion_aug_strength', 1.0))
        base_dx = random.uniform(-trans_range, trans_range)
        base_dy = random.uniform(-trans_range, trans_range)

        # ---- 'coordinated' motion: coordinated trajectory + smooth ramps ----
        # Per-clip draws; magnitudes up to the configured amplitudes. The
        # rotation/scale values ramp smoothly 0 -> end across the clip
        # (coordinated = learnable), never per-frame random jitter.
        _coord = (getattr(self, 'motion_aug_level', 'basic') == 'coordinated')
        _traj = _scale_traj = _rot_traj = None
        if _coord:
            _nf = self.num_frames
            _traj = _coordinated_trajectory(
                _nf, max_translate=(0.05, 0.10, 0.12, 0.15),
                strength=float(getattr(self, 'motion_aug_strength', 1.0)))
            _msa = float(getattr(self, 'scale_amp', 0.0) or 0.0)
            if _msa > 0.0:
                _a = random.uniform(0.0, _msa)
                _s_end = (1.0 + _a) if random.random() < 0.5 else (1.0 - _a)
                _scale_traj = [1.0 + (_s_end - 1.0) * (t / max(_nf - 1, 1))
                               for t in range(_nf)]
            _mra = float(getattr(self, 'rot_amp', 0.0) or 0.0)
            if _mra > 0.0:
                _r_end = random.uniform(-_mra, _mra)
                _rot_traj = [_r_end * (t / max(_nf - 1, 1))
                             for t in range(_nf)]

        # ---- Shared flip decision (same for all frames to preserve track identity) ----
        do_flip = random.random() < 0.5

        # ---- Build frames ----
        imgs = []
        gt_instances_list = []
        img_shape_out = None

        # ---- Per-object visibility schedule across frames ----
        # Each object gets a bool array visible[num_frames].
        # Three patterns:
        #   Type A (always visible):    [T, T, T, T, T]  — majority
        #   Type B (late entry):        [F, F, T, T, T]  — appears mid-clip
        #   Type C (temporary occlude): [T, F, F, T, T]  — disappears then reappears
        #
        # Distribution controlled by occlude_ratio_early:
        #   ~(1-ratio) objects → Type A
        #   ~(ratio * 0.6) objects → Type B (late entry, biased to early frames)
        #   ~(ratio * 0.4) objects → Type C (temporary occlusion)
        #
        # Constraints:
        #   - At least 1 object visible in every frame
        #   - Type B: appear in frame 1 or 2 (not last frame)
        #   - Type C: occlude 1-2 consecutive frames in the middle, visible at start+end
        n_obj = len(gt_boxes_xywh)
        nf = self.num_frames
        # Default: all visible in all frames
        visibility = np.ones((n_obj, nf), dtype=bool)

        if n_obj > 1 and nf > 1:
            ratio = self.occlude_ratio_early
            for i in range(n_obj):
                r = random.random()
                if r < ratio * 0.6:
                    # Type B: late entry — occluded in frames 0..appear-1
                    # Bias toward early appearance: weighted toward frame 1
                    if nf <= 3:
                        appear = 1
                    else:
                        # 60% chance frame 1, 30% frame 2, 10% frame 3+
                        rr = random.random()
                        if rr < 0.6:
                            appear = 1
                        elif rr < 0.9:
                            appear = 2
                        else:
                            appear = min(3, nf - 2)  # leave ≥2 frames visible
                    visibility[i, :appear] = False

                elif r < ratio:
                    # Type C: temporary occlusion — visible, then hidden, then visible
                    if nf >= 4:
                        # Pick 1-2 consecutive frames to hide in the middle
                        occ_len = random.choice([1, 2]) if nf >= 5 else 1
                        # Start of occlusion: between frame 1 and nf-2
                        max_start = nf - occ_len - 1  # leave last frame visible
                        occ_start = random.randint(1, max(1, max_start))
                        visibility[i, occ_start:occ_start + occ_len] = False
                    elif nf == 3:
                        # 3 frames: hide in frame 1 only → [T, F, T]
                        visibility[i, 1] = False
                    # nf == 2: can't do temporary occlusion, stay Type A

            # Ensure at least 1 object visible in every frame
            for t in range(nf):
                if not visibility[:, t].any():
                    # Pick a random object to make visible
                    visibility[random.randint(0, n_obj - 1), t] = True

        for t in range(self.num_frames):
            img = img_rgb.copy()
            boxes = gt_boxes_xywh.copy()

            # 1. Resize (same scale for all frames)
            img, boxes, _ = _resize_image_and_boxes(
                img, boxes, target_short, self.max_size)
            cur_h, cur_w = img.shape[:2]

            # 2. Apply visibility schedule: occlude invisible objects
            frame_occlude = ~visibility[:, t]
            if frame_occlude.any():
                img = _occlude_objects(img, boxes, frame_occlude)
                keep_mask = visibility[:, t]
                boxes = boxes[keep_mask]
                frame_labels_raw = gt_labels_0idx[keep_mask]
                frame_obj_ids_raw = gt_obj_ids[keep_mask]
            else:
                frame_labels_raw = gt_labels_0idx.copy()
                frame_obj_ids_raw = gt_obj_ids.copy()

            # 3. Horizontal flip (shared across frames to preserve track identity)
            if do_flip:
                img, boxes = _hflip_image_and_boxes(img, boxes)

            # 4. Photometric jitter, independent per frame.
            img = _color_jitter(img)

            # 5. Motion. 'coordinated': coordinated whole-frame translation from the
            #    coordinated trajectory plus the smooth rotation/scale ramp values for
            #    this frame, applied as one affine about the image centre; the
            #    affine's inside/degenerate filter drops boxes warped out of
            #    view, so labels/ids are masked with it. 'basic': rigid
            #    whole-frame linear translation along the clip, so relative
            #    geometry between objects is preserved and the trajectory is a
            #    learnable displacement.
            if _coord:
                fx_t, fy_t = _traj[t]
                _sf = _scale_traj[t] if _scale_traj is not None else None
                _rf = _rot_traj[t] if _rot_traj is not None else None
                _ori_area = (boxes[:, 2] * boxes[:, 3] if len(boxes)
                             else np.zeros(0, dtype=np.float32))
                img, boxes, _keep_aff = _coordinated_affine_frame(
                    img, boxes, _ori_area, fx_t, fy_t,
                    scale_fixed=_sf, rot_fixed=_rf)
                if len(_keep_aff):
                    frame_labels_raw = frame_labels_raw[_keep_aff]
                    frame_obj_ids_raw = frame_obj_ids_raw[_keep_aff]
            else:
                dx_t = t * base_dx * (cur_w / orig_w)
                dy_t = t * base_dy * (cur_h / orig_h)
                if abs(dx_t) > 0.5 or abs(dy_t) > 0.5:
                    img, boxes = _translate_image_and_boxes(img, boxes, dx_t, dy_t)

            # 6. Filter out invalid boxes after augmentation
            if len(boxes):
                keep = _filter_boxes(boxes, cur_h, cur_w,
                                     min_area=1.0, max_outside_frac=0.9)
                boxes = boxes[keep]
                frame_labels = frame_labels_raw[keep]
                frame_obj_ids = frame_obj_ids_raw[keep]
            else:
                frame_labels = np.zeros(0, dtype=np.int64)
                frame_obj_ids = np.zeros(0, dtype=np.int64)

            # 6. Normalize + pad
            img_tensor, real_shape = _normalize_and_pad(img)
            if img_shape_out is None:
                img_shape_out = real_shape

            # 7. Convert boxes to normalized cxcywh
            if len(boxes):
                # Clip boxes to image boundary before conversion
                boxes[:, 0] = np.clip(boxes[:, 0], 0, cur_w)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, cur_h)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, cur_w - boxes[:, 0])
                boxes[:, 3] = np.clip(boxes[:, 3], 0, cur_h - boxes[:, 1])
                norm_cxcywh = _xywh_pixel_to_cxcywh_norm(boxes, cur_h, cur_w)
            else:
                norm_cxcywh = np.zeros((0, 4), dtype=np.float32)

            gt_inst = {
                'boxes': torch.from_numpy(norm_cxcywh).float(),    # (N,4) norm cxcywh
                'labels': torch.from_numpy(frame_labels).long(),    # (N,) 0-indexed
                'obj_ids': torch.from_numpy(frame_obj_ids).long(),  # (N,)
            }
            imgs.append(img_tensor)
            gt_instances_list.append(gt_inst)

        return {
            'imgs': imgs,                       # list of num_frames tensors (3,H,W)
            'gt_instances': gt_instances_list,  # list of num_frames dicts
            'img_shape': img_shape_out,         # (H, W) unpadded
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def mot_collate_fn(batch: list):
    """Collate for MOT training.

    bs=1: unwrap to single dict (backward compatible).
    bs>1: pad images to batch-max size, return batched structure.
    """
    if len(batch) == 1:
        return batch[0]

    # bs > 1: pad all frames to the same (max) spatial size.
    # Use the MIN frame count across the batch: during a sequence-length
    # curriculum transition a batch may straddle the boundary (some samples
    # generated at the old count, some at the new). Truncating to the min keeps
    # every per-frame stack rectangular; only boundary batches are affected.
    num_frames = min(len(b['imgs']) for b in batch)
    bs = len(batch)

    # Find max H, W across all samples and all frames
    max_h = max(batch[i]['imgs'][t].shape[1]
                for i in range(bs) for t in range(num_frames))
    max_w = max(batch[i]['imgs'][t].shape[2]
                for i in range(bs) for t in range(num_frames))
    # Round up to 32
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32

    # Stack per-frame: imgs_batched[t] = (bs, 3, max_h, max_w)
    imgs_batched = []
    for t in range(num_frames):
        padded = []
        for i in range(bs):
            img = batch[i]['imgs'][t]  # (3, h, w)
            c, h, w = img.shape
            pad = torch.zeros(c, max_h, max_w, dtype=img.dtype)
            pad[:, :h, :w] = img
            padded.append(pad)
        imgs_batched.append(torch.stack(padded))  # (bs, 3, max_h, max_w)

    # Collect per-sample metadata (truncate to the batch's common frame count)
    gt_instances_list = [batch[i]['gt_instances'][:num_frames] for i in range(bs)]  # bs x num_frames
    img_shapes = [batch[i]['img_shape'] for i in range(bs)]           # bs x (H, W)

    return {
        'imgs': imgs_batched,              # num_frames x (bs, 3, H, W)
        'gt_instances': gt_instances_list,  # bs x num_frames x dict
        'img_shapes': img_shapes,           # bs x (H, W)
        'batch_size': bs,
    }
