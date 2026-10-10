"""Minimal runtime helpers used by INGRAIN evaluation."""


import cv2
import numpy as np
import torch

from ingrain.structures.track_instances import TrackInstances



class RuntimeTrackerBase:
    """Track lifecycle manager used at evaluation time.

    Newborn pipeline:
      1. score_thresh — basic confidence filter on detection slots.
      2. Every surviving detection slot receives the next available obj_id.

    Track death:
      - score < filter_score_thresh → disappear_time += 1.
      - When disappear_time >= miss_tolerance → obj_idxes := -1.

    Per-ID collision cleanup:
      - Defensive cleanup: if two active slots ever share an obj_id, keep
        the higher-score one.

    """

    def __init__(self,
                 score_thresh: float = 0.5,
                 filter_score_thresh: float = 0.5,
                 miss_tolerance: int = 5,
                 use_iou_gate: bool = False,
                 iou_thresh: float = 0.5,
                 use_inter_track_nms: bool = False):
        self.score_thresh = float(score_thresh)
        self.filter_score_thresh = float(filter_score_thresh)
        self.miss_tolerance = int(miss_tolerance)
        self.use_iou_gate = bool(use_iou_gate)
        self.iou_thresh = float(iou_thresh)
        self.use_inter_track_nms = bool(use_inter_track_nms)
        self.max_obj_id = 0
        self._frame_idx = 0

    def reset(self):
        self.max_obj_id = 0
        self._frame_idx = 0
        self._frame_died_ids: list = []

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2,
                            cx + w / 2, cy + h / 2], dim=-1)

    @staticmethod
    def _pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a: (M, 4), b: (N, 4), both xyxy. Returns (M, N).
        if a.numel() == 0 or b.numel() == 0:
            return a.new_zeros((a.shape[0], b.shape[0]))
        lt = torch.max(a[:, None, :2], b[None, :, :2])
        rb = torch.min(a[:, None, 2:], b[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).clamp(min=1e-6)
        area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).clamp(min=1e-6)
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / union.clamp(min=1e-6)

    def update(self, track_instances: TrackInstances, num_det: int = 900):
        """In-place update of ``obj_idxes`` and ``disappear_time``."""
        scores = track_instances.scores
        obj_idxes = track_instances.obj_idxes

        # The death countdown resets at the DEATH threshold
        # (filter_score_thresh), not the birth threshold (score_thresh).
        high_mask = scores >= self.filter_score_thresh
        track_instances.disappear_time[high_mask] = 0

        # Ids holding a slot on entry. Death is reported at the END of update()
        # as the set difference against the ids still holding one — see the
        # note there for why per-branch bookkeeping is wrong.
        _oids_before = {int(o) for o in obj_idxes.tolist() if o >= 0}

        # Pre-compute xyxy boxes for IoU gating (only if needed).
        boxes_xyxy = None
        if (self.use_iou_gate or self.use_inter_track_nms) and \
                hasattr(track_instances, 'pred_boxes'):
            boxes_xyxy = self._cxcywh_to_xyxy(track_instances.pred_boxes)

        # Candidate dets: obj_idxes == -1 AND score >= score_thresh.
        det_candidates = []
        for i in range(num_det):
            if (obj_idxes[i].item() == -1
                    and scores[i].item() >= self.score_thresh):
                det_candidates.append((i, scores[i].item()))
        det_candidates.sort(key=lambda x: -x[1])
        for i, _score in det_candidates:
            if self.use_iou_gate and boxes_xyxy is not None:
                claimed = obj_idxes >= 0
                if claimed.any():
                    ious = self._pairwise_iou_xyxy(
                        boxes_xyxy[i:i + 1], boxes_xyxy[claimed])
                    if ious.numel() > 0 and ious.max().item() > self.iou_thresh:
                        continue
            obj_idxes[i] = self.max_obj_id
            self.max_obj_id += 1

        # Track death by exhausted miss tolerance.
        for i in range(len(track_instances)):
            obj_id = obj_idxes[i].item()
            score = scores[i].item()
            if obj_id >= 0 and score < self.filter_score_thresh:
                track_instances.disappear_time[i] += 1
                if track_instances.disappear_time[i] >= self.miss_tolerance:
                    obj_idxes[i] = -1

        if self.use_inter_track_nms and boxes_xyxy is not None:
            active_pos = (obj_idxes >= 0).nonzero(as_tuple=True)[0]
            if active_pos.numel() >= 2:
                active_oids = obj_idxes[active_pos]
                # Sort by obj_id ascending = oldest first (older birth = lower id)
                order = active_oids.argsort()
                sorted_slots = active_pos[order].tolist()
                kept = []
                for slot in sorted_slots:
                    if kept:
                        stack = boxes_xyxy[kept]
                        ious = self._pairwise_iou_xyxy(
                            boxes_xyxy[slot:slot + 1], stack)
                        if (ious.numel() > 0
                                and ious.max().item() > self.iou_thresh):
                            obj_idxes[slot] = -1
                            continue
                    kept.append(slot)

        # Per-ID defensive cleanup (keep higher-score slot if obj_id collision).
        active_pos = (obj_idxes >= 0).nonzero(as_tuple=True)[0]
        if active_pos.numel() >= 2:
            best_by_id = {}
            for slot in active_pos.tolist():
                oid = int(obj_idxes[slot].item())
                score = float(scores[slot].item())
                prev = best_by_id.get(oid)
                if prev is None:
                    best_by_id[oid] = (slot, score)
                    continue
                prev_slot, prev_score = prev
                if score > prev_score:
                    obj_idxes[prev_slot] = -1
                    best_by_id[oid] = (slot, score)
                else:
                    obj_idxes[slot] = -1

        # Report the ids that lost their last slot, so the caller can release
        # whatever else is keyed by them, such as the trajectory memory of a
        # track that just died. Computed as a set difference rather than
        # appended at each kill site, because the kill sites are NOT equivalent:
        # the miss-tolerance and inter-track-NMS branches retire an id, but the
        # per-id cleanup above (and NMS when two slots share an id) drops a
        # DUPLICATE slot while the id lives on elsewhere. Recording at those
        # sites would evict the memory of a track that is still being tracked.
        # The difference is correct by construction: an id is reported dead iff
        # it holds no active slot on exit. The tracker has no handle on the
        # model, so it only reports; the runner does the eviction.
        _oids_after = {int(o) for o in obj_idxes.tolist() if o >= 0}
        self._frame_died_ids = sorted(_oids_before - _oids_after)

        self._frame_idx += 1


_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preprocess(img_bgr: np.ndarray,
               target_short: int = 800,
               max_size: int = 1333):
    """BGR uint8 -> (1, 3, H, W) float tensor. Returns unpadded (H, W)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = min(target_short / min(h, w), max_size / max(h, w))
    nh, nw = int(h * scale), int(w * scale)
    img = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img = (img.astype(np.float32) - _MEAN) / _STD
    ph = (32 - nh % 32) % 32
    pw = (32 - nw % 32) % 32
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)),
                     mode='constant', constant_values=0.0)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return tensor, (nh, nw)
