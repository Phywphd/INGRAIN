"""INGRAIN Track Criterion: multi-frame matching + loss computation.

Handles:
- Track query matching by obj_id (direct assignment, no Hungarian)
- Detect query matching via Hungarian assignment to remaining/all GT
- positive_map-based focal loss for open-vocabulary classification
- L1 + GIoU box regression losses
- Cross-frame loss aggregation

Losses are computed on the last decoder layer only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from mmdet.registry import MODELS


@MODELS.register_module()
class INGRAINTrackCriterion(nn.Module):
    """Loss criterion for INGRAIN multi-object tracker.

    Computes per-frame losses with track-aware matching:
    track queries are matched to GT by obj_id, detect queries
    are matched to remaining GT by Hungarian assignment.
    Uses focal classification and L1 box losses with positive_map.
    """

    def __init__(self,
                 num_classes=256,
                 loss_cls_weight=1.0,
                 loss_box_weight=5.0,
                 giou_ratio=0.4,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 match_cost_cls=2.0,
                 match_cost_bbox=5.0,
                 match_cost_giou=2.0,
                 loss_cls_det_weight=0.0,
                 det_pool_mode='remaining',
                 dedup_bg_weight=2.0,
                 dedup_overlap_thr=0.5):
        super().__init__()
        self.num_classes = num_classes  # max_text_len
        self.loss_cls_weight = loss_cls_weight
        # lambda_box, and the GIoU weight relative to L1 inside the box loss:
        #   L_box = lambda_box * (L1 + giou_ratio * GIoU)
        # giou_ratio = 0.4 gives the standard 5:2 L1:GIoU ratio.
        self.loss_box_weight = loss_box_weight
        self.giou_ratio = giou_ratio
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.match_cost_cls = match_cost_cls
        self.match_cost_bbox = match_cost_bbox
        # GIoU in the Hungarian cost (cls + L1 + GIoU).
        self.match_cost_giou = match_cost_giou
        # Optional det-slice loss on Hungarian-matched det queries.
        self.loss_cls_det_weight = float(loss_cls_det_weight)
        if det_pool_mode not in ('remaining', 'all', 'dedup'):
            raise ValueError(
                f"det_pool_mode must be 'remaining'|'all'|'dedup', got "
                f"{det_pool_mode!r}")
        self.det_pool_mode = det_pool_mode
        # 'dedup': the 'remaining' pool plus a focal-negative on det queries
        # whose pred box overlaps a tracked GT (IoU>dedup_overlap_thr), folded
        # into loss_cls_det (no separate loss key).
        self.dedup_bg_weight = float(dedup_bg_weight)
        self.dedup_overlap_thr = float(dedup_overlap_thr)

        # Accumulated losses across frames, optionally weighted (see
        # accumulate_frame_losses / get_losses below).
        self._losses = {}
        self._num_frames = 0
        self._total_frame_weight = 0.0

    def reset(self):
        """Reset accumulated losses for new video clip."""
        self._losses = {}
        self._num_frames = 0
        self._total_frame_weight = 0.0

    def _sigmoid_focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """Sigmoid focal loss (element-wise, no reduction)."""
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss

    @torch.no_grad()
    def _hungarian_match(self, cls_scores, bbox_preds,
                         gt_positive_maps, gt_bboxes, text_token_mask):
        """Hungarian matching for detect queries against GT.

        Args:
            cls_scores: (num_det, max_text_len) logits
            bbox_preds: (num_det, 4) in cxcywh normalized
            gt_positive_maps: (num_gt, max_text_len) float
            gt_bboxes: (num_gt, 4) in cxcywh normalized
            text_token_mask: (max_text_len,) bool

        Returns:
            matched_det_inds: (num_matched,) indices into det queries
            matched_gt_inds: (num_matched,) indices into GT
        """
        num_det = cls_scores.shape[0]
        num_gt = gt_bboxes.shape[0]

        if num_gt == 0 or num_det == 0:
            return (torch.tensor([], dtype=torch.long, device=cls_scores.device),
                    torch.tensor([], dtype=torch.long, device=cls_scores.device))

        # Classification cost: focal-loss based, fully vectorized.
        # pos_cost / neg_cost are computed once on (num_det, num_valid), then
        # the (num_det, num_gt) cost matrix is formed with two matmuls.
        out_prob = cls_scores.sigmoid()             # (num_det, max_text_len)
        valid_mask = text_token_mask                 # (max_text_len,) bool
        out_prob_v = out_prob[:, valid_mask]         # (num_det, num_valid)

        # Element-wise focal costs on valid tokens
        pos_cost_per = -self.focal_alpha * \
            ((1 - out_prob_v) ** self.focal_gamma) * \
            (-(out_prob_v + 1e-8).log())             # (num_det, num_valid)
        neg_cost_per = -(1 - self.focal_alpha) * \
            (out_prob_v ** self.focal_gamma) * \
            (-(1 - out_prob_v + 1e-8).log())         # (num_det, num_valid)

        tgt_valid = gt_positive_maps[:, valid_mask]  # (num_gt, num_valid)

        # cost[i, j] = Σ_k pos[i,k]*tgt[j,k] + neg[i,k]*(1-tgt[j,k])
        cost_cls = pos_cost_per @ tgt_valid.t() + \
                   neg_cost_per @ (1 - tgt_valid).t()  # (num_det, num_gt)

        # BBox L1 + GIoU cost (cls + L1 + GIoU).
        cost_bbox = torch.cdist(bbox_preds, gt_bboxes, p=1)
        from torchvision.ops import generalized_box_iou, box_convert
        cost_giou = -generalized_box_iou(
            box_convert(bbox_preds.clamp(0, 1), 'cxcywh', 'xyxy'),
            box_convert(gt_bboxes.clamp(0, 1), 'cxcywh', 'xyxy'))   # (N_det, N_gt)
        C = (self.match_cost_cls * cost_cls +
             self.match_cost_bbox * cost_bbox +
             self.match_cost_giou * cost_giou)

        C = C.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(C)
        return (torch.tensor(row_ind, dtype=torch.long, device=cls_scores.device),
                torch.tensor(col_ind, dtype=torch.long, device=cls_scores.device))

    def _compute_losses(self, cls_scores, bbox_preds,
                        matched_pred_inds, matched_gt_inds,
                        gt_positive_maps, gt_bboxes,
                        text_token_mask, num_total_pos,
                        img_shape=None):
        """Compute cls + bbox losses for matched predictions.

        Args:
            cls_scores: (num_queries, max_text_len) logits
            bbox_preds: (num_queries, 4) cxcywh normalized
            matched_pred_inds: (M,) matched prediction indices
            matched_gt_inds: (M,) matched GT indices
            gt_positive_maps: (num_gt, max_text_len) float targets
            gt_bboxes: (num_gt, 4) cxcywh normalized
            text_token_mask: (max_text_len,) bool, True = valid
            num_total_pos: int, for loss normalization
            img_shape: unused, kept for call-site compatibility

        Returns:
            loss_cls, loss_box
        """
        num_queries = cls_scores.shape[0]

        # Build per-query targets
        labels = cls_scores.new_zeros(num_queries, self.num_classes)
        bbox_targets = cls_scores.new_zeros(num_queries, 4)
        bbox_weights = cls_scores.new_zeros(num_queries, 4)

        if len(matched_pred_inds) > 0:
            labels[matched_pred_inds] = gt_positive_maps[matched_gt_inds]
            bbox_targets[matched_pred_inds] = gt_bboxes[matched_gt_inds]
            bbox_weights[matched_pred_inds] = 1.0

        # Classification loss: focal loss with text_token_mask
        # Only compute loss on valid text token positions
        valid_mask = text_token_mask  # (max_text_len,)
        num_valid = valid_mask.sum().item()
        if num_valid > 0:
            cls_valid = cls_scores[:, valid_mask]      # (num_queries, num_valid)
            labels_valid = labels[:, valid_mask]        # (num_queries, num_valid)
            focal = self._sigmoid_focal_loss(
                cls_valid.reshape(-1), labels_valid.reshape(-1),
                alpha=self.focal_alpha, gamma=self.focal_gamma)
            num_total_pos_clamp = max(num_total_pos, 1)
            loss_cls = focal.sum() / num_total_pos_clamp * self.loss_cls_weight
        else:
            loss_cls = cls_scores.new_tensor(0.0)

        # Box loss: L_box = lambda_box * (L1 + giou_ratio * GIoU), i.e. the
        # standard weighted combination of L1 and GIoU, returned as one term.
        zero = cls_scores.new_tensor(0.0)
        if len(matched_pred_inds) > 0:
            pos_bbox_preds = bbox_preds[matched_pred_inds]      # cxcywh [0,1]
            pos_bbox_targets = bbox_targets[matched_pred_inds]
            _n = max(num_total_pos, 1)
            _w_l1 = self.loss_box_weight
            _w_giou = self.loss_box_weight * self.giou_ratio
            loss_box = F.l1_loss(pos_bbox_preds, pos_bbox_targets,
                                 reduction='sum') / _n * _w_l1
            if _w_giou > 0:
                from torchvision.ops import generalized_box_iou, box_convert
                pred_xyxy = box_convert(
                    pos_bbox_preds.clamp(0, 1), 'cxcywh', 'xyxy')
                tgt_xyxy = box_convert(
                    pos_bbox_targets.clamp(0, 1), 'cxcywh', 'xyxy')
                giou = torch.diag(generalized_box_iou(pred_xyxy, tgt_xyxy))
                loss_box = loss_box + (1.0 - giou).sum() / _n * _w_giou
        else:
            loss_box = zero.clone()

        return loss_cls, loss_box

    # ------------------------------------------------------------------
    # Per-frame loss for track supervision
    # ------------------------------------------------------------------

    def compute_track_frame_loss(self,
                                 hidden_states_last: torch.Tensor,
                                 cls_scores_last: torch.Tensor,
                                 bbox_preds_last: torch.Tensor,
                                 gt_instance,
                                 text_token_mask: torch.Tensor,
                                 active_track_instances,
                                 num_dn: int,
                                 num_det: int,
                                 img_shape=None):
        """Single-frame track loss.

        Returns a dict with:
          loss_cls_track, loss_box_track
              — focal/L1 on the TRACK query slice, track-locked by obj_id.

        Det query losses are handled by the separate det-loss path.

        Args:
            hidden_states_last: (num_queries, C) last decoder layer hidden
                states.  Layout [DN | Det | Track].
            cls_scores_last: (num_queries, max_text_len) last-layer cls logits.
            bbox_preds_last: (num_queries, 4) last-layer bbox preds (cxcywh).
            gt_instance: InstanceData with bboxes, positive_maps, obj_ids.
            text_token_mask: (max_text_len,) bool, True = valid token.
            active_track_instances: TrackInstances of active tracks (obj_idxes
                >= 0) in the same order as the decoder's track slice.
                May be None if num_track == 0.
            num_dn: int
            num_det: int
            img_shape: (H, W) or None

        Returns:
            dict of losses + det match info for downstream use:
                {
                  'loss_cls_track': ..., 'loss_box_track': ...
                }, det_matched_pred (indices into det slice),
                   det_matched_gt   (indices into gt_instance).
        """
        device = cls_scores_last.device
        num_total_queries = cls_scores_last.shape[0]
        num_track = num_total_queries - num_dn - num_det
        assert num_track >= 0

        # Strip DN portion.
        det_cls = cls_scores_last[num_dn:num_dn + num_det]
        det_bbox = bbox_preds_last[num_dn:num_dn + num_det]
        # Track slice
        if num_track > 0:
            track_cls = cls_scores_last[num_dn + num_det:]
            track_bbox = bbox_preds_last[num_dn + num_det:]
        else:
            track_cls = cls_scores_last.new_zeros(
                (0, cls_scores_last.shape[-1]))
            track_bbox = bbox_preds_last.new_zeros((0, 4))

        # Pad text_token_mask to num_classes if needed
        if text_token_mask.shape[0] < self.num_classes:
            ttm_padded = text_token_mask.new_zeros(self.num_classes)
            ttm_padded[:text_token_mask.shape[0]] = text_token_mask
        else:
            ttm_padded = text_token_mask[:self.num_classes]

        gt_positive_maps = gt_instance.positive_maps
        gt_bboxes = gt_instance.bboxes
        gt_obj_ids = gt_instance.obj_ids

        # ── 1) Track cls/bbox (track-lock by obj_id) ──────────────────────
        trk_pred_local: list = []
        trk_gt_inds: list = []
        trk_gt_t = torch.tensor([], dtype=torch.long, device=device)
        if num_track > 0 and active_track_instances is not None:
            track_obj_idxes = active_track_instances.obj_idxes
            for j in range(num_track):
                oid = track_obj_idxes[j].item()
                if oid < 0:
                    continue
                match = (gt_obj_ids == oid).nonzero(as_tuple=True)[0]
                if len(match) > 0:
                    trk_pred_local.append(j)
                    trk_gt_inds.append(match[0].item())

            trk_pred_t = torch.tensor(
                trk_pred_local, dtype=torch.long, device=device)
            trk_gt_t = torch.tensor(
                trk_gt_inds, dtype=torch.long, device=device)
            num_pos = max(len(trk_pred_t), 1)
            tl_cls, tl_box = self._compute_losses(
                track_cls, track_bbox,
                trk_pred_t, trk_gt_t,
                gt_positive_maps, gt_bboxes,
                ttm_padded, num_pos, img_shape)
        else:
            zero = cls_scores_last.new_zeros(())
            tl_cls = zero.clone()
            tl_box = zero.clone()

        # ── 2) Det Hungarian over GT pool.
        # Track-aware Hungarian: GTs already owned by a matched track query
        # are EXCLUDED from the det pool. This trains Q_det to be the
        # "newborn / untracked detector" instead of duplicating already-
        # tracked targets. Without this, det queries are explicitly trained
        # to re-detect tracked GTs → duplicate fires on already-tracked
        # objects.
        #
        # `trk_gt_t` (computed above in section 1) holds the gt indices
        # currently owned by track queries. We mask them out.
        #
        # For trajectory-conditioned DET-query generation, use det_pool_mode
        # = 'all': Q_det is the sole bbox/cls emission owner, so it must learn
        # all visible GTs, including already-tracked objects. Track queries can
        # still carry state, but they should not be the detection generator.
        if self.det_pool_mode == 'all':
            remaining_gt_idx = torch.arange(
                len(gt_bboxes), dtype=torch.long, device=device)
        elif num_track > 0 and len(trk_gt_t) > 0:
            tracked_gt_mask = torch.zeros(
                len(gt_bboxes), dtype=torch.bool, device=device)
            tracked_gt_mask[trk_gt_t] = True
            remaining_gt_idx = torch.nonzero(
                ~tracked_gt_mask, as_tuple=True)[0]
        else:
            remaining_gt_idx = torch.arange(
                len(gt_bboxes), dtype=torch.long, device=device)

        gt_pmap_pool = gt_positive_maps[remaining_gt_idx]
        gt_box_pool = gt_bboxes[remaining_gt_idx]

        if len(gt_box_pool) > 0 and num_det > 0:
            det_matched_pred, det_matched_local = self._hungarian_match(
                det_cls, det_bbox,
                gt_pmap_pool, gt_box_pool,
                ttm_padded)
            # Map local (in remaining pool) → original gt index
            det_matched_gt = remaining_gt_idx[det_matched_local]
        else:
            det_matched_pred = torch.tensor([], dtype=torch.long, device=device)
            det_matched_local = torch.tensor([], dtype=torch.long, device=device)
            det_matched_gt = det_matched_local

        # ── 2b) Det cls/bbox loss.
        # Always compute the det focal loss when num_det > 0, including
        # when remaining_gt is empty: unmatched det queries then receive
        # negative supervision.
        _dedup_bg = cls_scores_last.new_zeros(())  # loss_dedup_bg display (detached)
        if self.loss_cls_det_weight > 0.0 and num_det > 0:
            num_pos_det = max(len(det_matched_pred), 1)
            dl_cls, dl_box = self._compute_losses(
                det_cls, det_bbox,
                det_matched_pred, det_matched_local,
                gt_pmap_pool, gt_box_pool,
                ttm_padded, num_pos_det, img_shape)
            dl_cls = dl_cls * self.loss_cls_det_weight
            dl_box = dl_box * self.loss_cls_det_weight   # same scale as cls
            # 'dedup': focal-negative on det queries whose pred box overlaps
            # a tracked GT (duplicate-of-a-track).
            if (self.det_pool_mode == 'dedup' and num_track > 0
                    and len(trk_gt_t) > 0):
                from torchvision.ops import box_iou, box_convert
                _dt = box_convert(det_bbox.clamp(0, 1), 'cxcywh', 'xyxy')
                _tk = box_convert(
                    gt_bboxes[trk_gt_t].clamp(0, 1), 'cxcywh', 'xyxy')
                _dup = box_iou(_dt, _tk).max(dim=1).values > self.dedup_overlap_thr
                if len(det_matched_pred) > 0:   # never suppress a real newborn
                    _dup = _dup.clone()
                    _dup[det_matched_pred] = False
                if bool(_dup.any()):
                    _dv = det_cls[_dup][:, ttm_padded]          # (D, num_valid)
                    _fl = self._sigmoid_focal_loss(
                        _dv.reshape(-1), torch.zeros_like(_dv).reshape(-1),
                        alpha=self.focal_alpha, gamma=self.focal_gamma)
                    # Normalize by num_pos_det, the same denominator as the
                    # main det cls focal.
                    _dedup_term = self.dedup_bg_weight * (
                        _fl.sum() / max(num_pos_det, 1))
                    dl_cls = dl_cls + _dedup_term
                    # Detached copy for logging ONLY (requires_grad=False →
                    # _apply_loss_weights skips it → NOT double-counted in
                    # total; shows in the primary_losses display dict).
                    _dedup_bg = _dedup_term.detach()
        else:
            dl_cls = cls_scores_last.new_zeros(())
            dl_box = cls_scores_last.new_zeros(())

        losses = {
            'loss_cls_track': tl_cls,
            'loss_box_track': tl_box,         # L1 + GIoU
            'loss_cls_det': dl_cls,
            'loss_dedup_bg': _dedup_bg,         # detached display of dedup neg term
            'loss_box_det': dl_box,           # L1 + GIoU
        }
        return losses, det_matched_pred, det_matched_gt

    @staticmethod
    def _cxcywh_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Pairwise IoU between two sets of cxcywh boxes → (len(a), len(b))."""
        if a.numel() == 0 or b.numel() == 0:
            return a.new_zeros((a.shape[0], b.shape[0]))

        def _to_xyxy(x):
            cx, cy, w, h = x.unbind(-1)
            return torch.stack(
                [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], -1)

        A = _to_xyxy(a); B = _to_xyxy(b)
        area_a = (A[:, 2] - A[:, 0]).clamp(min=0) * (A[:, 3] - A[:, 1]).clamp(min=0)
        area_b = (B[:, 2] - B[:, 0]).clamp(min=0) * (B[:, 3] - B[:, 1]).clamp(min=0)
        lt = torch.max(A[:, None, :2], B[None, :, :2])
        rb = torch.min(A[:, None, 2:], B[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        union = area_a[:, None] + area_b[None, :] - inter + 1e-6
        return inter / union

    def accumulate_frame_losses(self, frame_losses, frame_weight: float = 1.0):
        """Add frame losses to the accumulated loss dict with a per-frame scale.

        ``frame_weight == 1.0`` (the value both call sites pass) makes every
        frame contribute uniformly.
        """
        for k, v in frame_losses.items():
            weighted = v * frame_weight
            if k in self._losses:
                self._losses[k] = self._losses[k] + weighted
            else:
                self._losses[k] = weighted
        self._num_frames += 1
        self._total_frame_weight += frame_weight

    def get_losses(self):
        """Return accumulated losses normalized by the total frame weight.

        With uniform ``frame_weight=1.0`` this is equivalent to dividing by
        ``num_frames``; with non-uniform weights the normalization keeps the
        reported loss magnitudes comparable across schedules.
        """
        if self._total_frame_weight == 0:
            return self._losses
        return {k: v / self._total_frame_weight for k, v in self._losses.items()}
