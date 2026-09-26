"""INGRAIN per-frame runner for TAO benchmark evaluation.

Runs the tracking pipeline:
  the detector (chunked multi-class prediction) + trajectory state propagation +
  RuntimeTrackerBase (score lifecycle only) +
  query propagation.

Design notes
------------
GD's cross-modal encoder is text-dependent, so a single forward cannot
cover all 1203 TAO classes. We therefore chunk the class list into
``chunked_size`` groups and run a full forward **per chunk**.

Within one frame:
  * Visual backbone features are computed once (text-independent) and
    reused for every chunk.
  * Encoder / decoder / bbox_head are re-run per chunk because the
    cross-modal fusion depends on the prompt.
  * Per-query merge: for each of the 900 det queries (and each active
    track query), we pick the chunk where its max-over-tokens score is
    highest, and take that chunk's (score, bbox, class_label).
  * State carried across observations (trajectory-memory push, tracker,
    query propagation) comes from chunk 0.

The resulting (boxes, scores, cats, ids) per frame plug directly into
``ResultFormatter``.
"""

import os
import re
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

# Path setup so we can import inference.py and ingrain.*
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ingrain.structures.track_instances import TrackInstances
from ingrain.models.ops import inverse_sigmoid
from inference import RuntimeTrackerBase, preprocess


# ---------------------------------------------------------------------------
# Text chunk construction
# ---------------------------------------------------------------------------

def _split_into_chunks(class_names: List[str],
                        cat_ids: List[int],
                        chunked_size: int
                        ) -> List[Tuple[List[str], List[int]]]:
    """Split the full class list into chunks of up to ``chunked_size``.

    Returns a list of (chunk_class_names, chunk_cat_ids) pairs.
    The chunk index also acts as the "local chunk offset" used to map
    local label → global cat_id.
    """
    chunks = []
    for start in range(0, len(class_names), chunked_size):
        end = min(start + chunked_size, len(class_names))
        chunks.append((class_names[start:end], cat_ids[start:end]))
    return chunks


def _build_text_dict(model, prompt_classes: List[str], device):
    """Build a text_dict for one chunk's prompt.

    Returns (text_dict, tokens_positive, caption) so callers can later
    build per-class positive maps if needed.
    """
    prompt = ' . '.join(prompt_classes)
    tokenized, caption, tokens_positive, _ = model.get_tokens_and_prompts(
        prompt, True)
    text_dict = model.language_model([caption])
    if model.text_feat_map is not None:
        text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])
    for k, v in text_dict.items():
        if isinstance(v, torch.Tensor):
            text_dict[k] = v.to(device)
    return text_dict, tokens_positive, caption


def _pad_positive_maps_to_max_text_len(pmap: Tensor,
                                        max_text_len: int) -> Tensor:
    """Pad / truncate positive_maps along the token axis to ``max_text_len``.

    ``pmap`` has shape (num_classes, token_len) where token_len is the raw
    tokenized caption length (<= max_text_len).  cls_scores during forward
    are padded to ``max_text_len`` (contrastive head projects into that
    space).  We pad pmap with zeros so the per-class mean aggregation over
    tokens aligns with the cls tensor's columns.
    """
    num_classes, token_len = pmap.shape
    if token_len == max_text_len:
        return pmap
    if token_len > max_text_len:
        return pmap[:, :max_text_len]
    padded = pmap.new_zeros(num_classes, max_text_len)
    padded[:, :token_len] = pmap
    return padded


# ---------------------------------------------------------------------------
# Utility: box conversions
# ---------------------------------------------------------------------------

def _cxcywh_norm_to_xyxy_pixel(boxes_cxcywh: Tensor,
                                img_w: int, img_h: int) -> Tensor:
    """(N, 4) normalized cxcywh → (N, 4) pixel xyxy."""
    if len(boxes_cxcywh) == 0:
        return boxes_cxcywh.new_zeros(0, 4)
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class IngrainRunner:
    """Frame-level INGRAIN evaluator with chunked multi-class prediction."""

    def __init__(self,
                 model,
                 class_names: List[str],
                 cat_ids: List[int],
                 *,
                 chunked_size: int = 40,
                 score_thresh: float = 0.30,
                 filter_score_thresh: float = 0.15,
                 miss_tolerance: int = 5,
                 max_dets: int = 300,
                 topk_cls: int = 1,
                 source_filter: str = 'all',
                 use_iou_gate: bool = False,
                 iou_thresh: float = 0.5,
                 e2e_assoc: bool = False,
                 use_inter_track_nms: bool = False,
                 ):
        """Args:
            model: INGRAINTracker on the target device (train() or eval() OK).
            class_names: all class names to detect (order matches cat_ids).
            cat_ids: global TAO category_id per class (1-indexed).
            chunked_size: classes per text chunk.
            score_thresh / filter_score_thresh / miss_tolerance:
                RuntimeTrackerBase thresholds.
            max_dets: hard cap on tracks reported per frame.
            topk_cls: how many class predictions to emit per alive slot.
        """
        self.model = model
        self.class_names = class_names
        self.cat_ids = cat_ids
        self.chunked_size = chunked_size
        self.max_dets = max_dets
        self.score_thresh = score_thresh
        self.topk_cls = max(1, int(topk_cls))
        self.device = next(model.parameters()).device
        # Diagnostic: filter outputs by query source ('all' | 'det' | 'track').
        # Used for source-decomposition runs (eval which slice produces FPs).
        if source_filter not in ('all', 'det', 'track'):
            raise ValueError(f"source_filter must be one of all|det|track, "
                             f"got {source_filter!r}")
        self.source_filter = source_filter
        self.use_iou_gate = bool(use_iou_gate)
        self.iou_thresh = float(iou_thresh)
        # Association is slot continuity: obj_idxes are carried frame to
        # frame, and unmatched high-score det boxes become newborns via
        # tracker.update.
        self.e2e_assoc = bool(e2e_assoc)
        self.use_inter_track_nms = bool(use_inter_track_nms)

        # Build per-chunk caches: text_dict + per-class positive maps
        # (class_pmaps aggregates token logits → per-class score at eval time,
        # mirroring GD's get_positive_map / atss_vlfusion_head convention).
        self._chunks = _split_into_chunks(
            class_names, cat_ids, chunked_size)
        self._chunk_text_dicts: List[dict] = []
        self._chunk_cat_ids: List[List[int]] = []        # global cat_ids per chunk
        self._chunk_class_pmaps: List[Tensor] = []       # (C_k, max_text_len) float
        self._chunk_class_token_counts: List[Tensor] = []  # (C_k,) float, for mean
        self._chunk_valid_tokens: List[Tensor] = []
        self._build_chunk_caches()

        # Runtime tracker (track lifecycle manager from inference.py).
        self.tracker = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
            use_iou_gate=self.use_iou_gate,
            iou_thresh=self.iou_thresh,
            use_inter_track_nms=self.use_inter_track_nms)
        if self.use_iou_gate or self.use_inter_track_nms:
            print(f"  [IoU runtime] gate={self.use_iou_gate}, "
                  f"inter_track_nms={self.use_inter_track_nms}, "
                  f"thresh={self.iou_thresh}")


        # Per-video state
        self._track_instances: Optional[TrackInstances] = None
        self._frame_idx: int = 0
        # Chunk-0 encoder memory + spatial shapes, stashed per frame for the
        # enc-ROI observation.
        self._enc_mem0 = None
        self._enc_ss0 = None
        # e2e-assoc: per-oid record of each track's last CONFIDENT propagated
        # content. The content computed from an observation goes straight into
        # that observation's query_tgt (eval mirror of detector.py's
        # _propagate_tracks); this cache exists so a track that is currently
        # occluded re-propagates its pre-occlusion content instead of a
        # low-confidence decode. No matcher; slot continuity IS the association.
        self._q_next_memory: dict = {}

    # ------------------------------------------------------------------
    # Cache: tokenize all chunks once
    # ------------------------------------------------------------------

    def _build_chunk_caches(self) -> None:
        """Tokenize + embed each chunk's prompt, precompute per-class pmaps."""
        max_text_len = self.model.bbox_head.cls_branches[0].max_text_len \
            if hasattr(self.model.bbox_head.cls_branches[0], 'max_text_len') \
            else 256

        for chunk_classes, chunk_cats in self._chunks:
            text_dict, _, _ = _build_text_dict(
                self.model, chunk_classes, self.device)
            n_tokens = int(text_dict['embedded'].shape[1])
            if n_tokens > max_text_len:
                raise ValueError(
                    f"chunk prompt has {n_tokens} tokens, exceeding "
                    f"max_text_len={max_text_len}: {chunk_classes[:8]}...")

            # Build token-level positive_maps for this chunk's classes.
            # pmap[c, t] = 1 iff token t is part of class c's name span.
            tokenized, _, tokens_positive, _ = \
                self.model.get_tokens_and_prompts(
                    ' . '.join(chunk_classes), True)
            per_class_tp = [tokens_positive[i]
                             for i in range(len(chunk_classes))]
            _, pmap = self.model.get_positive_map(tokenized, per_class_tp)
            pmap = _pad_positive_maps_to_max_text_len(
                pmap.to(self.device).float(), max_text_len)      # (C_k, mtl)
            token_counts = pmap.sum(dim=-1).clamp(min=1.0)       # (C_k,)

            # Valid token mask (BERT's attention mask; [CLS]/[SEP]/./? inclusive)
            ttm_row = text_dict['text_token_mask'][0]
            if ttm_row.shape[0] < max_text_len:
                ttm_padded = ttm_row.new_zeros(max_text_len)
                ttm_padded[:ttm_row.shape[0]] = ttm_row
            else:
                ttm_padded = ttm_row[:max_text_len]

            self._chunk_text_dicts.append(text_dict)
            self._chunk_cat_ids.append(chunk_cats)
            self._chunk_class_pmaps.append(pmap)
            self._chunk_class_token_counts.append(token_counts)
            self._chunk_valid_tokens.append(ttm_padded)

        # Flat lookups: (total_C,) tensors for chunk-idx and global cat_id at
        # each flattened class position.  Used by the multi-class emission
        # path to recover which chunk owns a given top-K class.
        cat_flat: List[int] = []
        chunk_flat: List[int] = []
        for k, cats in enumerate(self._chunk_cat_ids):
            cat_flat.extend(cats)
            chunk_flat.extend([k] * len(cats))
        self._flat_cat_ids = torch.tensor(
            cat_flat, dtype=torch.long, device=self.device)
        self._flat_chunk_ids = torch.tensor(
            chunk_flat, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_video_start(self) -> None:
        """Reset per-video state: memory bank + tracker + frame counter."""
        self.model.reset_memory()
        self.tracker.reset()
        self._track_instances = None
        self._frame_idx = 0
        self._q_next_memory.clear()
        # INGRAIN mem-fuse eval mirror: reset the trajectory memory banks per
        # video (mirrors trajectory_memory.reset() per-clip in train). Both the
        # seed memory_bank (enc-ROI obs) and the interleave per-depth banks.
        _tmem = getattr(self.model, 'trajectory_memory', None)
        if _tmem is not None:
            # Unconditional: the seed bank exists whenever the module does.
            # reset_memory() a few lines above already clears it; this is the
            # explicit per-video reset of every bank the runner owns.
            _tmem.seed_mem_bank.reset()
            for _ilb in getattr(_tmem, 'depth_mem_bank', []):
                _ilb.reset()
            if hasattr(self.model, 'decoder'):
                self.model.decoder._mf_il_fired = False   # re-arm fire print

    def _best_score_in_chunk(self,
                             per_chunk_class_scores: List[Tensor],
                             query_idx: int,
                             chunk_id: int) -> float:
        scores = per_chunk_class_scores[int(chunk_id)]
        if scores.numel() == 0:
            return 0.0
        return float(scores[int(query_idx)].max().item())

    @property
    def next_id(self) -> int:
        """Exposed for ResultFormatter's global_offset bookkeeping."""
        return self.tracker.max_obj_id

    def _age_tracks_on_missing_frame(self) -> None:
        """Bump disappear_time for every active track; kill at miss_tolerance.

        Called ONLY when ``cv2.imread`` can't decode the image. Mirrors the
        death branch of ``RuntimeTrackerBase.update`` but assumes every track
        "scored below filter_score_thresh" because no inference was run. The
        normal death path is that branch itself; both release the dead tracks'
        trajectory memory.
        """
        ti = self._track_instances
        if ti is None:
            return
        mt = self.tracker.miss_tolerance
        alive = (ti.obj_idxes >= 0)
        if not alive.any():
            return
        # disappear_time++ on every active slot, kill past threshold
        ti.disappear_time[alive] = ti.disappear_time[alive] + 1
        exceeded = alive & (ti.disappear_time >= mt)
        if exceeded.any():
            _dead_ids = ti.obj_idxes[exceeded].tolist()
            ti.obj_idxes[exceeded] = -1
            # Release the dead tracks' trajectory memory with the tracks
            # themselves. Without this the banks only clear at the end of the
            # video, so a dead id's history stays resident.
            _tmem_ev = getattr(self.model, 'trajectory_memory', None)
            if _tmem_ev is not None and hasattr(_tmem_ev, 'evict_tracks'):
                _tmem_ev.evict_tracks(_dead_ids)

    @torch.no_grad()
    def process_frame(self, img_path: str
                      ) -> Tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        self._cur_img_path = str(img_path)
        """Run the tracking pipeline on one frame; return tracked detections.

        Returns:
            boxes_xyxy: (K, 4) float ndarray in ORIGINAL pixel space.
            scores:     (K,)   float ndarray.
            cats:       (K,)   int64 ndarray of global TAO category_id.
            ids:        (K,)   int64 ndarray of track_id (local to this video).
            sources:    (K,)   str ndarray, emitting query slot per detection:
                        'det' or 'track'.
        """
        # ── Load + preprocess image ────────────────────────────────────
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            # Missing frame: advance the clock AND age active tracks the
            # same way RuntimeTrackerBase.update() would on a frame where
            # no track scores above filter_score_thresh (everyone gets
            # miss++; kill at miss_tolerance).  Without this, stale tracks
            # survive indefinitely through unreadable frames.
            self._age_tracks_on_missing_frame()
            self._frame_idx += 1
            # Same arity as the normal return below: an unreadable frame
            # contributes no detections but must not change the contract.
            return (np.zeros((0, 4)), np.zeros(0),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype='<U5'))
        ori_h, ori_w = img_bgr.shape[:2]
        img_tensor, img_shape = preprocess(img_bgr)
        img_tensor = img_tensor.to(self.device)
        ph, pw = img_tensor.shape[-2:]

        # ── Build DetDataSample once ───────────────────────────────────
        ds = DetDataSample()
        ds.set_metainfo(dict(
            img_shape=img_shape,
            batch_input_shape=(ph, pw),
        ))
        ds.gt_instances = InstanceData(
            bboxes=torch.zeros(0, 4, device=self.device),
            labels=torch.zeros(0, dtype=torch.long, device=self.device),
        )

        # ── Visual features — shared across chunks ────────────────────
        visual_features = self.model.extract_feat(img_tensor)

        # ── Decode + finalize. One decoder pass per text chunk. A
        # track's positional reference is its own predicted box from the
        # preceding observation, never re-anchored onto a fresh detection.
        # _decode_frame returns the frame's output tuple directly
        # (out_boxes_xyxy, scores, cats, ids, sources).
        return self._decode_frame(visual_features, ds, ori_w, ori_h)

    @staticmethod
    def _seed_mem_refine(tmem, ti, num_det, j, content):
        """Trajectory memory at depth d = 0 (the seed), for one track row.

        Refines ``content`` (1 row, (C,)) with this track's stored history, then
        stores the PRE-injection content as next-frame history, so the bank only
        ever holds states from PREVIOUS observations (B_{i,t} = {q_{i,t-k}}).
        Key = oid (single video, no per-sample offset).

        Content-agnostic on purpose: the caller passes the enc-ROI observation
        when INGRAIN_TRACK_CONTENT_ENCROI is on and the decoded track state when
        it is off, so d = 0 is present in both configurations.
        """
        _oidt = ti.obj_idxes[num_det + j:num_det + j + 1]
        _obs = content.unsqueeze(0)                     # (1, C) raw
        out = tmem.propagate_memory_state(q_tr=_obs, active_obj_idxes=_oidt)[0]
        tmem.seed_mem_bank.update(_oidt, _obs)
        return out

    def _decode_frame(self, visual_features, ds, ori_w, ori_h):
        """Decode one frame from self._track_instances. Called exactly once per
        frame: it only READS self._track_instances (never destructively mutates
        it), and it advances the propagated-content and trajectory-memory state
        once, at the end."""
        # ── Per-chunk forward ─────────────────────────────────────────
        num_chunks = len(self._chunks)
        chunk_cls: List[Tensor] = []
        chunk_bbox: List[Tensor] = []
        chunk_hidden: List[Tensor] = []
        # Chunked inference decodes this ONE observation once per text chunk, so
        # the per-depth memory must not be written from inside the loop: 31
        # writes would fill K=16 with 31 views of the CURRENT frame and leave no
        # cross-frame history, contradicting B_{i,t} = {q_{i,t-k}}. Buffer them
        # and commit once below, so every chunk also reads the same
        # pre-observation history (training decodes once, so it stays immediate).
        _tmem_il = getattr(self.model, 'trajectory_memory', None)
        if _tmem_il is not None:
            _tmem_il.interleave_write_mode = 'defer'
            _tmem_il._interleave_pending = {}

        for k in range(num_chunks):
            enc_in, dec_in = self.model.pre_transformer(
                visual_features, [ds])
            enc_out = self.model.forward_encoder(
                **enc_in, text_dict=self._chunk_text_dicts[k])
            if k == 0:
                # Stash chunk-0 encoder memory + spatial_shapes for RoI identity
                # at match time (text-fused encoder feats, same source as train).
                self._enc_mem0 = enc_out.get('memory')
                self._enc_ss0 = enc_out.get('spatial_shapes')

            tmp_dec, head_in = self.model.pre_decoder(
                **enc_out, batch_data_samples=[ds],
                track_instances=self._track_instances)
            dec_in.update(tmp_dec)
            # INGRAIN_MEMFUSE_INTERLEAVE eval mirror: stash this frame's carried-
            # track obj_idxes so the SHARED _run_track_copy interleave (layers
            # 0/1/2) fires in eval identically to train. Single video => no
            # per-sample offset (oid is the key). Always set (ids or None).
            if os.environ.get('INGRAIN_MEMFUSE_INTERLEAVE') == '1':
                _act_il = head_in.get('_active_track_instances')
                _oids_il = (getattr(_act_il, 'obj_idxes', None)
                            if _act_il is not None else None)
                if _oids_il is not None and _oids_il.numel() > 0:
                    self.model.decoder._mf_interleave_oids = _oids_il.clone()
                    # object.__setattr__: a plain assignment would make
                    # nn.Module register trajectory_memory as a SUBMODULE of the
                    # decoder, duplicating ~60 tensors into state_dict() and
                    # parameters(). Mirrors what the detector does on the
                    # training path.
                    object.__setattr__(self.model.decoder,
                                       '_mf_interleave_sc',
                                       self.model.trajectory_memory)
                else:
                    self.model.decoder._mf_interleave_oids = None
                    self.model.decoder._mf_interleave_sc = None
            else:
                self.model.decoder._mf_interleave_oids = None
                self.model.decoder._mf_interleave_sc = None
            dec_out = self.model.forward_decoder(**dec_in)
            head_in.update(dec_out)

            num_dn = 0
            dn_meta = head_in.get('dn_meta')
            if dn_meta is not None:
                num_dn = dn_meta.get('num_denoising_queries', 0)
            num_det = self.model.num_queries

            cls_all, bbox_all = self.model._bbox_head_forward(
                head_in['hidden_states'], head_in['references'],
                head_in['memory_text'], head_in['text_token_mask'],
                num_dn=num_dn, num_det=num_det)

            # Last layer, strip DN. Shape (total_q, mtl / 4 / C).
            chunk_cls.append(cls_all[-1, 0, num_dn:])
            chunk_bbox.append(bbox_all[-1, 0, num_dn:])
            chunk_hidden.append(head_in['hidden_states'][-1, 0, num_dn:])

        # One state per observation, as in training.
        if _tmem_il is not None:
            _tmem_il.commit_interleave_writes()
            _tmem_il.interleave_write_mode = 'immediate'

        num_det = self.model.num_queries
        total_q = chunk_hidden[0].shape[0]
        num_track = total_q - num_det

        # ── Per-query per-class scores, flat across all chunks ────────
        # For each chunk, compute (Q, C_k) per-class scores via GD's
        # convert_grounding_to_cls_scores convention (sigmoid → token-mean).
        # Then concat along the class axis to get a single (Q, total_C) tensor
        # so we can do global top-1 (for the tracker) and global top-K
        # (for multi-class emission — matches GD official chunked predict).
        per_chunk_class_scores = []
        for k in range(num_chunks):
            probs_k = chunk_cls[k].sigmoid()              # (Q, mtl)
            pmap_k = self._chunk_class_pmaps[k]           # (C_k, mtl) binary
            counts_k = self._chunk_class_token_counts[k]  # (C_k,)
            class_scores_k = (probs_k @ pmap_k.t()) / counts_k.unsqueeze(0)
            per_chunk_class_scores.append(class_scores_k)
        flat_scores = torch.cat(per_chunk_class_scores, dim=-1)  # (Q, total_C)

        # Top-1: drives the tracker, memory bank, and query propagation.
        top1_scores, top1_cls = flat_scores.max(dim=-1)                # (Q,)
        top1_chunk = self._flat_chunk_ids[top1_cls]                    # (Q,)
        top1_cat = self._flat_cat_ids[top1_cls]                        # (Q,)

        prev_active = None
        if self._track_instances is not None and num_track > 0:
            prev_active = self._track_instances[
                self._track_instances.obj_idxes >= 0]
            assert len(prev_active) == num_track, (
                f"Track-slice mismatch: {len(prev_active)} vs {num_track}")

        # Each query's box comes from the chunk where it scored highest; the
        # state carried across frames comes from chunk 0.
        q_arange = torch.arange(total_q, device=self.device)
        bbox_stack = torch.stack(chunk_bbox, dim=0)                    # (K, Q, 4)

        merged_bbox = bbox_stack[top1_chunk, q_arange]
        merged_cat = top1_cat.clone()                                  # (Q,)

        best_score = top1_scores.clone()                               # (Q,)

        ref_hidden = chunk_hidden[0]        # (total_q, C) canonical
        ref_cls = chunk_cls[0]

        # Top-K: used ONLY at output emission time (tracker is unchanged).
        # When topk_cls == 1, this emits one class per alive slot.
        if self.topk_cls > 1:
            K = min(self.topk_cls, flat_scores.shape[-1])
            topk_scores_all, topk_cls_idx = flat_scores.topk(K, dim=-1)  # (Q, K)
            topk_cats_all = self._flat_cat_ids[topk_cls_idx]             # (Q, K)
        else:
            topk_scores_all = None
            topk_cats_all = None

        # ── Build TrackInstances using MERGED (bbox, score) ───────────
        ti = TrackInstances()
        ti.output_embedding = ref_hidden
        ti.pred_logits = ref_cls        # field-shape compatibility only
        ti.pred_boxes = merged_bbox
        ti.scores = best_score
        ti.cat_ids = merged_cat
        # Carry-forward ref, mirroring detector._propagate_tracks: base carried
        # ref = prev predicted box in logit space. Det/newborn rows keep the
        # bare carried ref.
        _ref_logit = inverse_sigmoid(merged_bbox.clone().clamp(1e-4, 1 - 1e-4))
        ti.ref_pts = _ref_logit
        ti.query_tgt = ref_hidden.clone()
        ti.obj_idxes = torch.full((total_q,), -1, dtype=torch.long,
                                    device=self.device)
        ti.disappear_time = torch.zeros(total_q, dtype=torch.long,
                                           device=self.device)
        ti.matched_gt_idxes = torch.full((total_q,), -1, dtype=torch.long,
                                            device=self.device)
        if prev_active is not None and num_track > 0:
            ti.obj_idxes[num_det:] = prev_active.obj_idxes
            if prev_active.has('disappear_time'):
                ti.disappear_time[num_det:] = prev_active.disappear_time
            # e2e-assoc: each track slot's query content is overwritten with the
            # propagated content that training shaped (mirror of
            # detector._propagate_tracks) — but LATER in this same call, once
            # this observation's content has been computed, so the content that
            # leaves frame t in self._track_instances is the one computed FROM
            # frame t. Until then the slot carries its decoded state. Pure slot
            # continuity → obj_id rides the slot, no matcher. The content is in
            # query-content (pre-norm hidden) space.

        # Propagated track content: computed from THIS observation and written
        # straight into the track slot's query_tgt, so the query that decodes
        # observation t+1 is the content computed from observation t — the same
        # one-step propagation detector._propagate_tracks performs in training.
        # Writing it straight into query_tgt (rather than routing it through
        # _q_next_memory, which is consumed at the START of the next call) is
        # what keeps the two paths one step apart rather than two. The cache is
        # kept ONLY to re-propagate the last confident content of a track that
        # is currently occluded.
        #
        # With INGRAIN_TRACK_CONTENT_ENCROI off there is no enc-ROI observation
        # and the track rows propagate their decoded state (ti.query_tgt is the
        # decoded hidden, set above); the loop still runs so that the seed
        # trajectory memory refines and stores THAT content. The seed sits
        # outside the enc-ROI branch, so the memory depth set stays
        # d={0,1,2,3} either way.
        _encroi_on = (os.environ.get('INGRAIN_TRACK_CONTENT_ENCROI') == '1'
                      and getattr(self.model, 'enc_roi_proj', None) is not None
                      and self._enc_mem0 is not None)
        _tmem = getattr(self.model, 'trajectory_memory', None)
        _seed_on = (os.environ.get('INGRAIN_ENCROI_MEM_FUSE') == '1'
                    and _tmem is not None
                    and getattr(_tmem, 'seed_mem_attn', None) is not None)
        if (self.e2e_assoc and num_track > 0 and (_encroi_on or _seed_on)):
            with torch.no_grad():
                oids_e2e = [int(ti.obj_idxes[num_det + j].item())
                            for j in range(num_track)]
                # Confidence gate: tracks scoring above the threshold propagate
                # the content computed from THIS observation; occluded/low-score
                # tracks re-propagate their last confident content so
                # reappearance resumes from the pre-occlusion state.
                for j, oid in enumerate(oids_e2e):
                    if oid < 0:
                        continue
                    if float(ti.scores[num_det + j].item()) >= self.score_thresh:
                        if _encroi_on:
                            # Inference mirror of detector._propagate_tracks:
                            # encoder-ROI content from THIS frame's box+memory.
                            # Same _roi_pool_memory + enc_roi_proj as train.
                            _erb = ti.pred_boxes[num_det + j:num_det + j + 1]
                            _er = self.model._roi_pool_memory(
                                self._enc_mem0, self._enc_ss0, _erb)
                            _content = self.model.enc_roi_proj(
                                _er.float())[0].detach()   # raw enc-ROI obs
                            _hf = getattr(self.model, 'hidden_fuse', None)
                            if _hf is not None:
                                # eval mirror of the train fusion: blend THIS
                                # frame's decoder hidden into the enc-ROI
                                # content = E⊕hd, BEFORE the mem-fuse
                                # refinement below. output_embedding is this
                                # frame's decoder output for the track row,
                                # which is what the MLP was trained on
                                # (detector.py sets query_tgt = hs_last right
                                # before fusing).
                                _Hdec = ti.output_embedding[
                                    num_det + j].to(_content.dtype)
                                _content = _hf(
                                    _content.unsqueeze(0),
                                    _Hdec.unsqueeze(0))[0].detach()
                            if not getattr(self, '_dbg_encroi_fired', False):
                                self._dbg_encroi_fired = True
                                print("[OGP] observation-guided propagation "
                                      "active: track content taken from the "
                                      "enc-ROI observation", flush=True)
                        else:
                            # Decoded track state — what this row already
                            # carries, mirroring detector._propagate_tracks,
                            # which refines the decoded hidden it just wrote
                            # into query_tgt.
                            _content = ti.query_tgt[num_det + j].detach().float()
                        # INGRAIN_ENCROI_MEM_FUSE eval mirror: refine with this
                        # track's history (memory attention) + store this
                        # frame's raw obs. Key = oid (single video, no offset).
                        # Reads the bank BEFORE the update (history excludes
                        # this frame).
                        if _seed_on:
                            _content = self._seed_mem_refine(
                                _tmem, ti, num_det, j, _content)
                        self._q_next_memory[oid] = _content.detach()
                    else:
                        _content = self._q_next_memory.get(oid)
                        if _content is None:
                            continue   # newborn, never yet confident
                    ti.query_tgt[num_det + j] = _content.to(ti.query_tgt.dtype)

        # Tracker lifecycle: score filtering and miss tolerance.
        self.tracker.update(ti, num_det=num_det)
        # Release the trajectory memory of tracks that just died. This is the
        # normal death path (miss_tolerance exceeded); the missing-frame ageing
        # branch has its own call and only fires when a frame cannot be read.
        _died = getattr(self.tracker, '_frame_died_ids', None)
        if _died:
            _tmem_ev = getattr(self.model, 'trajectory_memory', None)
            if _tmem_ev is not None and hasattr(_tmem_ev, 'evict_tracks'):
                _tmem_ev.evict_tracks(_died)

        # ── trajectory-memory push (always on) ─────────────────────────
        # The trajectory-state modules consume the trajectory-memory bank on
        # the next frame.
        tmem_push = getattr(self.model, 'trajectory_memory', None)
        # INGRAIN_ENCROI_MEM_FUSE eval mirror: when the mem-fuse seed is on, the
        # bank is written with enc-ROI obs in the content path above; skip this
        # decoder-hidden write so one bank doesn't mix two feature spaces
        # (mirrors the training path in the detector).
        if (tmem_push is not None
                and os.environ.get('INGRAIN_ENCROI_MEM_FUSE') != '1'):
            active_mask_ti = ti.obj_idxes >= 0
            if active_mask_ti.any():
                from ingrain.models.detector import _SimpleView
                tmem_view = _SimpleView(
                    obj_idxes=ti.obj_idxes[active_mask_ti],
                    output_embedding=ti.output_embedding[active_mask_ti])
                tmem_push.update_memory(tmem_view, frame_idx=self._frame_idx)

        # ── Snapshot & collect outputs ─────────────────────────────────
        alive = ti.obj_idxes >= 0
        alive_idx = alive.nonzero(as_tuple=True)[0]

        # Diagnostic: filter alive_idx by query source. det queries are
        # positions [0, num_det); track queries are [num_det, total_q).
        if self.source_filter != 'all' and len(alive_idx) > 0:
            is_track = alive_idx >= num_det
            if self.source_filter == 'det':
                keep_mask = ~is_track
            else:  # 'track'
                keep_mask = is_track
            alive_idx = alive_idx[keep_mask]

        # Compute per-slot source tag BEFORE topk expansion. det/track is a
        # property of the query slot, so K copies of one slot share the tag.
        alive_source = np.where(
            alive_idx.cpu().numpy() < num_det, 'det', 'track')

        if self.topk_cls > 1 and topk_scores_all is not None:
            # Multi-class emission: each alive slot contributes K detections
            # sharing (bbox, track_id) but with K different (cat, score) pairs.
            K = topk_scores_all.shape[-1]
            alive_boxes_cxcywh = ti.pred_boxes[alive_idx]            # (A, 4)
            alive_ids = ti.obj_idxes[alive_idx]                      # (A,)
            alive_topk_scores = topk_scores_all[alive_idx]           # (A, K)
            alive_topk_cats = topk_cats_all[alive_idx]               # (A, K)

            A = alive_boxes_cxcywh.shape[0]
            out_boxes_cxcywh = (
                alive_boxes_cxcywh.unsqueeze(1).expand(A, K, 4)
                .reshape(A * K, 4).cpu())
            out_scores = alive_topk_scores.reshape(A * K).cpu()
            out_ids = (
                alive_ids.unsqueeze(1).expand(A, K)
                .reshape(A * K).cpu().numpy().astype(np.int64))
            out_cats = (
                alive_topk_cats.reshape(A * K).cpu().numpy().astype(np.int64))
            out_sources = np.repeat(alive_source, K)
        else:
            # Top-1 emission.
            out_scores = ti.scores[alive_idx].cpu()
            out_boxes_cxcywh = ti.pred_boxes[alive_idx].cpu()
            out_ids = ti.obj_idxes[alive_idx].cpu().numpy().astype(np.int64)
            out_cats = merged_cat[alive_idx].cpu().numpy().astype(np.int64)
            out_sources = alive_source

        # Emission gate: one threshold for births and continuations alike.
        keep = out_scores >= self.score_thresh
        out_scores = out_scores[keep]
        out_boxes_cxcywh = out_boxes_cxcywh[keep]
        out_ids = out_ids[keep.numpy()]
        out_cats = out_cats[keep.numpy()]
        out_sources = out_sources[keep.numpy()]

        # Convert boxes: normalized cxcywh → original pixel xyxy
        out_boxes_xyxy = _cxcywh_norm_to_xyxy_pixel(
            out_boxes_cxcywh, ori_w, ori_h).numpy()

        # Cap at max_dets (keep top by score).
        if len(out_scores) > self.max_dets:
            top = np.argsort(-out_scores.numpy())[:self.max_dets]
            out_boxes_xyxy = out_boxes_xyxy[top]
            out_scores = out_scores[top]
            out_ids = out_ids[top]
            out_cats = out_cats[top]
            out_sources = out_sources[top]

        # ── Query propagation (canonical chunk hidden) ────────────────
        # ``cat_ids`` is tracker-only metadata. Query propagation concatenates the current
        # frame tracks with freshly generated empty tracks, so remove metadata
        # fields that the empty template does not carry.
        if ti.has('cat_ids'):
            del ti._fields['cat_ids']
        init = self.model._generate_empty_tracks(device=self.device)
        init.disappear_time = torch.zeros(
            len(init), dtype=torch.long, device=self.device)
        # Identity propagation, mirroring training: only matched tracks carry
        # over, with the content and reference already set above.
        active = ti[ti.obj_idxes >= 0]
        self._track_instances = TrackInstances.cat([init, active])

        self._frame_idx += 1

        return (out_boxes_xyxy,
                out_scores.numpy().astype(np.float32),
                out_cats,
                out_ids,
                out_sources)
