"""INGRAINTracker: Multi-Object Tracker built on Grounding DINO.

Extends GD with:
1. A multi-frame training loop that propagates track queries across observations
2. SDP: a staged dual-path decoder — in the decoupling stage (the first n = 3
   layers) a separate trainable track path decodes the track queries; the
   remaining layers form the fusion stage, which decodes both query sets jointly
3. MDM: a multi-depth trajectory memory (seed + per-depth memory attention)
4. OGP: observation-guided propagation — an observation pooled from the encoder
   memory inside the predicted box, fused with the decoded track state

Key design: subclasses GroundingDINO, so every GD-native key loads unchanged;
only the newly introduced modules are randomly initialised. Training covers the
track path and the fusion stage.
"""
import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple, Union
from torch import Tensor

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'mmgroundingdino'))
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.models.detectors.grounding_dino import GroundingDINO

from ingrain.structures.track_instances import TrackInstances
from ingrain.models.ops import inverse_sigmoid
from ingrain.losses.track_loss import INGRAINTrackCriterion


class ObservationStateFusion(nn.Module):
    """Fuse the encoder-ROI track content E with the decoder hidden H:

        out = E + MLP([E ; H])

    The MLP's last layer is zero-initialised, so at start ``out == E`` exactly.
    """

    def __init__(self, dim: int = 256, hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, E: Tensor, H: Tensor) -> Tensor:
        return E + self.mlp(torch.cat([E, H], dim=-1))


class _SimpleView:
    """Lightweight duck-typed TrackInstances view for trajectory-memory calls."""

    def __init__(self, obj_idxes: Tensor, output_embedding: Tensor,
                 query_tgt: Tensor | None = None):
        self.obj_idxes = obj_idxes
        self.output_embedding = output_embedding
        if query_tgt is not None:
            self.query_tgt = query_tgt


@MODELS.register_module()
class INGRAINTracker(GroundingDINO):
    """INGRAIN: Tracking with INherited GRounding Alignment IN the Native
    Vision-Language Query Space.

    A multi-object tracker built by extending Grounding DINO with track
    query propagation, extra track self-attention in the decoder,
    and a multi-depth trajectory memory.

    Args:
        trajectory_memory_cfg (dict, optional): Config for TrajectoryMemory.
            If None, the trajectory memory is disabled.
        track_loss_cfg (dict, optional): Config for INGRAINTrackCriterion.
        num_clip_frames (int): Number of frames per training clip. Default: 2.
        score_thresh (float): Score threshold for new track confirmation
            (inference). Default: 0.4.
        miss_tolerance (int): Max frames a track can disappear before removal
            (inference). Default: 5.
        **kwargs: All GroundingDINO kwargs passed through unchanged.
    """

    def __init__(self,
                 trajectory_memory_cfg: dict = None,
                 track_loss_cfg: dict = None,
                 num_clip_frames: int = 2,
                 score_thresh: float = 0.4,
                 miss_tolerance: int = 5,
                 e2e_assoc: bool = False,
                 e2e_random_drop: float = 0.0,
                 e2e_fp_ratio: float = 0.0,
                 dn_anchor_denoise_p: float = 0.0,
                 dn_anchor_cxy: float = 0.12,
                 dn_anchor_wh: float = 0.2,
                 dn_anchor_object_scaled: bool = False,
                 **kwargs) -> None:
        self._trajectory_memory_cfg = trajectory_memory_cfg
        self._track_loss_cfg = track_loss_cfg or dict()
        # End-to-end association: the losses shape the PROPAGATED track
        # query (not a side embedding/matcher).
        self.e2e_assoc = bool(e2e_assoc)
        # Track-survival augmentation on the propagation path: random drop of
        # real tracks and false-positive decoy tracks.
        self._e2e_random_drop = float(e2e_random_drop)
        self._e2e_fp_ratio = float(e2e_fp_ratio)
        self._PHANTOM_BASE = 10_000_000   # sentinel obj_id floor for FP slots
        self._fp_next_id = self._PHANTOM_BASE
        # Diagnostics (--diag-log); see ingrain/models/diag.py. Set
        # externally by train.py via set_diag().
        self._diag = None        # DiagAccumulator | None
        self._diag_step = 0
        self._global_step = 0
        # The content propagated to the next observation is what OGP and MDM
        # produced, and the track's positional reference is its own previous
        # predicted box. The detection path is left unchanged: it
        # runs the pretrained decoder layers, never the track path.
        self._num_clip_frames = num_clip_frames
        self._score_thresh = score_thresh
        self._miss_tolerance = miss_tolerance
        super().__init__(**kwargs)

        # Build INGRAIN-specific modules after super().__init__
        self.criterion = INGRAINTrackCriterion(**self._track_loss_cfg)

        if self._trajectory_memory_cfg is not None:
            self.trajectory_memory = MODELS.build(
                dict(type='TrajectoryMemory', **self._trajectory_memory_cfg))
        else:
            self.trajectory_memory = None

        # Runtime gate for the multi-depth trajectory memory, driven by the
        # training loop's --mem-warmup-epoch. Independent of the env flags that
        # decide whether the memory modules exist at all: those are build-time,
        # this one is per-epoch. True = memory participates (the default, so
        # inference and any caller that never touches it are unaffected).
        self.memory_enabled = True

        # OGP observation projection: the track query's content is the
        # encoder-ROI feature at its predicted box rather than the decoder
        # hidden. Identity-init, so
        # it starts as the raw encoder-ROI feature in the same 256-d space.
        self.enc_roi_proj = None
        if os.environ.get('INGRAIN_TRACK_CONTENT_ENCROI') == '1':
            import torch.nn as _nn
            self.enc_roi_proj = _nn.Linear(256, 256)
            _nn.init.eye_(self.enc_roi_proj.weight)
            _nn.init.zeros_(self.enc_roi_proj.bias)

        # Fuse the decoder hidden into the enc-ROI track content (concat-MLP,
        # zero-init last layer => starts == enc-ROI). Env-gated INGRAIN_HIDDEN_FUSE=1.
        self.hidden_fuse = None
        if os.environ.get('INGRAIN_HIDDEN_FUSE') == '1':
            self.hidden_fuse = ObservationStateFusion(256)

        # DN anchor denoise: see _propagate_tracks. Train-time only.
        self._dn_anchor_denoise_p = float(dn_anchor_denoise_p or 0.0)
        self._dn_anchor_cxy = float(dn_anchor_cxy)
        self._dn_anchor_wh = float(dn_anchor_wh)
        # Center jitter scaled by the object's own w/h (|dx| <= cxy * w).
        self._dn_anchor_object_scaled = bool(dn_anchor_object_scaled)

        # Track state for inference
        self._next_obj_id = 0

    # ------------------------------------------------------------------
    # Runtime memory helpers
    # ------------------------------------------------------------------

    def reset_memory(self) -> None:
        """Clear per-clip trajectory state."""
        if self.trajectory_memory is not None:
            self.trajectory_memory.reset()

    # ------------------------------------------------------------------
    # Trajectory-conditioned tracking-by-query helpers (enc-RoI / diag)
    # ------------------------------------------------------------------

    def set_diag(self, diag) -> None:
        """Attach a DiagAccumulator (or None) for training-time instrumentation."""
        self._diag = diag

    def _roi_pool_memory(self, memory, spatial_shapes, boxes_cxcywh):
        """ROI-avg-pool the encoder memory at boxes -> (N, C) instance feats.
        Boxes are detached; gradient flows only to the consumer projection."""
        from torchvision.ops import roi_align as _ra
        if memory.dim() == 3:
            memory = memory[0]
        C = memory.shape[-1]
        b = boxes_cxcywh.detach()
        cx, cy, w, h = b.unbind(-1)
        bx = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1).clamp(0, 1)
        N = bx.shape[0]
        acc = torch.zeros(N, C, device=memory.device, dtype=memory.dtype)
        nlev = 0; off = 0
        for hw in spatial_shapes.tolist():
            Hl, Wl = int(hw[0]), int(hw[1])
            m = memory[off:off+Hl*Wl].transpose(0, 1).reshape(1, C, Hl, Wl)
            scale = torch.tensor([Wl, Hl, Wl, Hl], device=bx.device, dtype=bx.dtype)
            rois = torch.cat([torch.zeros(N, 1, device=bx.device, dtype=bx.dtype),
                              bx * scale], dim=-1)
            acc = acc + _ra(m, rois, output_size=1, spatial_scale=1.0,
                            aligned=True).reshape(N, C)
            nlev += 1; off += Hl * Wl
        return acc / max(1, nlev)

    def _collect_diag(self, *, hidden_last, bbox_last,
                         num_dn, num_det, active_trk, det_m, gt_m, gt_inst,
                         cls_last=None):
        """Populate self._diag with the per-frame module diagnostics."""
        if self._diag is None:
            return
        D = self._diag
        # Diagnostics only ever read these out as Python scalars, so drop the
        # graph up front: it keeps the block genuinely no-grad (as claimed
        # below) and avoids a requires_grad->scalar warning on every step.
        hidden_last = hidden_last.detach()
        bbox_last = bbox_last.detach()
        gt_obj = gt_inst['obj_ids']
        gt_box = gt_inst['boxes']
        num_track = (active_trk.obj_idxes.shape[0]
                     if active_trk is not None else 0)

        # ── track-query box IoU + recover-missed + duplicate-fire ──
        det_box = bbox_last[num_dn:num_dn + num_det]
        if num_track > 0:
            trk_box = bbox_last[num_dn + num_det:]
            trk_obj = active_trk.obj_idxes[:num_track].to(torch.long)
            iou_tg = self.criterion._cxcywh_iou(trk_box, gt_box)   # (T, G)
            iou_dg = (self.criterion._cxcywh_iou(det_box, gt_box)
                      if len(gt_box) else None)                    # (Ndet, G)
            n_recover = 0
            qtr_ious = []
            for j in range(num_track):
                oid = int(trk_obj[j].item())
                if oid < 0:
                    continue
                gr = (gt_obj == oid).nonzero(as_tuple=True)[0]
                if len(gr) == 0:
                    continue
                g = int(gr[0].item())
                qtr_ious.append(float(iou_tg[j, g].item()))
                # recover-missed: GT covered by THIS track but by NO det
                if iou_dg is not None and float(iou_tg[j, g]) >= 0.5:
                    if float(iou_dg[:, g].max().item()) < 0.5:
                        n_recover += 1
            D.extend('qtr_iou', qtr_ious)
            D.add('recover_missed', float(n_recover), 1)
            # dup-fire: a GT matched by BOTH a det (Hungarian) and a track
            dup = 0
            for g in (gt_m.tolist() if len(gt_m) else []):
                oid = int(gt_obj[g].item())
                if bool((trk_obj == oid).any()):
                    dup += 1
            D.add('dup_fire', float(dup), 1)

        # ── Classification-drift watch: matched-det cls score + track counts.
        # The classification branches are kept fixed, so drift is unlikely; a
        # collapsing matched-det score would flag that head being perturbed.
        # (Full Base/Novel ClsA is measured at eval.)
        if cls_last is not None and len(det_m) > 0:
            _ds = cls_last[num_dn:num_dn + num_det].sigmoid().max(-1).values
            D.add('det_score', float(_ds[det_m].mean().item()))
        D.add('n_alive', float(num_track))
        D.add('n_newborn', float(len(det_m)))

    # ------------------------------------------------------------------
    # Bbox head forward
    # ------------------------------------------------------------------

    def _bbox_head_forward(self,
                           hidden_states: Tensor,
                           references,
                           memory_text: Tensor,
                           text_token_mask: Tensor,
                           num_dn: int,
                           num_det: int):
        """Replicate GroundingDINOHead.forward for the training path."""
        from ingrain.models.ops import inverse_sigmoid as _inv_sig
        L = hidden_states.shape[0]
        all_cls, all_bbox = [], []
        for layer_id in range(L):
            h = hidden_states[layer_id]              # (B, Q, C) — reg uses this
            outputs_class = self.bbox_head.cls_branches[layer_id](
                h, memory_text, text_token_mask)
            ref = _inv_sig(references[layer_id])
            tmp_reg = self.bbox_head.reg_branches[layer_id](h)
            if ref.shape[-1] == 4:
                tmp_reg = tmp_reg + ref
            else:
                assert ref.shape[-1] == 2
                tmp_reg = torch.cat(
                    [tmp_reg[..., :2] + ref, tmp_reg[..., 2:]], dim=-1)
            outputs_coord = tmp_reg.sigmoid()
            all_cls.append(outputs_class)
            all_bbox.append(outputs_coord)
        return torch.stack(all_cls), torch.stack(all_bbox)

    def _init_layers(self) -> None:
        """Initialize layers, replacing decoder with IngrainTransformerDecoder."""
        # Import here to avoid circular imports
        from ingrain.models.layers.ingrain_decoder import IngrainTransformerDecoder

        # Store full decoder config (with INGRAIN-specific keys) before super call
        decoder_cfg = copy.deepcopy(self.decoder)

        # Remove INGRAIN-specific keys so GD's _init_layers won't choke
        self.decoder.pop('num_isolation_layers', None)

        # Call super to build all GD layers normally
        super()._init_layers()

        # Replace decoder with INGRAIN version (IngrainDecoderLayer)
        # Strip 'type' from layer_cfg since we construct directly
        layer_cfg_clean = {k: v for k, v in decoder_cfg['layer_cfg'].items()
                           if k != 'type'}
        decoder_cfg['layer_cfg'] = layer_cfg_clean
        self.decoder = IngrainTransformerDecoder(**decoder_cfg)

    # ------------------------------------------------------------------
    # Track instance management
    # ------------------------------------------------------------------

    def _generate_empty_tracks(self, device='cpu'):
        """Create fresh detect queries as empty TrackInstances."""
        ti = TrackInstances()
        ti.ref_pts = torch.zeros(self.num_queries, 4, device=device)
        ti.query_tgt = self.query_embedding.weight.detach().clone()
        ti.obj_idxes = torch.full(
            (self.num_queries,), -1, dtype=torch.long, device=device)
        ti.matched_gt_idxes = torch.full(
            (self.num_queries,), -1, dtype=torch.long, device=device)
        ti.scores = torch.zeros(self.num_queries, device=device)
        ti.pred_boxes = torch.zeros(self.num_queries, 4, device=device)
        ti.pred_logits = torch.zeros(
            self.num_queries, 256, device=device)  # max_text_len
        ti.output_embedding = torch.zeros(
            self.num_queries, self.embed_dims, device=device)
        return ti

    # ------------------------------------------------------------------
    # Overridden forward methods
    # ------------------------------------------------------------------

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
        track_instances: TrackInstances = None,
    ) -> Dict:
        """Forward transformer with track query injection."""
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict,
            batch_data_samples=batch_data_samples,
            track_instances=track_instances)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
        track_instances: TrackInstances = None,
    ) -> Tuple[Dict]:
        """GD pre_decoder + track query injection.

        Appends track queries after detect queries:
        query = [dn_queries | Q_det | Q_tr]
        """
        # Clear per-frame trajectory-memory stats before this frame's forward so that
        # downstream diagnostic readers see fresh tensors only.
        if self.trajectory_memory is not None:
            self.trajectory_memory._last_memory_attention_stats = None

        # Call GD's pre_decoder to get [dn | Q_det]
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(
            memory, memory_mask, spatial_shapes,
            memory_text, text_token_mask, batch_data_samples)

        # Determine DN query count
        num_dn = 0
        if self.training and 'dn_meta' in head_inputs_dict and \
                head_inputs_dict['dn_meta'] is not None:
            num_dn = head_inputs_dict['dn_meta'].get(
                'num_denoising_queries', 0)

        num_det = self.num_queries

            # Inject ONLY active track queries (obj_idxes >= 0)
            # The propagated set is [init_det + active_tracks],
            # but only active_tracks should be appended here — GD's proposal
        # already handles detect queries via query_embedding + topk proposals.
        num_track = 0
        if track_instances is not None and len(track_instances) > 0:
            # Filter to only active tracks
            active_mask = track_instances.obj_idxes >= 0
            if active_mask.any():
                active_tracks = track_instances[active_mask]
                bs = memory.shape[0]

                # The track's positional reference is its own previous
                # predicted box, set at propagation via
                # inverse_sigmoid(pred_boxes) — nothing refreshes it here.

                q_tr = active_tracks.query_tgt.unsqueeze(0).repeat(bs, 1, 1)
                ref_tr = active_tracks.ref_pts.unsqueeze(0).repeat(bs, 1, 1)

                decoder_inputs_dict['query'] = torch.cat(
                    [decoder_inputs_dict['query'], q_tr], dim=1)

                # Track refs are in inverse-sigmoid, apply sigmoid
                decoder_inputs_dict['reference_points'] = torch.cat(
                    [decoder_inputs_dict['reference_points'],
                     ref_tr.sigmoid()], dim=1)

                num_track = len(active_tracks)

                if decoder_inputs_dict.get('dn_mask') is not None:
                    decoder_inputs_dict['dn_mask'] = self._extend_dn_mask(
                        decoder_inputs_dict['dn_mask'],
                        num_track, num_dn, num_det)

                # Store the active tracks for criterion to use
                head_inputs_dict['_active_track_instances'] = active_tracks

        # Store boundary info for forward_decoder
        decoder_inputs_dict['num_det'] = num_det
        decoder_inputs_dict['num_dn'] = num_dn
        decoder_inputs_dict['num_track'] = num_track

        return decoder_inputs_dict, head_inputs_dict

    def _extend_dn_mask(self, dn_mask, num_track, num_dn, num_det):
        """Extend DN attention mask to include track queries.

        Track queries follow the same visibility rules as matching queries:
        - Can see all matching queries and track queries
        - Cannot see DN queries
        - DN queries cannot see track queries
        """
        old_size = dn_mask.shape[0]  # num_dn + num_det
        new_size = old_size + num_track
        device = dn_mask.device

        new_mask = torch.zeros(new_size, new_size, dtype=torch.bool,
                               device=device)
        # Copy original mask
        new_mask[:old_size, :old_size] = dn_mask

        # Track queries cannot see DN queries
        new_mask[old_size:, :num_dn] = True

        # DN queries cannot see track queries
        new_mask[:num_dn, old_size:] = True

        # Matching queries (num_dn:num_dn+num_det) CAN see track queries
        # → new_mask[num_dn:old_size, old_size:] stays False

        # Track queries CAN see matching queries and each other
        # → new_mask[old_size:, num_dn:] stays False

        return new_mask

    def forward_decoder(self, query, memory, memory_mask,
                        reference_points, spatial_shapes,
                        level_start_index, valid_ratios,
                        dn_mask=None,
                        num_det=900, num_dn=0, num_track=0,
                        **kwargs):
        """Forward decoder with track boundary info."""
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            num_det_offset=num_dn + num_det,
            num_track=num_track,
            num_det=num_det,
            cls_branches=None,
            **kwargs)

        # Same return format as the base decoder
        references = list(inter_references.unbind(0))
        out = dict(hidden_states=inter_states, references=references)
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Single-frame loss (GD-compatible). For multi-frame, use forward_train_mot()."""
        return self._loss_single_frame(batch_inputs, batch_data_samples)

    def _loss_single_frame(self, batch_inputs, batch_data_samples):
        """Single-frame loss: identical to GD but through INGRAIN's pipeline."""
        # Prepare text (same as GD.loss)
        text_prompts = [ds.text for ds in batch_data_samples]
        gt_labels = [ds.gt_instances.labels for ds in batch_data_samples]

        if 'tokens_positive' in batch_data_samples[0]:
            tokens_positive = [ds.tokens_positive for ds in batch_data_samples]
            positive_maps = []
            for tp, text_prompt, gt_label in zip(
                    tokens_positive, text_prompts, gt_labels):
                tokenized = self.language_model.tokenizer(
                    [text_prompt],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                new_tp = [tp[label.item()] for label in gt_label]
                _, positive_map = self.get_positive_map(tokenized, new_tp)
                positive_maps.append(positive_map)
            new_text_prompts = text_prompts
        else:
            new_text_prompts = []
            positive_maps = []
            if len(set(text_prompts)) == 1:
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(text_prompts[0], True)
                new_text_prompts = [caption_string] * len(batch_inputs)
                for gt_label in gt_labels:
                    new_tp = [tokens_positive[label] for label in gt_label]
                    _, positive_map = self.get_positive_map(tokenized, new_tp)
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(text_prompts, gt_labels):
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(text_prompt, True)
                    new_tp = [tokens_positive[label] for label in gt_label]
                    _, positive_map = self.get_positive_map(tokenized, new_tp)
                    positive_maps.append(positive_map)
                    new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, ds in enumerate(batch_data_samples):
            pm = positive_maps[i].to(batch_inputs.device).bool().float()
            ttm = text_dict['text_token_mask'][i]
            ds.gt_instances.positive_maps = pm
            ds.gt_instances.text_token_mask = ttm.unsqueeze(0).repeat(
                len(pm), 1)

        visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(
            visual_features, text_dict, batch_data_samples,
            track_instances=None)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def forward_train_mot(self, data_dict):
        """Multi-frame training forward pass.

        Pipeline per frame:
          1. extract_feat → encoder → pre_decoder (track inject) → decoder
             → bbox_head.
          2. Push each active track's last-layer hidden state into the
             trajectory memory.
          3. Criterion.compute_track_frame_loss.
          4. Propagate tracks to the next observation.

        Frame weighting is not applied: base calibration is preserved by
        construction.

        Args:
            data_dict: see train.py for schema.  Must include 'imgs',
                'gt_instances', 'text_dict', 'text_token_mask', 'img_shape'.

        Returns:
            loss_dict: mean-per-frame losses accumulated over the clip.
        """
        imgs = data_dict['imgs']
        gt_instances_list = data_dict['gt_instances']
        text_dict = data_dict['text_dict']
        text_token_mask = data_dict['text_token_mask']
        img_shape = data_dict.get('img_shape', None)
        prompt_class_pmaps = data_dict.get('prompt_class_pmaps', None)
        num_frames = len(imgs)
        device = imgs[0].device

        # Reset per-clip state
        self.criterion.reset()
        self.reset_memory()

        # Precompute clip-level text class embeddings.
        # text_class_embeds has shape (C_clip, embed_dim) — one row per
        # class in the clip's prompt.  Computed once per clip.
        # Note: prompt_class_pmaps are padded to GD's max_text_len but
        # text_dict['embedded'] has only the actual token count — so we must
        # truncate the pmap to align with the embedded sequence length.
        text_class_embeds = None
        if (self.trajectory_memory is not None
                and prompt_class_pmaps is not None):
            text_emb = text_dict['embedded'][0]     # (n_tokens, D)
            n_tokens = text_emb.shape[0]
            cls_embeds = []
            for pm in prompt_class_pmaps:
                pm_dev = pm.to(device)
                pm_aligned = pm_dev[:n_tokens].bool()
                if pm_aligned.any():
                    cls_embeds.append(text_emb[pm_aligned].mean(dim=0))
                else:
                    cls_embeds.append(text_emb.new_zeros(text_emb.shape[-1]))
            text_class_embeds = torch.stack(cls_embeds, dim=0)  # (C_clip, D)

        active_tracks = None  # No track queries for first frame

        for t in range(num_frames):
            frame_img = imgs[t]
            gt_inst = gt_instances_list[t]

            # --- Backbone forward (backbone and encoder are kept fixed) ---
            visual_features = self.extract_feat(frame_img)

            from mmdet.structures import DetDataSample
            from mmengine.structures import InstanceData
            ds = DetDataSample()
            real_shape = img_shape if img_shape is not None else frame_img.shape[-2:]
            ds.set_metainfo(dict(
                img_shape=real_shape,
                batch_input_shape=frame_img.shape[-2:],
            ))
            ds.gt_instances = InstanceData(
                bboxes=torch.zeros(0, 4, device=device),
                labels=torch.zeros(0, dtype=torch.long, device=device),
            )
            batch_data_samples = [ds]

            encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
                visual_features, batch_data_samples)
            encoder_outputs_dict = self.forward_encoder(
                **encoder_inputs_dict, text_dict=text_dict)

            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict,
                batch_data_samples=batch_data_samples,
                track_instances=active_tracks)
            decoder_inputs_dict.update(tmp_dec_in)

            # INGRAIN_MEMFUSE_INTERLEAVE: stash the track obj_idxes so
            # _run_track_copy can key the per-depth memory attention, as on
            # the batched path. No per-sample offset: one sample per forward.
            if (getattr(self, 'memory_enabled', True)
                    and os.environ.get('INGRAIN_MEMFUSE_INTERLEAVE') == '1'):
                _act_mf = head_inputs_dict.get('_active_track_instances')
                _oids_attr = getattr(_act_mf, 'obj_idxes', None)
                if (_act_mf is not None and _oids_attr is not None
                        and _oids_attr.numel() > 0):
                    self.decoder._mf_interleave_oids = _oids_attr.clone()
                    object.__setattr__(self.decoder, '_mf_interleave_sc',
                                       self.trajectory_memory)
                else:
                    self.decoder._mf_interleave_oids = None
                    object.__setattr__(self.decoder, '_mf_interleave_sc', None)
            else:
                self.decoder._mf_interleave_oids = None
                object.__setattr__(self.decoder, '_mf_interleave_sc', None)

            decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
            head_inputs_dict.update(decoder_outputs_dict)

            num_dn = 0
            dn_meta = head_inputs_dict.get('dn_meta')
            if dn_meta is not None:
                num_dn = dn_meta.get('num_denoising_queries', 0)
            num_det = self.num_queries

            all_cls_scores, all_bbox_preds = self._bbox_head_forward(
                head_inputs_dict['hidden_states'],
                head_inputs_dict['references'],
                head_inputs_dict['memory_text'],
                head_inputs_dict['text_token_mask'],
                num_dn=num_dn,
                num_det=num_det,
            )

            # Strip DN portion, take last decoder layer.  Shape layout is
            # [DN | Det | Track] along dim 2 (axes: layers, batch, queries, C).
            hidden_last = head_inputs_dict['hidden_states'][-1, 0]
            cls_last = all_cls_scores[-1, 0]
            bbox_last = all_bbox_preds[-1, 0]
            total_q = hidden_last.shape[0]
            num_track = total_q - num_dn - num_det

            active_track_for_criterion = head_inputs_dict.get(
                '_active_track_instances', None)

            # --- Track loss computation ---
            gt_for_crit = InstanceData()
            gt_for_crit.bboxes = gt_inst['boxes']
            gt_for_crit.positive_maps = gt_inst['positive_maps']
            gt_for_crit.obj_ids = gt_inst['obj_ids']

            ttm_row = text_token_mask[0] if text_token_mask.dim() == 2 \
                else text_token_mask

            frame_losses, det_matched, gt_matched = \
                self.criterion.compute_track_frame_loss(
                    hidden_states_last=hidden_last,
                    cls_scores_last=cls_last,
                    bbox_preds_last=bbox_last,
                    gt_instance=gt_for_crit,
                    text_token_mask=ttm_row,
                    active_track_instances=active_track_for_criterion,
                    num_dn=num_dn,
                    num_det=num_det,
                    img_shape=img_shape,
                )
            # Update trajectory memory (seed_mem_bank) for next-frame attention.
            # INGRAIN_ENCROI_MEM_FUSE=1: the bank is written with enc-ROI content
            # inside _propagate_tracks; skip the decoder-hidden write here so one
            # bank does not mix two feature spaces (mirrors the batched path).
            if (self.trajectory_memory is not None
                    and os.environ.get('INGRAIN_ENCROI_MEM_FUSE') != '1'
                    and text_class_embeds is not None
                    and num_track > 0
                    and active_track_for_criterion is not None):
                # Build a lightweight TrackInstances-like view for the trajectory memory
                track_hidden = hidden_last[num_dn + num_det:]   # (num_track, C)
                tmem_view = _SimpleView(
                    obj_idxes=active_track_for_criterion.obj_idxes,
                    output_embedding=track_hidden)
                self.trajectory_memory.update_memory(tmem_view, frame_idx=t)

            if self._diag is not None:
                self._collect_diag(
                    hidden_last=hidden_last, bbox_last=bbox_last,
                    num_dn=num_dn, num_det=num_det,
                    active_trk=active_track_for_criterion,
                    det_m=det_matched, gt_m=gt_matched, gt_inst=gt_inst,
                    cls_last=cls_last)

            self.criterion.accumulate_frame_losses(frame_losses, frame_weight=1.0)

            # --- Propagate tracks for next frame ---
            is_last = (t == num_frames - 1)
            if not is_last:
                # We only have real-GT matches for det (no "trk matches" from
                # Hungarian since track-lock is separate).  Extract trk matches
                # here for _propagate_tracks compatibility.
                trk_matched_pred = []
                trk_matched_gt = []
                if (num_track > 0 and active_track_for_criterion is not None):
                    gt_obj_ids = gt_inst['obj_ids']
                    for j in range(num_track):
                        oid = active_track_for_criterion.obj_idxes[j].item()
                        if oid < 0:
                            continue
                        match = (gt_obj_ids == oid).nonzero(as_tuple=True)[0]
                        if len(match) > 0:
                            trk_matched_pred.append(num_det + j)
                            trk_matched_gt.append(match[0].item())
                trk_mp_t = torch.tensor(
                    trk_matched_pred, dtype=torch.long, device=device)
                trk_mg_t = torch.tensor(
                    trk_matched_gt, dtype=torch.long, device=device)

                active_tracks = self._propagate_tracks(
                    head_inputs_dict, gt_inst,
                    all_cls_scores, all_bbox_preds,
                    det_matched, gt_matched,
                    trk_mp_t, trk_mg_t,
                    num_dn=num_dn, num_det=num_det,
                    # The enc-ROI observation and the fusion gate need this
                    # frame's encoder memory during TRAINING. It lives in
                    # encoder_outputs_dict, which is where it is read from, so
                    # enc_roi_proj/hidden_fuse stay in the gradient path.
                    memory=encoder_outputs_dict.get('memory'),
                    spatial_shapes=encoder_outputs_dict.get('spatial_shapes'),
                    prev_track_obj_idxes=(
                        active_track_for_criterion.obj_idxes
                        if (num_track > 0
                            and active_track_for_criterion is not None)
                        else None))

        return self.criterion.get_losses()

    def _propagate_tracks(self, head_inputs_dict, gt_inst,
                          all_cls_scores, all_bbox_preds,
                          det_matched, gt_matched,
                          trk_matched_pred, trk_matched_gt,
                          num_dn=0, num_det=900,
                          prev_track_obj_idxes=None,
                          memory=None, spatial_shapes=None,
                          tmem_obj_offset=0):
        """Propagate tracks to next frame.

        Reuses predictions from the existing bbox_head call (no redundant call).
        Assigns obj_ids to BOTH matched det AND matched track queries.
        """
        device = head_inputs_dict['hidden_states'].device
        hidden_states = head_inputs_dict['hidden_states']

        # Last layer output, strip DN
        hs_last = hidden_states[-1, 0, num_dn:]  # (num_det + num_track, C)

        # Reuse predictions from the bbox_head call in forward_train_mot
        pred_logits = all_cls_scores[-1, 0, num_dn:]
        pred_boxes = all_bbox_preds[-1, 0, num_dn:]

        total_queries = hs_last.shape[0]
        gt_obj_ids = gt_inst['obj_ids']   # track IDs

        # Build TrackInstances for all queries (det + track)
        ti = TrackInstances()
        ti.output_embedding = hs_last.detach()
        ti.pred_logits = pred_logits.detach()
        ti.pred_boxes = pred_boxes.detach()
        ti.scores = pred_logits.detach().sigmoid().max(-1).values
        # Carry-forward ref: track ref = own prev predicted box (logit space).
        _ref_logit = inverse_sigmoid(
            pred_boxes.detach().clone().clamp(1e-4, 1 - 1e-4))
        ti.ref_pts = _ref_logit
        ti.query_tgt = hs_last.detach().clone()
        ti.obj_idxes = torch.full(
            (total_queries,), -1, dtype=torch.long, device=device)
        ti.matched_gt_idxes = torch.full(
            (total_queries,), -1, dtype=torch.long, device=device)

        # Assign obj_ids for MATCHED DETECT queries
        if len(det_matched) > 0:
            for dm, gm in zip(det_matched, gt_matched):
                dm_val = dm.item() if isinstance(dm, torch.Tensor) else dm
                gm_val = gm.item() if isinstance(gm, torch.Tensor) else gm
                ti.obj_idxes[dm_val] = gt_obj_ids[gm_val]
                ti.matched_gt_idxes[dm_val] = gm_val

        # Assign obj_ids for MATCHED TRACK queries.
        if len(trk_matched_pred) > 0:
            for tp, tg in zip(trk_matched_pred, trk_matched_gt):
                tp_val = tp.item() if isinstance(tp, torch.Tensor) else tp
                tg_val = tg.item() if isinstance(tg, torch.Tensor) else tg
                ti.obj_idxes[tp_val] = gt_obj_ids[tg_val]
                ti.matched_gt_idxes[tp_val] = tg_val

        # Anchor denoise (training only, gated by self.training): with prob p,
        # replace a matched row's carried ref with a jittered GT box. Requires
        # --train-track-ref-refine and isolation < 6, else the carried ref is
        # detached.
        _dn_p = float(getattr(self, '_dn_anchor_denoise_p', 0.0) or 0.0)
        if self.e2e_assoc and self.training and _dn_p > 0.0:
            _gt_boxes = gt_inst.get('boxes')
            if _gt_boxes is not None and _gt_boxes.shape[0] > 0:
                _cxy = float(getattr(self, '_dn_anchor_cxy', 0.12))
                _wh = float(getattr(self, '_dn_anchor_wh', 0.2))
                _n_dn = 0; _shift_dbg = 0.0
                for j in range(total_queries):
                    gm = int(ti.matched_gt_idxes[j].item())
                    if gm < 0 or gm >= _gt_boxes.shape[0]:
                        continue
                    if float(torch.rand((), device=device)) >= _dn_p:
                        continue
                    gb = _gt_boxes[gm].to(device).float()   # cxcywh [0,1]
                    # Object-scaled center jitter (|dx| <= cxy*w); the
                    # frame-fraction alternative is off.
                    _cxy_scale = (gb[2:] if getattr(
                        self, '_dn_anchor_object_scaled', False) else 1.0)
                    cxy = gb[:2] + (torch.rand(2, device=device) * 2 - 1) * _cxy * _cxy_scale
                    whs = gb[2:] * (1.0 + (torch.rand(2, device=device) * 2 - 1) * _wh)
                    jb = torch.cat([cxy, whs]).clamp(1e-4, 1 - 1e-4)
                    _shift_dbg += float((ti.ref_pts[j].sigmoid()[:2] - jb[:2]).abs().sum())
                    ti.ref_pts[j] = inverse_sigmoid(jb)
                    _n_dn += 1
                # One-shot confirmation that the track-anchor denoising fired,
                # with the resulting reference-shift magnitude.
                if _n_dn > 0 and not getattr(self, '_dn_dbg_printed', False):
                    self._dn_dbg_printed = True
                    print(f"[track-anchor denoise] {_n_dn} matched-track refs jittered to GT "
                          f"(mean |center-shift|={_shift_dbg/max(_n_dn,1):.3f}, cxy={_cxy}, obj_scaled="
                          f"{bool(getattr(self,'_dn_anchor_object_scaled',False))}) → next-frame loss_box_track trains track-path reg recovery", flush=True)

        # In det_pool_mode='all', det queries are the sole bbox/cls emission
        # owner and are trained to match every visible GT, including objects
        # already carried by a track. That means a det query and a track query
        # can both receive the same obj_id in this frame. Do not propagate both
        # as separate trajectory states; keep the first occurrence, and because
        # layout is [det | track], this gives priority to the matched det state.
        if getattr(self.criterion, 'det_pool_mode', 'remaining') == 'all':
            seen_obj_ids = set()
            active_idx = (ti.obj_idxes >= 0).nonzero(as_tuple=True)[0]
            for idx_t in active_idx:
                idx = int(idx_t.item())
                oid = int(ti.obj_idxes[idx].item())
                if oid in seen_obj_ids:
                    ti.obj_idxes[idx] = -1
                    ti.matched_gt_idxes[idx] = -1
                else:
                    seen_obj_ids.add(oid)

        # Under --e2e-assoc the track slot's propagated content = the
        # q_next (full content, REPLACE not residual), IN-GRAPH so the box/cls
        # losses shape what propagates. Newborn tracks (no cached q_next) seed
        # from the in-graph det hidden hs_last[i]. Det slots keep the detached
        # base — they are NOT carried (only obj_idxes>=0 propagate).
        # ── Track-survival augmentation (training-only, e2e path): drop
        # real tracks at random and promote decoy tracks from unmatched
        # detections. ──────────────────────────────────────────────────────
        # Runs before the q_next write-loop so decoy tracks get the newborn seed.
        if self.e2e_assoc and self.training:
            # random_drop: kill each real active track with prob p — its
            # object re-births as a newborn next frame (teaches recovery).
            if getattr(self, '_e2e_random_drop', 0.0) > 0:
                for i in range(ti.obj_idxes.shape[0]):
                    oid = int(ti.obj_idxes[i].item())
                    if (0 <= oid < self._PHANTOM_BASE
                            and float(torch.rand(())) < self._e2e_random_drop):
                        ti.obj_idxes[i] = -1
            # FP-insert: fake-promote unmatched det slots with sentinel ids
            # >= _PHANTOM_BASE. They propagate exactly like real tracks
            # (trajectory state + q_next). The sentinel never collides with a
            # real GT id, so these slots never match any GT obj_id and the
            # box/cls track losses (matched by obj_id equality) skip them
            # automatically.
            if getattr(self, '_e2e_fp_ratio', 0.0) > 0:
                unmatched = [i for i in range(ti.obj_idxes.shape[0])
                             if int(ti.obj_idxes[i].item()) == -1]
                n_real = int(((ti.obj_idxes >= 0)
                              & (ti.obj_idxes < self._PHANTOM_BASE)).sum())
                k = min(len(unmatched),
                        int(round(self._e2e_fp_ratio * max(n_real, 1))))
                if k > 0:
                    perm = torch.randperm(len(unmatched))[:k]
                    for p in perm.tolist():
                        ti.obj_idxes[unmatched[p]] = self._fp_next_id
                        self._fp_next_id += 1

        if self.e2e_assoc:
            # Carried-track content = the RAW decoder hidden (in-graph), so the
            # next-frame box/cls losses shape what propagates. The enc-ROI +
            # hidden_fuse block below then fuses E with this hd.
            for i in range(ti.query_tgt.shape[0]):
                oid = int(ti.obj_idxes[i].item())
                if oid < 0:
                    continue
                ti.query_tgt[i] = hs_last[i]   # carried/newborn seed, in-graph

        # Replace the track-query content with the encoder-ROI observation at
        # each slot's predicted box (the encoder is kept fixed, so gradient reaches
        # only enc_roi_proj). Runs last, after every other q_tgt modification.
        if (os.environ.get('INGRAIN_TRACK_CONTENT_ENCROI') == '1'
                and self.enc_roi_proj is not None and memory is not None
                and spatial_shapes is not None):
            _roi = self._roi_pool_memory(memory, spatial_shapes, ti.pred_boxes)
            _E = self.enc_roi_proj(_roi.to(ti.query_tgt.dtype))
            if getattr(self, 'hidden_fuse', None) is not None:
                # ti.query_tgt is still the decoder hidden (hs_last) here — fuse it
                # INTO the enc-ROI content. zero-init MLP => starts == enc-ROI.
                ti.query_tgt = self.hidden_fuse(_E, ti.query_tgt)
            else:
                ti.query_tgt = _E

        # INGRAIN_ENCROI_MEM_FUSE=1: trajectory memory at depth d=0 (the seed).
        # Refine the propagated track content with each track's OWN trajectory
        # history (memory attention), then store this observation's content as
        # next-frame history. Read+write use the same per-sample offset namespace
        # as the shared trajectory-memory bank (tmem_obj_offset). seed_mem_attn out_proj is
        # zero-init → identity at start; gradient reaches it via this in-graph
        # query and the next-frame track box/cls losses.
        #
        # Deliberately OUTSIDE the enc-ROI block above: the seed refines whatever
        # content is being propagated — the enc-ROI observation (optionally fused
        # with the decoded state) with INGRAIN_TRACK_CONTENT_ENCROI=1, the decoded
        # track state with it off. It sits outside that branch so the memory
        # depth set stays d={0,1,2,3} either way, keeping the enc-ROI switch a
        # single change.
        if (getattr(self, 'memory_enabled', True)
                and os.environ.get('INGRAIN_ENCROI_MEM_FUSE') == '1'
                and self.trajectory_memory is not None
                and getattr(self.trajectory_memory, 'seed_mem_attn', None)
                is not None):
            _mf_ids = ti.obj_idxes.clone()
            _mf_v = _mf_ids >= 0
            _mf_ids[_mf_v] = _mf_ids[_mf_v] + int(tmem_obj_offset)
            _mf_obs = ti.query_tgt.detach()   # propagated content, this frame
            ti.query_tgt = self.trajectory_memory.propagate_memory_state(
                q_tr=ti.query_tgt, active_obj_idxes=_mf_ids)
            self.trajectory_memory.seed_mem_bank.update(_mf_ids, _mf_obs)

        # init_tracks are not fed to pre_decoder.
        init_tracks = self._generate_empty_tracks(device=device)
        # Identity propagation: only matched tracks (obj_idxes >= 0) carry over,
        # with query_tgt = the fused content and ref_pts =
        # inverse_sigmoid(pred_boxes) already set above.
        active = ti[ti.obj_idxes >= 0]
        return TrackInstances.cat([init_tracks, active])

    # ------------------------------------------------------------------
    # Batched multi-sample training (batch backbone+encoder)
    # ------------------------------------------------------------------

    def forward_train_mot_batched(self, data_dict):
        """Multi-sample batched training.

        Same per-frame pipeline as ``forward_train_mot`` but with:
          - Batched backbone + encoder across bs samples (speed)
          - Per-sample decoder (track queries differ per sample)
          - Per-sample trajectory-memory namespace
        """
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        imgs = data_dict['imgs']               # num_frames x (bs, 3, H, W)
        gt_all = data_dict['gt_instances']     # bs x num_frames x dict
        text_dict = data_dict['text_dict']
        text_token_mask = data_dict['text_token_mask']
        img_shapes = data_dict['img_shapes']   # bs x (H, W)
        prompt_class_pmaps = data_dict.get('prompt_class_pmaps', None)
        bs = data_dict['batch_size']
        num_frames = len(imgs)
        device = imgs[0].device

        self.criterion.reset()
        self.reset_memory()
        # Per-sample memory banks (isolated — obj_ids don't share across clips)
        active_tracks_list = [None] * bs

        # Precompute clip-level text class embeddings.
        # All samples share the same clip-level prompt, so text_class_embeds
        # is computed once outside the frame loop.  Per-sample separation in
        # the trajectory memory's internal banks is achieved by offsetting obj_idxes (see the
        # SAMPLE_OBJ_OFFSET below).
        text_class_embeds = None
        if self.trajectory_memory is not None and prompt_class_pmaps is not None:
            text_emb = text_dict['embedded'][0]       # (n_tokens, D)
            n_tokens = text_emb.shape[0]
            cls_embeds = []
            for pm in prompt_class_pmaps:
                pm_aligned = pm.to(device)[:n_tokens].bool()
                if pm_aligned.any():
                    cls_embeds.append(text_emb[pm_aligned].mean(dim=0))
                else:
                    cls_embeds.append(text_emb.new_zeros(text_emb.shape[-1]))
            text_class_embeds = torch.stack(cls_embeds, dim=0)  # (C_clip, D)

        # Each sample gets a disjoint obj_id namespace inside the trajectory-memory
        # bank to avoid cross-sample memory pollution while keeping a single
        # shared trajectory-memory instance.  10 million is safely above per-clip obj_id
        # counts (LVIS annotation.id values fit in < 10M).
        SAMPLE_OBJ_OFFSET = 10_000_000

        for t in range(num_frames):
            frame_imgs = imgs[t]  # (bs, 3, H, W)

            # ── Batched backbone + encoder ──────────────────────────
            batch_ds = []
            for i in range(bs):
                ds = DetDataSample()
                ds.set_metainfo(dict(
                    img_shape=img_shapes[i],
                    batch_input_shape=frame_imgs.shape[-2:],
                ))
                ds.gt_instances = InstanceData(
                    bboxes=torch.zeros(0, 4, device=device),
                    labels=torch.zeros(0, dtype=torch.long, device=device),
                )
                batch_ds.append(ds)

            visual_features = self.extract_feat(frame_imgs)   # batched

            encoder_inputs, decoder_inputs = self.pre_transformer(
                visual_features, batch_ds)

            text_dict_bs = {}
            for k, v in text_dict.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == 1 and bs > 1:
                    text_dict_bs[k] = v.expand(bs, *v.shape[1:]).contiguous()
                else:
                    text_dict_bs[k] = v

            encoder_outputs = self.forward_encoder(
                **encoder_inputs, text_dict=text_dict_bs)   # batched

            # ── Per-sample decoder + loss ───────────────────────────
            _shared_keys = {'spatial_shapes', 'level_start_index'}
            for i in range(bs):
                gt_inst = gt_all[i][t]

                # Slice encoder outputs for sample i
                enc_out_i = {}
                for k, v in encoder_outputs.items():
                    if k in _shared_keys:
                        enc_out_i[k] = v
                    elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == bs:
                        enc_out_i[k] = v[i:i+1]
                    else:
                        enc_out_i[k] = v

                # Pre-decoder per-sample
                tmp_dec, head_inputs = self.pre_decoder(
                    **enc_out_i,
                    batch_data_samples=[batch_ds[i]],
                    track_instances=active_tracks_list[i])

                dec_in_i = {}
                for k, v in decoder_inputs.items():
                    if k in _shared_keys:
                        dec_in_i[k] = v
                    elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == bs:
                        dec_in_i[k] = v[i:i+1]
                    else:
                        dec_in_i[k] = v
                dec_in_i.update(tmp_dec)

                # Produce a pure-GD det-slice reference for the anchor.
                # INGRAIN_MEMFUSE_INTERLEAVE: stash this sample's track
                # obj_idxes (offset-applied, SAME namespace as the memory
                # bank) on the decoder so _run_track_copy can key the per-depth
                # interleaved memory attention. Train-path only; always set (to
                # ids or None) so no stale value survives across samples.
                if (getattr(self, 'memory_enabled', True)
                        and os.environ.get('INGRAIN_MEMFUSE_INTERLEAVE') == '1'):
                    _act_mf = head_inputs.get('_active_track_instances')
                    _oids_attr = getattr(_act_mf, 'obj_idxes', None)
                    if (_act_mf is not None and _oids_attr is not None
                            and _oids_attr.numel() > 0):
                        _oids_mf = _act_mf.obj_idxes.clone()
                        _vmf = _oids_mf >= 0
                        _oids_mf[_vmf] = _oids_mf[_vmf] + i * SAMPLE_OBJ_OFFSET
                        self.decoder._mf_interleave_oids = _oids_mf
                        # object.__setattr__ bypasses nn.Module.__setattr__, so
                        # the trajectory memory is NOT registered as a decoder submodule.
                        # Plain assignment aliases the whole memory stack into
                        # every state_dict as decoder._mf_interleave_sc.*, which
                        # then loads as 60+ "unexpected" keys — routine noise
                        # that hides a genuine architecture mismatch.
                        object.__setattr__(self.decoder, '_mf_interleave_sc',
                                           self.trajectory_memory)
                    else:
                        self.decoder._mf_interleave_oids = None
                        object.__setattr__(self.decoder, '_mf_interleave_sc', None)
                else:
                    self.decoder._mf_interleave_oids = None
                    object.__setattr__(self.decoder, '_mf_interleave_sc', None)
                dec_out = self.forward_decoder(**dec_in_i)
                head_inputs.update(dec_out)

                num_dn = 0
                dn_meta = head_inputs.get('dn_meta')
                if dn_meta:
                    num_dn = dn_meta.get('num_denoising_queries', 0)
                num_det = self.num_queries

                all_cls, all_bbox = self._bbox_head_forward(
                    head_inputs['hidden_states'],
                    head_inputs['references'],
                    head_inputs['memory_text'],
                    head_inputs['text_token_mask'],
                    num_dn=num_dn,
                    num_det=num_det,
                )

                hidden_last = head_inputs['hidden_states'][-1, 0]
                cls_last = all_cls[-1, 0]
                bbox_last = all_bbox[-1, 0]
                total_q = hidden_last.shape[0]
                num_track = total_q - num_dn - num_det

                active_trk_i = head_inputs.get('_active_track_instances', None)

                # text_token_mask for this sample
                ttm_i = text_token_mask[i] if text_token_mask.dim() == 2 \
                    and text_token_mask.shape[0] > 1 \
                    else (text_token_mask[0] if text_token_mask.dim() == 2
                          else text_token_mask)

                gt_crit = InstanceData()
                gt_crit.bboxes = gt_inst['boxes']
                gt_crit.positive_maps = gt_inst['positive_maps']
                gt_crit.obj_ids = gt_inst['obj_ids']

                frame_losses, det_m, gt_m = \
                self.criterion.compute_track_frame_loss(
                        hidden_states_last=hidden_last,
                        cls_scores_last=cls_last,
                        bbox_preds_last=bbox_last,
                        gt_instance=gt_crit,
                        text_token_mask=ttm_i,
                        active_track_instances=active_trk_i,
                        num_dn=num_dn,
                        num_det=num_det,
                        img_shape=img_shapes[i],
                    )
                # Per-sample trajectory-memory update (seed_mem_bank).
                if (self.trajectory_memory is not None
                        and text_class_embeds is not None
                        and num_track > 0
                        and active_trk_i is not None):
                    track_hidden_i = hidden_last[num_dn + num_det:]  # (num_track, C)
                    # Offset obj_idxes so sample i's memory namespace is
                    # disjoint from sample j's in the shared trajectory memory.
                    offset = i * SAMPLE_OBJ_OFFSET
                    tmem_obj_idxes = active_trk_i.obj_idxes.clone()
                    valid_mask = tmem_obj_idxes >= 0
                    tmem_obj_idxes[valid_mask] = tmem_obj_idxes[valid_mask] + offset
                    tmem_view = _SimpleView(
                        obj_idxes=tmem_obj_idxes,
                        output_embedding=track_hidden_i)
                    # INGRAIN_ENCROI_MEM_FUSE=1: the bank is written with enc-ROI
                    # content inside _propagate_tracks (same space as the enc-ROI
                    # query); skip the decoder-hidden write here so one bank does
                    # not mix two feature spaces.
                    if os.environ.get('INGRAIN_ENCROI_MEM_FUSE') != '1':
                        self.trajectory_memory.update_memory(tmem_view, frame_idx=t)

                if self._diag is not None:
                    self._collect_diag(
                        hidden_last=hidden_last, bbox_last=bbox_last,
                        num_dn=num_dn, num_det=num_det,
                        active_trk=active_trk_i,
                        det_m=det_m, gt_m=gt_m, gt_inst=gt_inst,
                        cls_last=cls_last)

                self.criterion.accumulate_frame_losses(
                    frame_losses, frame_weight=1.0)

                # Propagate tracks for next frame
                is_last = (t == num_frames - 1)
                if not is_last:
                    # Track matches (for _propagate_tracks)
                    trk_m_p = []
                    trk_m_g = []
                    if num_track > 0 and active_trk_i is not None:
                        gt_obj_ids = gt_inst['obj_ids']
                        for j in range(num_track):
                            oid = active_trk_i.obj_idxes[j].item()
                            if oid < 0:
                                continue
                            match = (gt_obj_ids == oid).nonzero(as_tuple=True)[0]
                            if len(match) > 0:
                                trk_m_p.append(num_det + j)
                                trk_m_g.append(match[0].item())
                    trk_m_p_t = torch.tensor(
                        trk_m_p, dtype=torch.long, device=device)
                    trk_m_g_t = torch.tensor(
                        trk_m_g, dtype=torch.long, device=device)

                    active_tracks_list[i] = self._propagate_tracks(
                        head_inputs, gt_inst,
                        all_cls, all_bbox,
                        det_m, gt_m, trk_m_p_t, trk_m_g_t,
                        num_dn=num_dn, num_det=num_det,
                        memory=enc_out_i.get('memory'),
                        spatial_shapes=enc_out_i.get('spatial_shapes'),
                        tmem_obj_offset=i * SAMPLE_OBJ_OFFSET,
                        prev_track_obj_idxes=(
                            active_trk_i.obj_idxes
                            if (num_track > 0 and active_trk_i is not None)
                            else None))

        # criterion accumulated bs × num_frames × weight=1.0, get_losses
        # divides by _total_frame_weight, so the return is per-(sample, frame) mean.
        return self.criterion.get_losses()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, batch_inputs, batch_data_samples,
                rescale: bool = True):
        """Single-frame inference with track management.

        Track instances are stored in batch_data_samples for cross-frame
        propagation during video inference.
        """
        # For pure detection (no tracking), use GD's predict
        if not hasattr(batch_data_samples[0], 'track_instances'):
            return super().predict(batch_inputs, batch_data_samples, rescale)

        # Get or initialize track instances
        track_instances = batch_data_samples[0].get(
            'track_instances', None)

        # Text processing (same as GD predict)
        text_prompts = [ds.text for ds in batch_data_samples]
        custom_entities = batch_data_samples[0].get('custom_entities', False)
        tokens_positives = [ds.get('tokens_positive', None)
                           for ds in batch_data_samples]
        enhanced_text_prompts = [ds.get('caption_prompt', None)
                                for ds in batch_data_samples]

        _positive_maps_and_prompts = [
            self.get_tokens_positive_and_prompts(
                text_prompts[0], custom_entities,
                enhanced_text_prompts[0], tokens_positives[0])
        ] * len(batch_inputs)
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        text_dict = self.language_model(list(text_prompts))
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, ds in enumerate(batch_data_samples):
            ds.token_positive_map = token_positive_maps[i]

        # Visual features
        visual_feats = self.extract_feat(batch_inputs)

        # Forward with track queries
        head_inputs_dict = self.forward_transformer(
            visual_feats, text_dict, batch_data_samples,
            track_instances=track_instances)

        # Get predictions
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        # Process results and update tracks
        for ds, pred_instances, entity in zip(
                batch_data_samples, results_list, entities):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(entity):
                        label_names.append('unknown')
                    else:
                        label_names.append(entity[labels])
                pred_instances.label_names = label_names
            ds.pred_instances = pred_instances

        return batch_data_samples
