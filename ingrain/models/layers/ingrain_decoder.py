"""IngrainTransformerDecoder: GD decoder that passes track query boundary
information (num_det_offset, num_track) to each IngrainDecoderLayer.

Inherits from GroundingDinoTransformerDecoder and overrides forward()
to thread the track/detect boundary through the layer loop.
"""
import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import ModuleList

from mmdet.registry import MODELS

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'mmgroundingdino'))
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder)
from mmdet.models.layers.transformer.utils import (
    MLP, coordinate_to_encoding, inverse_sigmoid)

from .ingrain_decoder_layer import IngrainDecoderLayer


@MODELS.register_module()
class IngrainTransformerDecoder(GroundingDinoTransformerDecoder):
    """GD Transformer decoder with track query support.

    Overrides _init_layers to use IngrainDecoderLayer, and forward to pass
    num_det_offset and num_track to each layer.

    Args:
        num_isolation_layers (int): Number of initial decoder layers where
            det queries and track queries cannot attend to each other in
            self-attention (content isolation mask) — the decoupling stage.
            Default: 1; the released recipe passes 3.
    """

    def __init__(self, num_isolation_layers: int = 1, **kwargs):
        self.num_isolation_layers = num_isolation_layers
        super().__init__(**kwargs)
        # Trainable track-ref refinement: when True, the inter-layer
        # reference for the track slice keeps gradient in the fusion-stage layers
        # (lid >= num_isolation_layers, where reg_branches are trainable), so
        # loss_box_track trains the iterative relocation. Det/DN slices and all
        # early layers stay detached, so detection is unperturbed.
        self.train_track_ref_refine = False
        # Directional isolation: track reads det, det never reads track. Set via
        # --iso-directional.
        self.iso_directional = False

    @staticmethod
    def _build_half_isolation_mask(self_attn_mask, num_det_offset, num_track,
                                   device=None):
        """Fusion-stage half mask: block det+DN -> track (True), keep
        track -> det+DN open. Layout [DN | Det | Track]; track starts at
        num_det_offset. Used by the directional-isolation mode."""
        if num_track == 0:
            return self_attn_mask
        total = num_det_offset + num_track
        if self_attn_mask is not None:
            m = self_attn_mask.clone()
        else:
            m = torch.zeros(total, total, dtype=torch.bool, device=device)
        m[:num_det_offset, num_det_offset:] = True   # det+DN cannot read track
        return m

    def _init_layers(self) -> None:
        """Initialize decoder layers with IngrainDecoderLayer."""
        self.layers = ModuleList([
            IngrainDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def _build_isolation_mask(self_attn_mask,
                              num_det_offset: int,
                              num_track: int,
                              device=None) -> Tensor:
        """Build content isolation mask that blocks det<->track attention.

        Returns a mask with det↔track cross-positions blocked (True = blocked).
        If self_attn_mask is None (e.g. DN disabled), creates a fresh mask.

        Layout: [DN | Det | Track]
          - num_det_offset = num_dn + num_det
          - Det range: [num_dn, num_det_offset)
          - Track range: [num_det_offset, num_det_offset + num_track)
        """
        if num_track == 0:
            return self_attn_mask

        total = num_det_offset + num_track
        if self_attn_mask is not None:
            # self_attn_mask already includes track queries (extended by
            # _extend_dn_mask), so its size should be total x total.
            # Clone and add isolation on top.
            iso_mask = self_attn_mask.clone()
        else:
            # No DN mask — create a clean mask (all False = all visible)
            iso_mask = torch.zeros(total, total, dtype=torch.bool,
                                   device=device)

        track_start = num_det_offset

        # Block det+dn queries → track queries  (DN→track already True if DN exists)
        iso_mask[:track_start, track_start:] = True
        # Block track queries → det+dn queries  (track→DN already True if DN exists)
        iso_mask[track_start:, :track_start] = True

        return iso_mask

    def configure_track_decoder_copy(self, reg_branches,
                                     n_split_layers=None,
                                     copy_parts=None,
                                     release_text_ca=False) -> int:
        """Build the track path of the decoupling stage: a separate, fully
        trainable set of decoder layers dedicated to the track query.

        When n_split_layers < num_layers the track path covers only the first
        ``n`` layers — the decoupling stage, where det stays on the detection
        path (self.layers), track on these trainable layers. At the stage boundary
        (layer ``n``) the track hidden produced here is injected back into the
        main [DN|Det|Track] tensor and the remaining (num_layers - n) layers of
        self.layers decode det+track jointly: the fusion stage. When
        n_split_layers is None the track path spans all layers and there is no
        fusion stage.

        Det keeps going through the detection path ``self.layers`` (preserves
        detection + open-vocab by construction); the track query is re-decoded
        through the track path with track-reads-det attention (via self-attn, no
        det<->track isolation), a trainable ref_point_head (so the track ref can
        shift from the read det position), trainable deformable image cross-attn
        and reg, and no ref detach (full gradient on the iterative
        track relocation).

        Text cross-attn (``cross_attn_text``) and the cls head are kept as
        pretrained. Registered submodules => saved in ckpt + appear in the optimizer.
        Built explicitly at setup by both call sites (train.py, evaluate.py),
        before any warm-start checkpoint is applied; a warm start then reaches
        the track path through the checkpoint's own decoder.track_dec_* keys."""
        if getattr(self, 'track_dec_layers', None) is not None:
            return 0  # idempotent
        import copy as _copy
        import logging
        _nl_total = len(self.layers)
        n = _nl_total if n_split_layers is None else int(n_split_layers)
        assert 1 <= n <= _nl_total, \
            f"n_split_layers must be in [1, {_nl_total}], got {n}"
        self.track_dec_n_layers = n
        # nn.ModuleList wrap so the new layer slice REGISTERS as a submodule
        # (a bare list of modules would not appear in state_dict / optimizer).
        self.track_dec_layers = nn.ModuleList(
            _copy.deepcopy(_l) for _l in self.layers[:n])
        # Track SAMPLING adapter (INGRAIN_TRACK_SAMPLING_ADAPTER) — stage-aware
        # placement so it sits ONLY where the TRACK query is actually processed:
        #   • the track path of the decoupling stage (track layers [0,n)) → KEEP
        #   • the fusion stage (main self.layers[n:], det+track joint)    → KEEP
        # The main decoder's decoupling-stage track slice (self.layers[0:n]) is a
        # PHANTOM: it is overwritten by the track path's output at the stage
        # boundary (layer n), so an adapter there only sees dead-end aux-loss
        # gradient and never reaches the final track output. Drop it.
        if os.environ.get('INGRAIN_TRACK_SAMPLING_ADAPTER') == '1':
            for _ml in self.layers[:n]:
                if getattr(_ml, 'track_sampling_adapter', None) is not None:
                    _ml.track_sampling_adapter = None
            _main_kept = [i for i in range(len(self.layers))
                          if getattr(self.layers[i],
                                     'track_sampling_adapter', None) is not None]
            logging.getLogger().info(
                f"[track-adapter] sampling-adapter placement (split depth n={n}): track "
                f"path track_dec_layers[0:{n}] + main decoder.layers{_main_kept} "
                f"(main decoder.layers[0:{n}] excluded — the track path owns "
                f"the decoupling stage)")
        self.track_dec_rph = _copy.deepcopy(self.ref_point_head)
        self.track_dec_norm = _copy.deepcopy(self.norm)
        self.track_dec_reg = nn.ModuleList(
            _copy.deepcopy(_r) for _r in reg_branches[:n])
        n_train = n_text = 0
        # copy_parts=None: the whole track path is trainable.
        # copy_parts={'self_attn',...}: narrow selection — only the named per-layer
        # parts (self_attn/cross_attn/ffn/norms) + optionally rph/reg trainable on
        # the track path; the rest (image cross_attn, reg, ref_point_head)
        # stay fixed. cross_attn_text is kept as pretrained unless
        # release_text_ca lifts it.
        _cp = set(copy_parts) if copy_parts else None
        if _cp is None:
            for _m in (self.track_dec_layers, self.track_dec_rph,
                       self.track_dec_norm, self.track_dec_reg):
                for p in _m.parameters():
                    p.requires_grad_(True)
        else:
            _lyr_parts = {p for p in _cp
                          if p in ('self_attn', 'cross_attn', 'ffn', 'norms')}
            for _nm, p in self.track_dec_layers.named_parameters():
                # boundary match '.{part}.' so 'cross_attn' never catches
                # 'cross_attn_text' (kept as pretrained below regardless).
                p.requires_grad_(any(f'.{_pt}.' in f'.{_nm}.'
                                     for _pt in _lyr_parts))
            for p in self.track_dec_rph.parameters():
                p.requires_grad_('rph' in _cp)
            for p in self.track_dec_reg.parameters():
                p.requires_grad_('reg' in _cp)
            for p in self.track_dec_norm.parameters():
                p.requires_grad_('norm' in _cp or 'norms' in _cp)
        # Keep the TEXT cross-attn as pretrained in every track-path layer (the track's
        # class still reads the PRETRAINED text-conditioned cls head downstream).
        # release_text_ca=True lifts it, which is the ablation that measures what
        # that costs.
        for _lyr in self.track_dec_layers:
            _tc = getattr(_lyr, 'cross_attn_text', None)
            if _tc is not None:
                for p in _tc.parameters():
                    p.requires_grad_(bool(release_text_ca))
        _cnt = [self.track_dec_layers, self.track_dec_rph,
                self.track_dec_norm, self.track_dec_reg]
        n_train = sum(p.numel() for _m in _cnt for p in _m.parameters()
                      if p.requires_grad)
        n_text = sum(p.numel() for _lyr in self.track_dec_layers
                     for p in (getattr(_lyr, 'cross_attn_text', None)
                               or nn.Identity()).parameters())
        self.has_track_decoder_copy = True
        _pstr = ("ALL parts" if _cp is None
                 else f"NARROW parts={sorted(_cp)}")
        _sm = (f"{n}L decoupling stage + {_nl_total - n}L fusion stage "
               f"(det↔track) | track path trains: {_pstr}") \
            if n < _nl_total \
            else (f"no fusion stage: the track path spans all layers "
                  f"| track path trains: {_pstr}")
        _txt = ("RELEASED %.2fM — ABLATION only"
                if release_text_ca else "%.2fM kept as pretrained") % (n_text / 1e6)
        logging.getLogger().info(
            "  [decoupling-stage] track path built: %.2fM trainable "
            "(text-cross-attn %s); the detection path stays on self.layers; "
            "ref_point_head + reg + deformable-img-cross all TRAINABLE on the "
            "track path | %s", n_train / 1e6, _txt, _sm)
        return n_train

    def _run_track_copy(self, query0, ref0, value, key_padding_mask,
                        base_self_attn_mask, spatial_shapes, level_start_index,
                        valid_ratios, num_det_offset, num_track, num_det,
                        **kwargs):
        """Re-decode [DN|Det|Track] through the trainable track path of the
        decoupling stage and return the per-layer (normed track hidden, track
        reference). The track path's det/DN outputs are discarded (det uses the
        main decoder). NO det<->track isolation; NO ref detach (full-gradient
        relocation). Uses the BASE DN self_attn_mask (DN blocking) WITHOUT
        isolation."""
        # STRIP DN: the track path processes only [Det | Track]. DN queries are
        # detection-only and are dropped. self_attn_mask=None => full [Det|Track]
        # attention: track reads det, with no DN and no isolation.
        num_dn = num_det_offset - num_det
        q = query0[:, num_dn:].contiguous()
        ref = ref0[:, num_dn:].contiguous()
        _ndo = num_det          # track starts after Det in the stripped tensor
        track_h, track_ref = [], []
        for rlid, layer in enumerate(self.track_dec_layers):
            if ref.shape[-1] == 4:
                rpi = ref[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1)[:, None]
            else:
                rpi = ref[:, :, None] * valid_ratios[:, None]
            qse = coordinate_to_encoding(rpi[:, :, 0, :])
            qp = self.track_dec_rph(qse)
            q = layer(
                q, query_pos=qp, value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=None,            # full [Det|Track] attention
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=rpi,
                num_det_offset=_ndo,
                num_track=num_track, num_det=num_det, **kwargs)
            tmp = self.track_dec_reg[rlid](q)
            assert ref.shape[-1] == 4
            ref = (tmp + inverse_sigmoid(ref, eps=1e-3)).sigmoid()  # NO detach
            # INGRAIN_MEMFUSE_INTERLEAVE: after track-path layers 0/1/2,
            # refine the TRACK slice by attending to this track's per-depth
            # cross-frame history (persistent bank on trajectory_memory). Track-
            # only — det/DN slice q[:, :_ndo] is untouched (protects detection).
            # zero-init out_proj => no-op at start. Fires only when the caller
            # stashed _mf_interleave_oids before the decode, which both paths do:
            # the detector on the train path, and the evaluation runner on the
            # inference path (which defers the bank write until the text-chunk
            # loop is done, so a frame appends one history entry, not one per
            # chunk). The mechanism is active at inference. The refined q feeds the
            # NEXT layer AND the grafted track_h below (in-frame track loss trains
            # the ma directly).
            _mf_sc = getattr(self, '_mf_interleave_sc', None)
            _mf_oids = getattr(self, '_mf_interleave_oids', None)
            if (_mf_sc is not None and _mf_oids is not None and num_track > 0
                    and rlid < int(getattr(_mf_sc, 'interleave_depths', 0))
                    and q.shape[0] == 1
                    and int(_mf_oids.shape[0]) == num_track):
                _qt2 = _mf_sc.interleave_refine(rlid, q[0, _ndo:], _mf_oids)
                q = torch.cat([q[:, :_ndo], _qt2.unsqueeze(0)], dim=1)
                if not getattr(self, '_mf_il_fired', False):
                    self._mf_il_fired = True
                    print(f"[MDM] trajectory memory injected: track-path "
                          f"layer {rlid}, "
                          f"num_track={num_track} oids={int(_mf_oids.shape[0])}",
                          flush=True)
            track_h.append(self.track_dec_norm(q)[:, _ndo:])
            track_ref.append(ref[:, _ndo:])
        # Also return the RAW (un-normed, IN-GRAPH) final track hidden + ref so
        # forward() can inject them into the main [DN|Det|Track] tensor at the
        # stage boundary (gradient flows from the fusion stage back through the
        # track path). track_h[-1] is normed (readout only) → keep q raw.
        _raw_track_q = q[:, _ndo:]
        _raw_track_ref = ref[:, _ndo:]
        return track_h, track_ref, _raw_track_q, _raw_track_ref

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                num_det_offset: int = 0, num_track: int = 0,
                num_det: int = 0, cls_branches=None, **kwargs):
        """Forward function with track query boundary propagation.

        Same as DinoTransformerDecoder.forward() but passes num_det_offset
        and num_track as extra kwargs to each IngrainDecoderLayer.

        For the first `num_isolation_layers` layers, an isolation mask is
        applied that blocks det↔track cross-attention in self-attention,
        preventing track features from contaminating detect queries too early.

        Args:
            num_det_offset (int): Index where track queries start
                in the combined query tensor (= num_dn + num_det).
            num_track (int): Number of track queries.
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        # Keep an (in-graph, non-detached) handle on the initial [DN|Det|Track]
        # query/ref for the separate trainable track path (gradient flows into
        # the track path + back through the carried track content).
        # query/reference_points are reassigned to new tensors inside the loop, so
        # these handles stay pinned to the initial decoder input.
        _track_init_q = query if getattr(self, 'has_track_decoder_copy', False) \
            else None
        _track_init_ref = reference_points if getattr(
            self, 'has_track_decoder_copy', False) else None

        # Pre-build isolation mask (only when there are track queries)
        iso_mask = None
        if num_track > 0 and self.num_isolation_layers > 0:
            if getattr(self, 'iso_directional', False):
                # Directional isolation: det+DN still cannot read track (the
                # pollution direction stays blocked), but track can read det in the
                # isolated layers too.
                iso_mask = self._build_half_isolation_mask(
                    self_attn_mask, num_det_offset, num_track,
                    device=query.device)
            else:
                iso_mask = self._build_isolation_mask(
                    self_attn_mask, num_det_offset, num_track,
                    device=query.device)
        # Directional fusion mask: half-mask for the un-isolated fusion-stage
        # layers (det+DN can't read track; track->det stays open). Built when
        # iso_directional is set; with --decoupling-depth 0 the per-layer loop's
        # `elif half_mask is not None` branch applies it to all layers, so det+DN
        # never read track while track still reads det, and the track-ref
        # gradient (lid>=num_isolation_layers=0) still runs in every layer.
        half_mask = None
        if getattr(self, 'iso_directional', False) and num_track > 0:
            half_mask = self._build_half_isolation_mask(
                self_attn_mask, num_det_offset, num_track,
                device=query.device)
            if (getattr(self, 'iso_directional', False)
                    and self.num_isolation_layers == 0
                    and not getattr(self, '_logged_iso0_directional', False)):
                self._logged_iso0_directional = True
                _blk = bool(half_mask[:num_det_offset, num_det_offset:].all())
                print(f"[directional isolation] iso=0 half mask active "
                      f"on all {len(self.layers)} layers; det+DN->track blocked="
                      f"{_blk}; track->det open; num_track={num_track}", flush=True)

        # The track path spans only the first n=track_dec_n_layers layers (the
        # decoupling stage). Run it now (the detection path is untouched) and stash (a) the
        # per-layer normed track hidden/ref for the post-loop graft of layers
        # [0,n), and (b) the raw (in-graph) final track hidden/ref — injected
        # into the main [DN|Det|Track] tensor at the stage boundary so the
        # remaining layers process det+track jointly (the fusion stage).
        _sm_active = (getattr(self, 'has_track_decoder_copy', False)
                      and num_track > 0 and _track_init_q is not None
                      and getattr(self, 'track_dec_n_layers', None) is not None
                      and self.track_dec_n_layers < self.num_layers)
        _sm_th = _sm_tref = _sm_raw_q = _sm_raw_ref = None
        if _sm_active:
            _sm_th, _sm_tref, _sm_raw_q, _sm_raw_ref = self._run_track_copy(
                _track_init_q, _track_init_ref, value, key_padding_mask,
                self_attn_mask, spatial_shapes, level_start_index, valid_ratios,
                num_det_offset, num_track, num_det, **kwargs)

        for lid, layer in enumerate(self.layers):
            # At the stage boundary, inject the decoupling stage's track output
            # into the main tensor (det slice stays from the detection
            # path). From here self.layers[n:] (trainable) run det+track
            # jointly = the fusion stage.
            if _sm_active and lid == self.track_dec_n_layers:
                query = torch.cat(
                    [query[:, :num_det_offset], _sm_raw_q], dim=1)
                reference_points = torch.cat(
                    [reference_points[:, :num_det_offset], _sm_raw_ref], dim=1)
            # Isolation mask for the decoupling-stage layers, half/normal later.
            if iso_mask is not None and lid < self.num_isolation_layers:
                layer_attn_mask = iso_mask
            elif half_mask is not None:
                layer_attn_mask = half_mask
            else:
                layer_attn_mask = self_attn_mask
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=layer_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                # INGRAIN additions: track query boundary info
                num_det_offset=num_det_offset,
                num_track=num_track,
                num_det=num_det,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                # In the fusion-stage layers (lid >= num_isolation_layers,
                # where reg_branches are trainable), keep gradient on the TRACK
                # slice so loss_box_track trains the iterative relocation. Det/DN
                # slice + all early layers stay detached, so detection is
                # unperturbed. Track queries live at [num_det_offset:].
                if (getattr(self, 'train_track_ref_refine', False)
                        and num_track > 0
                        and lid >= self.num_isolation_layers):
                    _ref_det = new_reference_points[:, :num_det_offset].detach()
                    _ref_trk = new_reference_points[:, num_det_offset:]
                    reference_points = torch.cat([_ref_det, _ref_trk], dim=1)
                else:
                    reference_points = new_reference_points.detach()

            # INGRAIN_MEMFUSE_INTERLEAVE, fusion-stage injection point. The
            # per-depth memory inside _run_track_copy covers the n
            # decoupling-stage layers, i.e. depths d = 1..n. This adds the next
            # one, after the FIRST fusion-stage layer (d = n+1), so the depth
            # sweep can go past the decoupling stage. Depth index n reuses the
            # same numbering the decoupling stage uses (0..n-1), so it needs
            # interleave_depths > n. Track slice only; det/DN rows are untouched.
            # Same ordering as in the decoupling stage: layer -> reg -> memory
            # refine -> next layer.
            _mf_sc_m = getattr(self, '_mf_interleave_sc', None)
            _mf_oids_m = getattr(self, '_mf_interleave_oids', None)
            _mf_k = getattr(self, 'track_dec_n_layers', None)
            if (_sm_active and _mf_sc_m is not None and _mf_oids_m is not None
                    and _mf_k is not None and lid == _mf_k and num_track > 0
                    and _mf_k < int(getattr(_mf_sc_m, 'interleave_depths', 0))
                    and query.shape[0] == 1
                    and int(_mf_oids_m.shape[0]) == num_track):
                _qm = _mf_sc_m.interleave_refine(
                    _mf_k, query[0, num_det_offset:], _mf_oids_m)
                query = torch.cat(
                    [query[:, :num_det_offset], _qm.unsqueeze(0)], dim=1)
                if not getattr(self, '_mf_fusion_fired', False):
                    self._mf_fusion_fired = True
                    print(f"[MDM] trajectory memory injected: fusion-stage "
                          f"depth d={_mf_k+1} "
                          f"fired at layer {lid} num_track={num_track}",
                          flush=True)

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)

        # Graft the track slice from the separate trainable track path.
        # det/DN slices stay from the detection path above (preserves det+OV);
        # only the track slice [num_det_offset:] is replaced by the track path's
        # re-decode (track reads det, trainable ref_point_head + deformable + reg, full
        # gradient). Done after any anchor/align re-runs so those see the clean
        # main-path hidden.
        if (getattr(self, 'has_track_decoder_copy', False) and num_track > 0
                and _track_init_q is not None
                and len(intermediate) == self.num_layers):
            if _sm_active:
                # Graft only the decoupling-stage layers [0, n) from the track
                # path that already ran pre-loop. The fusion-stage layers [n,
                # num_layers) already carry the joint det↔track output (the track
                # hidden was injected into the main tensor at the stage boundary).
                _t_h, _t_ref = _sm_th, _sm_tref
                _graft_n = self.track_dec_n_layers
            else:
                # No fusion stage: the track path spans every layer — run it now
                # and graft all of them.
                _t_h, _t_ref, _, _ = self._run_track_copy(
                    _track_init_q, _track_init_ref, value, key_padding_mask,
                    self_attn_mask, spatial_shapes, level_start_index,
                    valid_ratios, num_det_offset, num_track, num_det, **kwargs)
                _graft_n = self.num_layers
            for _i in range(_graft_n):
                intermediate[_i] = torch.cat(
                    [intermediate[_i][:, :num_det_offset], _t_h[_i]], dim=1)
                intermediate_reference_points[_i + 1] = torch.cat(
                    [intermediate_reference_points[_i + 1][:, :num_det_offset],
                     _t_ref[_i]], dim=1)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points
