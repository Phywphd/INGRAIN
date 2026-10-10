"""IngrainDecoderLayer: the GD decoder layer with a track-query adapter.

Identical to the Grounding DINO decoder layer except that, when the layer is
built with the adapter, the track slice of the query passes through a zero-init
residual bottleneck. With no track queries the layer is equivalent to the
original.
"""
import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'third_party', 'mmgroundingdino'))
from mmdet.models.layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoderLayer)


@MODELS.register_module()
class IngrainDecoderLayer(GroundingDinoTransformerDecoderLayer):
    """GD decoder layer with the track-query adapter."""

    def _init_layers(self) -> None:
        """Initialize the parent layers plus the track-query adapter."""
        super()._init_layers()
        # Track-query adapter (env INGRAIN_TRACK_SAMPLING_ADAPTER=1): a
        # zero-init residual bottleneck on the track slice. Identity at
        # initialisation; the det/DN slice is untouched.
        self.track_sampling_adapter = None
        self.tsa_scale = 0.1
        if os.environ.get('INGRAIN_TRACK_SAMPLING_ADAPTER') == '1':
            _r = int(os.environ.get('INGRAIN_TSA_DIM', '128'))
            _up = nn.Linear(self.embed_dims, _r)
            _down = nn.Linear(_r, self.embed_dims)
            nn.init.zeros_(_down.weight)
            nn.init.zeros_(_down.bias)
            self.track_sampling_adapter = nn.Sequential(
                nn.LayerNorm(self.embed_dims), _up, nn.GELU(), _down)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Forward pass.

        Extra kwargs:
            num_det_offset (int): Index where track queries start
                (= num_dn + num_det). Default: 0.
            num_track (int): Number of track queries. Default: 0.
        """
        num_det_offset = kwargs.pop('num_det_offset', 0)
        num_track = kwargs.pop('num_track', 0)
        kwargs.pop('num_det', 0)
        # Steps 1-4: Original GD decoder layer
        # (self_attn → text_cross_attn → img_cross_attn → FFN)
        if self.track_sampling_adapter is None:
            # Default path: monolithic parent forward, byte-identical to GD.
            query = super().forward(
                query=query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                **kwargs)
        else:
            # Adapter path: the parent's four steps are unrolled verbatim (a copy
            # of GroundingDinoTransformerDecoderLayer.forward) so the track-slice
            # adapter can be spliced in. Only taken when the adapter exists, so
            # every other recipe keeps the monolithic parent call above.
            query = self.self_attn(
                query=query, key=query, value=query,
                query_pos=query_pos, key_pos=query_pos,
                attn_mask=self_attn_mask, **kwargs)
            query = self.norms[0](query)
            query = self.cross_attn_text(
                query=query, query_pos=query_pos,
                key=memory_text, value=memory_text,
                key_padding_mask=text_attention_mask)
            query = self.norms[1](query)
            # Track slice only; the det/DN slice is byte-identical to GD.
            if num_track > 0 and num_det_offset > 0:
                _tsl = query[:, num_det_offset:]
                query = torch.cat(
                    [query[:, :num_det_offset],
                     _tsl + self.tsa_scale * self.track_sampling_adapter(_tsl)],
                    dim=1)
            query = self.cross_attn(
                query=query, key=key, value=value,
                query_pos=query_pos, key_pos=key_pos,
                attn_mask=cross_attn_mask, key_padding_mask=key_padding_mask,
                **kwargs)
            query = self.norms[2](query)
            query = self.ffn(query)
            query = self.norms[3](query)

        return query
