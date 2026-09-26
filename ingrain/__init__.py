"""INGRAIN: Tracking with INherited GRounding Alignment IN the Native
Vision-Language Query Space.

A multi-object tracker built on Grounding DINO. Import this package to
register all INGRAIN modules with the mmdet registry.
"""
# Register all modules with mmdet registry
from ingrain.models import (INGRAINTracker, TrajectoryMemory,
                            IngrainDecoderLayer, IngrainTransformerDecoder)
from ingrain.losses import INGRAINTrackCriterion

__all__ = [
    'INGRAINTracker', 'TrajectoryMemory',
    'IngrainDecoderLayer', 'IngrainTransformerDecoder',
    'INGRAINTrackCriterion',
]
