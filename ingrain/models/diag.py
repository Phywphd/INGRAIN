"""INGRAIN training diagnostics.

The total loss cannot attribute progress to a particular module, so these
diagnostics target the modules directly: the gradient norms show the signal
reaching each trainable group, and the box/coverage metrics show how much of
each frame the track queries are carrying.

Enabled by --diag-log, which emits one compact line every N optimizer steps.
All functions are torch-only, @no_grad, and empty-safe (return NaN so the
accumulator can drop them and the line prints '-').
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def module_grad_norm(module) -> float:
    """Total L2 grad norm over a module's params."""
    if module is None:
        return float('nan')
    tot = 0.0
    seen = False
    for p in module.parameters():
        if p.grad is not None:
            tot += float(p.grad.detach().float().pow(2).sum().item())
            seen = True
    return math.sqrt(tot) if seen else 0.0


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class DiagAccumulator:
    """Accumulate per-step diagnostics; emit one compact multi-group line.

    add(key, val): running mean of a scalar (NaN values are ignored).
    extend(key, vals): collect a list for percentile readout (e.g. track IoUs).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum: Dict[str, float] = {}
        self._cnt: Dict[str, int] = {}
        self._lists: Dict[str, List[float]] = {}

    def add(self, key: str, val, n: int = 1):
        if val is None:
            return
        v = float(val)
        if math.isnan(v):
            return
        self._sum[key] = self._sum.get(key, 0.0) + v * n
        self._cnt[key] = self._cnt.get(key, 0) + n

    def extend(self, key: str, vals):
        if not vals:
            return
        self._lists.setdefault(key, []).extend(
            float(v) for v in vals if not math.isnan(float(v)))

    def mean(self, key: str) -> float:
        c = self._cnt.get(key, 0)
        return self._sum[key] / c if c > 0 else float('nan')

    def pct(self, key: str, p: float) -> float:
        vals = self._lists.get(key, [])
        if not vals:
            return float('nan')
        t = torch.tensor(vals)
        return float(torch.quantile(t, p).item())

    def format_line(self, step: int) -> str:
        """Compact four-group diagnostic line. Missing entries show as '-'."""
        def f(key, fmt="{:.3f}"):
            v = self.mean(key)
            return fmt.format(v) if not math.isnan(v) else "-"

        def fp(key, p, fmt="{:.3f}"):
            v = self.pct(key, p)
            return fmt.format(v) if not math.isnan(v) else "-"

        # Track-query localization: how well the propagated boxes sit on their
        # assigned GT. The p10 tail is the informative half — a healthy median
        # with a collapsed tail means a subset of tracks has drifted off.
        track = (f"track_iou(med/p10)={fp('qtr_iou',0.5)}/{fp('qtr_iou',0.1)}")
        # recover_missed: GT covered by a track but by NO detection (what the
        # track path adds over the detector). dup_fire: GT claimed by both.
        coverage = (f"recover_missed={f('recover_missed','{:.2f}')} "
                    f"dup_det={f('dup_fire','{:.2f}')}")
        # det_score tracks the classification score of matched detections.
        counts = (f"alive/newborn={f('n_alive','{:.1f}')}/"
                  f"{f('n_newborn','{:.1f}')} det_score={f('det_score')}")
        # The signal reaching each trainable group, which is what these
        # diagnostics exist to show.
        grads = (f"grad_reg={f('grad_reg','{:.2e}')} "
                 f"grad_track_reg={f('grad_copyreg','{:.2e}')} "
                 f"grad_dec_self_attn={f('grad_dec_sa','{:.2e}')}")
        return (
            f"[DIAG step {step}]\n"
            f"  1) track box: {track}\n"
            f"  2) coverage : {coverage}\n"
            f"  3) counts   : {counts}\n"
            f"  4) gradnorm : {grads}"
        )
