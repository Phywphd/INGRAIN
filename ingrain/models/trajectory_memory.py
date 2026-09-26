"""INGRAIN trajectory state propagation modules.

- identity/memory state: per-track trajectory memory + memory attention
  (seed-space single fusion + per-depth interleaved fusion).
"""

from collections import defaultdict

import torch
import torch.nn as nn
from mmdet.registry import MODELS


class TrajectoryMemoryBank:
    """Non-parametric per-track FIFO queue of detached hidden states."""

    def __init__(self, max_history: int = 16, embed_dims: int = 256):
        self.max_history = int(max_history)
        self.embed_dims = int(embed_dims)
        self.memory: dict[int, list[torch.Tensor]] = defaultdict(list)

    def update(self, track_ids: torch.Tensor, embeddings: torch.Tensor) -> None:
        for i, tid in enumerate(track_ids):
            tid_int = int(tid.item())
            if tid_int < 0:
                continue
            self.memory[tid_int].append(embeddings[i].detach())
            if len(self.memory[tid_int]) > self.max_history:
                self.memory[tid_int].pop(0)

    def get(self, track_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = track_ids.device
        trajectories = []
        for tid in track_ids:
            tid_int = int(tid.item())
            if tid_int in self.memory and self.memory[tid_int]:
                traj = torch.stack(self.memory[tid_int], dim=0).to(device)
            else:
                traj = torch.zeros(1, self.embed_dims, device=device)
            trajectories.append(traj)
        if not trajectories:
            empty = torch.zeros(0, 0, self.embed_dims, device=device)
            mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
            return empty, mask
        max_len = max(t.shape[0] for t in trajectories)
        padded = torch.zeros(
            len(trajectories), max_len, self.embed_dims, device=device)
        mask = torch.zeros(
            len(trajectories), max_len, dtype=torch.bool, device=device)
        for i, traj in enumerate(trajectories):
            padded[i, :traj.shape[0]] = traj
            mask[i, :traj.shape[0]] = True
        return padded, mask

    def evict(self, track_ids) -> int:
        """Drop the stored history of the given tracks; returns how many were
        actually held. Called when a track dies, so its memory does not
        outlive it."""
        n = 0
        for tid in track_ids:
            if self.memory.pop(int(tid), None) is not None:
                n += 1
        return n

    def reset(self) -> None:
        self.memory.clear()


class TrajectoryMemoryAttention(nn.Module):
    """Each track query attends to its own trajectory tokens.

    A learnable token is always prepended to every track's memory, so a
    track whose bank is still empty has one valid key to attend to.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 residual_scale: float = 0.1,
                 mode: str = 'attention',
                 ema_alpha: float = 0.9):
        super().__init__()
        # Aggregation mode. The three modes differ ONLY in
        # how the K stored states are collapsed into one context vector;
        # normalisation, residual scale, FFN and the zero-init exit are shared,
        # so the comparison isolates the aggregator.
        #   attention — query-conditioned multi-head cross-attention
        #   mean      — masked mean over the valid history
        #   ema       — exponentially recency-weighted mean, weight alpha^age
        #               (the bank stores oldest-first, so age counts back from
        #               the last valid slot)
        if mode not in ('attention', 'mean', 'ema'):
            raise ValueError(f"memory aggregation mode must be one of "
                             f"'attention' | 'mean' | 'ema', got {mode!r}")
        self.mode = mode
        self.ema_alpha = float(ema_alpha)
        self.residual_scale = float(residual_scale)
        self.q_norm = nn.LayerNorm(embed_dims)
        self.mem_norm = nn.LayerNorm(embed_dims)
        self.bos_token = nn.Parameter(torch.zeros(embed_dims))
        nn.init.normal_(self.bos_token, std=0.02)
        self.attn = None
        self.agg_out = None
        if self.mode == 'attention':
            self.attn = nn.MultiheadAttention(
                embed_dims, num_heads, dropout=dropout, batch_first=True)
            nn.init.zeros_(self.attn.out_proj.weight)
            nn.init.zeros_(self.attn.out_proj.bias)
        else:
            # Parameter-free aggregation still needs a zero-init exit so that
            # all three modes start as the exact identity — otherwise the
            # ablation would also be comparing different initial states.
            self.agg_out = nn.Linear(embed_dims, embed_dims)
            nn.init.zeros_(self.agg_out.weight)
            nn.init.zeros_(self.agg_out.bias)
        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, embed_dims * 4),
            nn.GELU(),
            nn.Linear(embed_dims * 4, embed_dims),
        )
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def _aggregate(self, mem, memory_mask, q):
        """Collapse the per-track memory into one context vector per track."""
        if self.mode == 'attention':
            ctx, _ = self.attn(
                q, mem, mem,
                key_padding_mask=~memory_mask.bool(),
                need_weights=False,
            )
            return ctx.squeeze(1)
        w = memory_mask.to(mem.dtype)                       # (N, T)
        if self.mode == 'ema':
            # age 0 = newest valid slot. The bank appends, so the last valid
            # index is the most recent observation.
            idx = torch.arange(w.shape[1], device=mem.device).view(1, -1)
            last = memory_mask.to(torch.long).cumsum(dim=-1).argmax(
                dim=-1, keepdim=True)                       # (N, 1)
            age = (last - idx).clamp(min=0).to(mem.dtype)
            w = w * (self.ema_alpha ** age)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        ctx = (mem * w.unsqueeze(-1)).sum(dim=1)            # (N, C)
        return self.agg_out(ctx)

    def forward(self,
                q_tr: torch.Tensor,
                memory_tokens: torch.Tensor,
                memory_mask: torch.Tensor) -> torch.Tensor:
        if q_tr.numel() == 0:
            return q_tr
        N = q_tr.shape[0]
        # Prepend a BOS token + always-valid mask entry to each track's memory.
        bos = self.bos_token.view(1, 1, -1).expand(N, 1, -1)
        if memory_tokens.numel() == 0:
            memory_tokens = bos
            memory_mask = memory_mask.new_ones(N, 1, dtype=torch.bool)
        else:
            memory_tokens = torch.cat([bos, memory_tokens], dim=1)
            bos_mask = memory_mask.new_ones(N, 1, dtype=torch.bool)
            memory_mask = torch.cat([bos_mask, memory_mask], dim=1)

        q = self.q_norm(q_tr).unsqueeze(1)
        mem = self.mem_norm(memory_tokens)
        ctx = self._aggregate(mem, memory_mask, q)
        updated = q_tr + self.residual_scale * ctx
        updated = updated + self.residual_scale * self.ffn(updated)
        return updated


@MODELS.register_module()
class TrajectoryMemory(nn.Module):
    """INGRAIN trajectory state propagation controller."""

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        max_history: int = 16,
        ema_alpha: float = 0.9,
        enable_memory_attention: bool = False,
        memory_attention_num_heads: int = 8,
        memory_attention_dropout: float = 0.0,
        memory_attention_residual_scale: float = 0.1,
        memory_aggregation: str = 'attention',
        interleave_depths: int = 3,
    ):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.ema_alpha = float(ema_alpha)
        self.memory_aggregation = str(memory_aggregation)

        self.seed_mem_bank = TrajectoryMemoryBank(max_history, embed_dims)

        self.enable_memory_attention = bool(enable_memory_attention)
        self.seed_mem_attn = (
            TrajectoryMemoryAttention(
                embed_dims=embed_dims,
                num_heads=memory_attention_num_heads,
                dropout=memory_attention_dropout,
                residual_scale=memory_attention_residual_scale,
                mode=self.memory_aggregation,
                ema_alpha=ema_alpha,
            )
            if self.enable_memory_attention else None)

        # ── Interleaved memory attention (INGRAIN_MEMFUSE_INTERLEAVE) ─────────
        # Multi-depth memory attention: 3 independent ma modules + 3 persistent
        # per-depth banks, applied after decoupling-stage track layers 0/1/2
        # (complementing the single seed-only fusion of
        # INGRAIN_ENCROI_MEM_FUSE).
        # Each depth keeps its OWN feature space (layer-d hidden) so the query
        # and its stored history match — depth d's bank holds only depth-d
        # hiddens. Created unconditionally (zero-init out_proj => exact no-op;
        # never called unless the env flag is on AND the detector stashes track
        # oids before the decode), so outputs are identical with the flag off
        # and checkpoints stay compatible either way. Banks are plain state
        # (not submodules), reset per-clip alongside memory_bank.
        # interleave_depths is the number of TRACK-PATH injection points,
        # i.e. depths d = 1..interleave_depths; the seed is d = 0. So
        # interleave_depths=3 gives d = {0,1,2,3} (the full model) and 0 gives
        # d = {0}. Configurable via --memory-depths.
        self.interleave_depths = int(interleave_depths)
        self.depth_mem_attn = nn.ModuleList([
            TrajectoryMemoryAttention(
                embed_dims=embed_dims,
                num_heads=memory_attention_num_heads,
                dropout=memory_attention_dropout,
                residual_scale=memory_attention_residual_scale,
                mode=self.memory_aggregation,
                ema_alpha=ema_alpha,
            ) for _ in range(self.interleave_depths)])
        self.depth_mem_bank = [
            TrajectoryMemoryBank(max_history, embed_dims)
            for _ in range(self.interleave_depths)]

        # Bank-WRITE policy for interleave_refine. The bank holds states from
        # PREVIOUS observations only, so exactly one state per observation may
        # be stored.
        #   'immediate' — training: one decode per observation, so writing as
        #                 soon as a depth is refined already means once per
        #                 observation.
        #   'defer'     — chunked inference: the SAME observation is decoded
        #                 once per text chunk (31 chunks over 1203 classes at
        #                 the chunk size set in eval/config.py). Writing per call would store 31
        #                 states per observation and fill K=16 with views of the
        #                 CURRENT frame, leaving no cross-frame history at all.
        #                 Deferring records the first chunk's state and holds it
        #                 until commit_interleave_writes(), so every chunk reads
        #                 the same pre-observation history and one state lands.
        self.interleave_write_mode = 'immediate'
        self._interleave_pending: dict = {}

        self._last_memory_attention_stats: dict = {}

    def interleave_refine(self,
                          depth_idx: int,
                          q_track: torch.Tensor,
                          obj_idxes: torch.Tensor) -> torch.Tensor:
        """Multi-depth memory attention. Refine the track hidden at
        decoder depth ``depth_idx`` by attending to this track's per-depth
        cross-frame history, then store this frame's hidden as next-frame
        history. ``q_track`` (N, C) = track rows at this depth; ``obj_idxes``
        (N,) = offset-applied ids (same namespace as memory_bank). Returns the
        refined (N, C). zero-init out_proj => identity at start; history is
        detached (no BPTT through the bank, mirrors memory_bank)."""
        if (q_track.numel() == 0 or obj_idxes is None
                or depth_idx >= self.interleave_depths):
            return q_track
        ma = self.depth_mem_attn[depth_idx]
        bank = self.depth_mem_bank[depth_idx]
        traj, mask = bank.get(obj_idxes)
        out = ma(q_track, traj, mask)
        if self.interleave_write_mode == 'defer':
            # First decode of this observation wins; the later text chunks
            # re-read the identical history and contribute no second state.
            self._interleave_pending.setdefault(
                depth_idx, (obj_idxes.clone(), q_track.detach().clone()))
        else:
            bank.update(obj_idxes, q_track.detach())
        return out

    def commit_interleave_writes(self, store: bool = True) -> int:
        """Flush the per-depth writes buffered under ``'defer'``.

        Call once per observation, after every text chunk has been decoded.
        ``store=False`` discards the buffer instead, which is what a re-decode
        of the same frame needs: reading the memory is idempotent, advancing it
        is not. Returns the number of depths written. No-op under
        ``'immediate'``, where the buffer is always empty."""
        n = 0
        if store:
            for _d, (_oids, _q) in self._interleave_pending.items():
                self.depth_mem_bank[_d].update(_oids, _q)
                n += 1
        self._interleave_pending = {}
        return n

    def evict_tracks(self, track_ids) -> int:
        """Remove dead tracks from every memory bank (seed + per-depth), so a
        track's history is released with the track itself rather than
        persisting until the end of the video."""
        ids = list(track_ids)
        if not ids:
            return 0
        n = self.seed_mem_bank.evict(ids)
        for _b in getattr(self, 'depth_mem_bank', []):
            n += _b.evict(ids)
        return n

    def reset(self) -> None:
        self.seed_mem_bank.reset()
        for _b in getattr(self, 'depth_mem_bank', []):
            _b.reset()
        self._interleave_pending = {}
        # reset() is called at the start of every forward_train_mot clip.
        self._last_memory_attention_stats = None

    def update_memory(self, track_instances, frame_idx: int) -> None:
        active_mask = track_instances.obj_idxes >= 0
        if not active_mask.any():
            return
        self.seed_mem_bank.update(
            track_instances.obj_idxes[active_mask],
            track_instances.output_embedding[active_mask],
        )

    def _trajectory_tokens(self,
                           q_tr: torch.Tensor,
                           active_obj_idxes: torch.Tensor):
        if not isinstance(active_obj_idxes, torch.Tensor):
            active_obj_idxes = torch.tensor(
                list(active_obj_idxes), dtype=torch.long, device=q_tr.device)
        else:
            active_obj_idxes = active_obj_idxes.to(device=q_tr.device)
        return self.seed_mem_bank.get(active_obj_idxes)

    def propagate_memory_state(self,
                               q_tr: torch.Tensor,
                               active_obj_idxes: torch.Tensor) -> torch.Tensor:
        if self.seed_mem_attn is None or q_tr.numel() == 0:
            return q_tr
        if active_obj_idxes is None or len(active_obj_idxes) == 0:
            return q_tr

        traj_tokens, traj_mask = self._trajectory_tokens(q_tr, active_obj_idxes)
        # Cold start: the BOS token in TrajectoryMemoryAttention guarantees
        # every track at least one valid memory token, so an empty persistent
        # bank (frame-0 of a clip) needs no early return here.

        q_out = self.seed_mem_attn(q_tr, traj_tokens, traj_mask)
        with torch.no_grad():
            delta = (q_out - q_tr).norm(dim=-1)
            base = q_tr.norm(dim=-1).clamp(min=1e-6)
            self._last_memory_attention_stats = {
                'n': int(q_tr.shape[0]),
                'delta_mean': float(delta.mean().item()),
                'delta_max': float(delta.max().item()),
                'rel_delta_mean': float((delta / base).mean().item()),
                'memory_bank_filled': int((traj_mask.sum(dim=-1) > 1).sum().item()),
                'mean_mem_tokens_per_track': float(
                    traj_mask.float().sum(dim=-1).mean().item()),
            }
        return q_out
