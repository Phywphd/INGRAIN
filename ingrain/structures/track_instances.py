"""
TrackInstances: manages per-frame track query state for INGRAIN.

All tensor fields are stored in a plain dict and accessed
transparently via __getattr__/__setattr__.

Supported fields (all optional — set as needed):
  ref_pts           (N, 4)           inverse_sigmoid reference points
  query_tgt         (N, embed_dims)  content embedding
  obj_idxes         (N,)  long       track ID  (-1=unassigned, >=0=active)
  matched_gt_idxes  (N,)  long
  scores            (N,)  float
  pred_boxes        (N, 4)
  pred_logits       (N, max_text_len)
  output_embedding  (N, embed_dims)  decoder output
  disappear_time    (N,)  long       inference only
  query_pos         (N, embed_dims)  positional embedding (optional)
"""

import copy
from typing import Any, Dict, List, Union

import torch
from torch import Tensor


# Names that are NOT track fields — they live directly on the object
_RESERVED = frozenset({"_fields", "_num_instances"})


class TrackInstances:
    """Container for a set of track query instances.

    Fields are stored in ``self._fields`` (a plain dict).  Any attribute
    assignment whose name is not in ``_RESERVED`` is transparently routed
    into ``_fields``.

    Example::

        ti = TrackInstances()
        ti.obj_idxes  = torch.full((N,), -1, dtype=torch.long)
        ti.query_tgt  = torch.zeros(N, 256)
        ti.ref_pts    = torch.zeros(N, 4)

        active = ti[ti.obj_idxes >= 0]   # slices all fields at once
        merged = TrackInstances.cat([ti_a, ti_b])
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Use object.__setattr__ to bypass our custom __setattr__
        object.__setattr__(self, "_fields", {})
        object.__setattr__(self, "_num_instances", None)

    # ------------------------------------------------------------------
    # Attribute routing
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RESERVED:
            object.__setattr__(self, name, value)
        else:
            # Validate tensor shape consistency when we can determine N
            if isinstance(value, Tensor):
                n = self._num_instances
                if n is None:
                    # First tensor sets the length
                    object.__setattr__(self, "_num_instances", value.shape[0])
                else:
                    if value.shape[0] != n:
                        raise ValueError(
                            f"Field '{name}' has {value.shape[0]} instances, "
                            f"expected {n}."
                        )
            self._fields[name] = value

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup failed
        fields = object.__getattribute__(self, "_fields")
        if name in fields:
            return fields[name]
        raise AttributeError(
            f"'TrackInstances' object has no field '{name}'"
        )

    def __delattr__(self, name: str) -> None:
        if name in _RESERVED:
            object.__delattr__(self, name)
        elif name in self._fields:
            del self._fields[name]
            # Recompute _num_instances
            tensors = [v for v in self._fields.values() if isinstance(v, Tensor)]
            object.__setattr__(
                self,
                "_num_instances",
                tensors[0].shape[0] if tensors else None,
            )
        else:
            raise AttributeError(f"'TrackInstances' has no field '{name}'")

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        n = self._num_instances
        if n is None:
            return 0
        return n

    def __repr__(self) -> str:
        field_str = ", ".join(
            f"{k}: {v.shape}" if isinstance(v, Tensor) else f"{k}: {type(v)}"
            for k, v in self._fields.items()
        )
        return f"TrackInstances(N={len(self)}, fields=[{field_str}])"

    # ------------------------------------------------------------------
    # Indexing  (boolean mask OR integer indices)
    # ------------------------------------------------------------------

    def __getitem__(self, idx: Union[Tensor, slice, list]) -> "TrackInstances":
        new = TrackInstances()
        for k, v in self._fields.items():
            if isinstance(v, Tensor):
                new._fields[k] = v[idx]
            else:
                new._fields[k] = v  # non-tensor fields copied as-is
        # Recompute _num_instances from the sliced tensors
        tensors = [v for v in new._fields.values() if isinstance(v, Tensor)]
        object.__setattr__(
            new,
            "_num_instances",
            tensors[0].shape[0] if tensors else None,
        )
        return new

    # ------------------------------------------------------------------
    # Field helpers
    # ------------------------------------------------------------------

    def has(self, field_name: str) -> bool:
        """Return True if *field_name* has been set."""
        return field_name in self._fields

    def get(self, field_name: str, default: Any = None) -> Any:
        """Return the field value, or *default* if it does not exist."""
        return self._fields.get(field_name, default)

    def fields(self) -> Dict[str, Any]:
        """Return a copy of the internal field dict."""
        return dict(self._fields)

    # ------------------------------------------------------------------
    # Device / clone
    # ------------------------------------------------------------------

    def to(self, device: Union[str, torch.device]) -> "TrackInstances":
        """Move all tensor fields to *device* and return self (in-place)."""
        for k, v in self._fields.items():
            if isinstance(v, Tensor):
                self._fields[k] = v.to(device)
        return self

    def clone(self) -> "TrackInstances":
        """Return a deep copy with all tensors cloned."""
        new = TrackInstances()
        for k, v in self._fields.items():
            new._fields[k] = v.clone() if isinstance(v, Tensor) else copy.deepcopy(v)
        object.__setattr__(new, "_num_instances", self._num_instances)
        return new

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    @staticmethod
    def cat(instances_list: List["TrackInstances"]) -> "TrackInstances":
        """Concatenate a list of TrackInstances along dim=0.

        All instances must share the same set of field names.
        Non-tensor fields are taken from the *first* instance.
        """
        if not instances_list:
            return TrackInstances()

        # Collect all field names (union)
        all_keys: set = set()
        for inst in instances_list:
            all_keys.update(inst._fields.keys())

        new = TrackInstances()
        for k in all_keys:
            parts = []
            non_tensor_val = None
            for inst in instances_list:
                v = inst._fields.get(k, None)
                if v is None:
                    raise KeyError(
                        f"Field '{k}' is missing from at least one "
                        "TrackInstances in the list."
                    )
                if isinstance(v, Tensor):
                    parts.append(v)
                else:
                    non_tensor_val = v

            if parts:
                new._fields[k] = torch.cat(parts, dim=0)
            else:
                new._fields[k] = non_tensor_val

        tensors = [v for v in new._fields.values() if isinstance(v, Tensor)]
        object.__setattr__(
            new,
            "_num_instances",
            tensors[0].shape[0] if tensors else None,
        )
        return new
