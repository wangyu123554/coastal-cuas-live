"""Frame batching and local meters. Kept here so the allocator does not import the rest of the old policy."""

from __future__ import annotations

import math

import torch

from network.input_v10 import ENTITY_KIND, N_OWN_CAP

TRACK_KIND = ENTITY_KIND.index("track")


def _batch(frame, hide_tracks: bool) -> dict[str, torch.Tensor]:
    valid = [False if hide_tracks and kind and kind[0] == TRACK_KIND else flag for flag, kind in zip(frame.entity_valid, frame.category)]
    own = (frame.own_index + [-1] * N_OWN_CAP)[:N_OWN_CAP]
    unit = (frame.unit_valid + [False] * N_OWN_CAP)[:N_OWN_CAP]
    return {
        "cont": torch.tensor(frame.cont, dtype=torch.float32).unsqueeze(0),
        "feature_state": torch.tensor(frame.feature_state, dtype=torch.long).unsqueeze(0),
        "category": torch.tensor(frame.category, dtype=torch.long).unsqueeze(0),
        "entity_valid": torch.tensor(valid, dtype=torch.bool).unsqueeze(0),
        "part_parent": torch.tensor(frame.part_parent, dtype=torch.long).unsqueeze(0),
        "own_index": torch.tensor(own, dtype=torch.long).unsqueeze(0),
        "unit_valid": torch.tensor(unit, dtype=torch.bool).unsqueeze(0),
    }


def _meters(lat0: float, lon0: float, lat1: float, lon1: float) -> tuple[float, float, float]:
    north = (lat1 - lat0) * 111320.0
    east = (lon1 - lon0) * 111320.0 * math.cos(math.radians((lat0 + lat1) / 2.0))
    return north, east, math.hypot(north, east)
