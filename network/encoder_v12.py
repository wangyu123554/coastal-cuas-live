"""Scaled experiment encoder. Does not replace l1.encoder/1.1.

The 38 fields, four states, and categories stay as they are. Before the value
projection, each known number is divided by a fixed physical unit. The divisors
are constants, not statistics of a run, so a frame does not depend on later
rows. Unknown, missing, and not-applicable values stay out of the projection.
A known zero stays zero.
"""

from __future__ import annotations

import torch

from network.encoder_v11 import GlobalEncoderV11
from network.input_v11 import FIELDS

SCHEMA = "l1.encoder/1.2-scale"
PARENT_SCHEMA = "l1.encoder/1.1"

# One stored unit per field. The network sees value / unit.
FIELD_UNIT: dict[str, float] = {
    "east_m": 1000.0,
    "north_m": 1000.0,
    "up_m": 1000.0,
    "dest_east_m": 1000.0,
    "dest_north_m": 1000.0,
    "dest_up_m": 1000.0,
    "range_m": 1000.0,
    "max_range_m": 1000.0,
    "range_sigma": 1000.0,
    "delta_east_m": 100.0,
    "delta_north_m": 100.0,
    "delta_up_m": 100.0,
    "v_north_mps": 10.0,
    "v_east_mps": 10.0,
    "v_up_mps": 10.0,
    "yaw_deg": 180.0,
    "pitch_deg": 180.0,
    "fov_az_min_deg": 180.0,
    "fov_az_max_deg": 180.0,
    "fov_el_min_deg": 180.0,
    "fov_el_max_deg": 180.0,
    "bearing_deg": 180.0,
    "elevation_deg": 180.0,
    "bearing_sigma": 180.0,
    "elevation_sigma": 180.0,
    "scan_az_min_deg": 180.0,
    "scan_az_max_deg": 180.0,
    "scan_el_min_deg": 180.0,
    "scan_el_max_deg": 180.0,
    "state_age_s": 1.0,
    "observe_age_s": 1.0,
    "update_gap_s": 1.0,
    "track_life_s": 10.0,
    "update_count": 10.0,
    "quantity_remaining": 10.0,
    "snr": 10.0,
    "sensor_on": 1.0,
    "part_on": 1.0,
}


def _units() -> torch.Tensor:
    missing = [name for name in FIELDS if name not in FIELD_UNIT]
    if missing:
        raise RuntimeError(f"scale missing fields: {missing}")
    return torch.tensor([FIELD_UNIT[name] for name in FIELDS], dtype=torch.float32)


class ScaledGlobalEncoder(GlobalEncoderV11):
    """Same 4-layer encoder as 1.1, with fixed field units before value_proj."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("field_unit", _units(), persistent=False)

    def pre_attention(
        self,
        cont: torch.Tensor,
        feature_state: torch.Tensor,
        category: torch.Tensor,
        part_parent: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        unit = self.field_unit.to(device=cont.device, dtype=cont.dtype)
        return super().pre_attention(cont / unit, feature_state, category, part_parent, history)
