"""l1.input/1.1. Adds functional parts without changing the 32 columns of l1.input/1.0.

The 1.0 packer is called unchanged. New columns are appended, and processor or
weapon tokens are added from parts.csv. Sensor switches are not copied onto
weapons or processors.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from network.input_v10 import (
    D_CONT as D_CONT_10,
    DOMAIN,
    ENTITY_KIND,
    FIELDS as FIELDS_10,
    KNOWN,
    NOT_APPLICABLE,
    PLACEMENT,
    ROLE,
    SIDE,
    UNKNOWN,
    Frame,
    Record,
    east_north_m,
    known_value,
    load_run,
    pack_frame,
)

SCHEMA = "l1.input/1.1"
RECORD_SCHEMA = "l1.record/1.1"
EXTRA_FIELDS = (
    "scan_az_min_deg",
    "scan_az_max_deg",
    "scan_el_min_deg",
    "scan_el_max_deg",
    "part_on",
    "quantity_remaining",
)
FIELDS = FIELDS_10 + EXTRA_FIELDS
D_CONT = D_CONT_10 + len(EXTRA_FIELDS)
ENTITY_KIND_11 = ENTITY_KIND + ("function_part",)
ROUND_TYPES = {"CT_SHOT"}
# Offsets in kilometres from build_raid.targets(). North, then east. Not sim truth.
PROTECT_OFFSETS_KM = (
    ("sub_1", 0.15, 0.0),
    ("sub_2", 0.0, 0.15),
    ("tank_1", -0.08, 0.05),
    ("tank_2", -0.08, -0.05),
    ("tank_3", 0.05, -0.12),
)


@dataclass(frozen=True)
class PartSample:
    t_s: float
    platform: str
    kind: str
    name: str
    type_name: str
    on: int | None
    yaw_deg: float | None
    pitch_deg: float | None
    fov_az_min_deg: float | None
    fov_az_max_deg: float | None
    fov_el_min_deg: float | None
    fov_el_max_deg: float | None
    scan_az_min_deg: float | None
    scan_az_max_deg: float | None
    scan_el_min_deg: float | None
    scan_el_max_deg: float | None
    can_scan_az: int | None
    can_scan_el: int | None
    quantity_raw: float | None


def _opt_int(text: str) -> int | None:
    return None if text == "na" or text == "" else int(float(text))


def _opt_deg(text: str) -> float | None:
    if text == "na" or text == "":
        return None
    return int(text) / 1e6


def _opt_float(text: str) -> float | None:
    if text == "na" or text == "":
        return None
    return float(text)


def load_parts(path: Path) -> tuple[PartSample, ...]:
    rows: list[PartSample] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 19:
                continue
            rows.append(
                PartSample(
                    t_s=int(row[0]) / 1e6,
                    platform=row[1],
                    kind=row[2],
                    name=row[3],
                    type_name=row[4],
                    on=_opt_int(row[5]),
                    yaw_deg=_opt_deg(row[6]),
                    pitch_deg=_opt_deg(row[7]),
                    fov_az_min_deg=_opt_deg(row[8]),
                    fov_az_max_deg=_opt_deg(row[9]),
                    fov_el_min_deg=_opt_deg(row[10]),
                    fov_el_max_deg=_opt_deg(row[11]),
                    scan_az_min_deg=_opt_deg(row[12]),
                    scan_az_max_deg=_opt_deg(row[13]),
                    scan_el_min_deg=_opt_deg(row[14]),
                    scan_el_max_deg=_opt_deg(row[15]),
                    can_scan_az=_opt_int(row[16]),
                    can_scan_el=_opt_int(row[17]),
                    quantity_raw=_opt_float(row[18]),
                )
            )
    return tuple(rows)


def slew_observed(parts: tuple[PartSample, ...]) -> bool:
    """True only when a sensor yaw or pitch actually changes."""
    seen: dict[tuple[str, str], set[tuple[float, float]]] = {}
    for row in parts:
        if row.kind != "sensor" or row.yaw_deg is None or row.pitch_deg is None:
            continue
        seen.setdefault((row.platform, row.name), set()).add((row.yaw_deg, row.pitch_deg))
    return any(len(values) > 1 for values in seen.values())


def _put(states: list[int], values: list[float], name: str, state: int, number: float | None = None) -> None:
    index = FIELDS.index(name)
    states[index] = state
    if state == KNOWN and number is not None:
        values[index] = float(number)


def _widen(values: list[float], states: list[int]) -> tuple[list[float], list[int]]:
    return values + [0.0] * len(EXTRA_FIELDS), states + [NOT_APPLICABLE] * len(EXTRA_FIELDS)


def _latest(parts: tuple[PartSample, ...], query_time_s: float) -> dict[tuple[str, str, str], PartSample]:
    found: dict[tuple[str, str, str], PartSample] = {}
    for row in parts:
        if row.t_s <= query_time_s:
            found[(row.platform, row.kind, row.name)] = row
    return found


def _role(type_name: str) -> str:
    if type_name in {"CT_BEAM", "CT_LASER_FIRE", "CT_SHOT", "CT_GUN_FIRE"}:
        return "intercept"
    if type_name in {"CT_JAMMER", "CT_JAM_FIRE"}:
        return "jam"
    if type_name in {"CT_PICTURES", "CT_TRACKS"}:
        return "detect"
    return "other"


def _effector(type_name: str) -> str:
    if type_name == "CT_BEAM":
        return "energy"
    if type_name == "CT_JAMMER":
        return "rf_jam"
    if type_name == "CT_SHOT":
        return "consumable"
    return "none"


def _scan(values: list[float], states: list[int], row: PartSample) -> None:
    if row.can_scan_az == 1 and row.scan_az_min_deg is not None and row.scan_az_max_deg is not None:
        _put(states, values, "scan_az_min_deg", KNOWN, row.scan_az_min_deg)
        _put(states, values, "scan_az_max_deg", KNOWN, row.scan_az_max_deg)
    if row.can_scan_el == 1 and row.scan_el_min_deg is not None and row.scan_el_max_deg is not None:
        _put(states, values, "scan_el_min_deg", KNOWN, row.scan_el_min_deg)
        _put(states, values, "scan_el_max_deg", KNOWN, row.scan_el_max_deg)


def _quantity(states: list[int], values: list[float], row: PartSample) -> None:
    if row.kind != "weapon":
        return
    if row.type_name in ROUND_TYPES and row.quantity_raw is not None:
        _put(states, values, "quantity_remaining", KNOWN, row.quantity_raw)
    else:
        _put(states, values, "quantity_remaining", UNKNOWN)


def pack_v11(record: Record, parts: tuple[PartSample, ...], query_time_s: float) -> tuple[Frame, dict[str, float | None]]:
    """Return the 1.1 frame and the raw weapon quantities, which are not model features."""
    base = pack_frame(record, query_time_s)
    frame = Frame(
        SCHEMA,
        record.run_id,
        query_time_s,
        record.origin_lat_deg,
        record.origin_lon_deg,
        accepted=base.accepted,
        own_index=list(base.own_index),
        unit_valid=list(base.unit_valid),
        omissions=list(base.omissions),
    )
    for index in range(len(base.token_id)):
        values, states = _widen(base.cont[index], base.feature_state[index])
        frame.token_id.append(base.token_id[index])
        frame.entity_valid.append(base.entity_valid[index])
        frame.cont.append(values)
        frame.feature_state.append(states)
        frame.category.append(list(base.category[index]))
        frame.source_s.append(base.source_s[index])
        frame.part_parent.append(base.part_parent[index])
    latest = _latest(parts, query_time_s)
    raw_quantity: dict[str, float | None] = {}
    by_id = {token_id: index for index, token_id in enumerate(frame.token_id)}
    for key, row in sorted(latest.items()):
        platform, kind, name = key
        if platform not in by_id:
            frame.accepted = False
            continue
        if kind == "sensor":
            token = by_id.get(f"{platform}/{name}")
            if token is None:
                continue
            _scan(frame.cont[token], frame.feature_state[token], row)
            continue
        token_id = f"{platform}/{kind}/{name}"
        values, states = _widen([0.0] * D_CONT_10, [NOT_APPLICABLE] * D_CONT_10)
        if row.on is None:
            _put(states, values, "part_on", UNKNOWN)
        else:
            _put(states, values, "part_on", KNOWN, row.on)
        _quantity(states, values, row)
        raw_quantity[token_id] = row.quantity_raw
        cats = [0] * 8
        cats[0] = ENTITY_KIND_11.index("function_part")
        cats[1] = 1  # red, this log only contains the red side's own parts
        cats[2] = ["not_applicable", "detect", "jam", "intercept", "protect", "comm", "other"].index(_role(row.type_name))
        cats[4] = ["not_applicable", "none", "energy", "rf_jam", "consumable", "other"].index(_effector(row.type_name))
        frame.token_id.append(token_id)
        frame.entity_valid.append(True)
        frame.cont.append(values)
        frame.feature_state.append(states)
        frame.category.append(cats)
        frame.source_s.append(row.t_s)
        frame.part_parent.append(by_id[platform])
    _append_protect_targets(frame)
    return frame, raw_quantity


def _append_protect_targets(frame: Frame) -> None:
    """Scene points from targets(). Altitude and speed are not in that config."""
    for name, north_km, east_km in PROTECT_OFFSETS_KM:
        lat = frame.origin_lat_deg + north_km / 111.32
        lon = frame.origin_lon_deg + east_km / (111.32 * math.cos(math.radians(frame.origin_lat_deg)))
        east, north = east_north_m(lat, lon, frame.origin_lat_deg, frame.origin_lon_deg)
        values, states = _widen([0.0] * D_CONT_10, [NOT_APPLICABLE] * D_CONT_10)
        _put(states, values, "east_m", KNOWN, east)
        _put(states, values, "north_m", KNOWN, north)
        cats = [0] * 8
        cats[0] = ENTITY_KIND.index("objective")
        cats[1] = SIDE.index("gold")
        cats[2] = ROLE.index("protect")
        cats[5] = PLACEMENT.index("fixed")
        cats[7] = DOMAIN.index("land")
        frame.token_id.append(name)
        frame.entity_valid.append(True)
        frame.cont.append(values)
        frame.feature_state.append(states)
        frame.category.append(cats)
        frame.source_s.append(None)
        frame.part_parent.append(-1)


def value(frame: Frame, token: int, field_name: str) -> float | None:
    return known_value(frame, token, field_name) if field_name in FIELDS_10 else _extra_value(frame, token, field_name)


def _extra_value(frame: Frame, token: int, field_name: str) -> float | None:
    index = FIELDS.index(field_name)
    if frame.feature_state[token][index] != KNOWN:
        return None
    return frame.cont[token][index]


def load_v11(run_dir: Path, run_id: str) -> tuple[Record, tuple[PartSample, ...]]:
    return load_run(run_dir, run_id), load_parts(run_dir / "parts.csv")
