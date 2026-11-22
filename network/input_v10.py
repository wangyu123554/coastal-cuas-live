"""Approved observation contract l1.input/1.0.

This packer reads V20.1-style logs and builds one query frame. It does not import
the 54-column encoder, does not load weights, and does not emit commands.

Review decisions locked into this version:

- Lightweight history summary in the frame. Full source rows stay on the record.
- Continuous width is 32. The duplicate gap column was removed before any code shipped.
- Radar error numbers stay on the record. The frame marks them unknown.
- 64 is the own-slot cap and the budget for future named fields, not padding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "l1.input/1.0"
RECORD_SCHEMA = "l1.record/1.0"
D_CONT = 32
D_CONT_BUDGET = 64
N_OWN_CAP = 64
N_ENTITY_CAP = 256
N_PREFIX = 2
CAT_DIM = 8

# Type-file scan limits. They are not written into the field-of-view columns.
RADAR_ELEVATION_SCAN_LIMITS_DEG = (0.0, 50.0)

ORIGIN_LAT_DEG = 34.6197
ORIGIN_LON_DEG = 112.4539

FIELDS: tuple[str, ...] = (
    "east_m",
    "north_m",
    "up_m",
    "v_north_mps",
    "v_east_mps",
    "v_up_mps",
    "state_age_s",
    "observe_age_s",
    "track_life_s",
    "update_gap_s",
    "sensor_on",
    "yaw_deg",
    "pitch_deg",
    "fov_az_min_deg",
    "fov_az_max_deg",
    "fov_el_min_deg",
    "fov_el_max_deg",
    "range_m",
    "bearing_deg",
    "elevation_deg",
    "range_sigma",
    "bearing_sigma",
    "elevation_sigma",
    "snr",
    "update_count",
    "max_range_m",
    "dest_east_m",
    "dest_north_m",
    "dest_up_m",
    "delta_east_m",
    "delta_north_m",
    "delta_up_m",
)

NOT_APPLICABLE = 0
UNKNOWN = 1
MISSING = 2
KNOWN = 3
STATE_NAME = {0: "not_applicable", 1: "unknown", 2: "missing", 3: "known"}

ENTITY_KIND = ("not_applicable", "context", "intent", "own_platform", "sensor_part", "track", "objective", "zone")
SIDE = ("not_applicable", "red", "blue", "gold", "neutral", "unknown")
ROLE = ("not_applicable", "detect", "jam", "intercept", "protect", "comm", "other")
SENSOR_CLASS = ("not_applicable", "none", "rf", "eoir", "geometric", "other")
EFFECTOR_CLASS = ("not_applicable", "none", "energy", "rf_jam", "consumable", "other")
PLACEMENT = ("not_applicable", "fixed", "route_capable", "unknown")
LIFECYCLE = ("not_applicable", "absent", "init", "update", "coast", "drop")
DOMAIN = ("not_applicable", "land", "air", "surface", "unknown")
CATEGORIES = (
    ("entity_kind", ENTITY_KIND),
    ("side", SIDE),
    ("role", ROLE),
    ("sensor_class", SENSOR_CLASS),
    ("effector_class", EFFECTOR_CLASS),
    ("placement", PLACEMENT),
    ("lifecycle", LIFECYCLE),
    ("domain", DOMAIN),
)

RED_ORDER = (
    "radar_1",
    "radar_2",
    "ir_1",
    "ir_2",
    "ir_3",
    "laser_1",
    "laser_2",
    "laser_3",
    "jam_1",
    "jam_2",
    "net_1",
    "net_2",
    "net_3",
)


def _code(table: tuple[str, ...], name: str) -> int:
    return table.index(name)


@dataclass(frozen=True)
class Profile:
    role: str
    sensor_class: str
    effector_class: str
    placement: str
    domain: str
    max_range_m: float | None
    reports_range: bool
    reports_bearing: bool
    reports_elevation: bool
    reports_velocity: bool
    reports_snr: bool


PROFILES: dict[str, Profile] = {
    "CUASCAT_RADAR_8": Profile("detect", "rf", "none", "fixed", "land", 8000.0, True, True, False, False, False),
    "CT_IR_TURRET": Profile("detect", "eoir", "none", "fixed", "land", None, False, False, False, True, True),
    "CT_LASER": Profile("intercept", "geometric", "energy", "fixed", "land", 4000.0, False, False, False, False, False),
    "CT_JAM": Profile("jam", "geometric", "rf_jam", "fixed", "land", 4000.0, False, False, False, False, False),
    "CT_GUN": Profile("intercept", "geometric", "consumable", "fixed", "land", 600.0, False, False, False, False, False),
}


@dataclass(frozen=True)
class StateSample:
    t_s: float
    name: str
    type_name: str
    lat_deg: float
    lon_deg: float
    up_m: float
    v_north_mps: float
    v_east_mps: float
    v_up_mps: float
    dest_east_m: float | None = None
    dest_north_m: float | None = None
    dest_up_m: float | None = None
    placement: str | None = None
    domain: str | None = None


@dataclass(frozen=True)
class PoseSample:
    t_s: float
    name: str
    sensor: str
    on: int
    yaw_deg: float
    pitch_deg: float
    fov_az_min_deg: float
    fov_az_max_deg: float
    fov_el_min_deg: float
    fov_el_max_deg: float


@dataclass(frozen=True)
class TrackSample:
    kind: str
    event_s: float
    update_s: float
    start_s: float
    observer: str
    sensor: str
    track_name: str
    track_number: int
    update_count: int
    loc_valid: bool
    alt_known: bool
    vel_valid: bool
    range_valid: bool
    bearing_valid: bool
    elev_valid: bool
    snr_valid: bool
    lat_deg: float | None
    lon_deg: float | None
    up_m: float | None
    v_north_mps: float | None
    v_east_mps: float | None
    v_up_mps: float | None
    range_m: float | None
    bearing_deg: float | None
    elevation_deg: float | None
    range_sigma_raw: float | None
    bearing_sigma_raw: float | None
    elevation_sigma_raw: float | None
    snr: float | None


@dataclass(frozen=True)
class Record:
    """Full source history. Row counts are not seconds."""

    schema: str
    run_id: str
    states: tuple[StateSample, ...]
    poses: tuple[PoseSample, ...]
    tracks: tuple[TrackSample, ...]
    origin_lat_deg: float = ORIGIN_LAT_DEG
    origin_lon_deg: float = ORIGIN_LON_DEG


@dataclass(frozen=True)
class Omission:
    kind: str
    name: str
    reason: str


@dataclass
class Frame:
    schema: str
    run_id: str
    query_time_s: float
    origin_lat_deg: float
    origin_lon_deg: float
    accepted: bool
    cont: list[list[float]] = field(default_factory=list)
    feature_state: list[list[int]] = field(default_factory=list)
    category: list[list[int]] = field(default_factory=list)
    entity_valid: list[bool] = field(default_factory=list)
    token_id: list[str] = field(default_factory=list)
    source_s: list[float | None] = field(default_factory=list)
    part_parent: list[int] = field(default_factory=list)
    own_index: list[int] = field(default_factory=list)
    unit_valid: list[bool] = field(default_factory=list)
    omissions: list[Omission] = field(default_factory=list)

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        n = len(self.cont)
        width = len(self.cont[0]) if n else D_CONT
        return {
            "cont": (n, width),
            "feature_state": (n, width),
            "category": (n, CAT_DIM),
            "entity_valid": (n,),
            "own_index": (len(self.own_index),),
            "unit_valid": (len(self.unit_valid),),
            "part_parent": (n,),
        }


def _blank() -> tuple[list[float], list[int]]:
    return [0.0] * D_CONT, [NOT_APPLICABLE] * D_CONT


def _put(values: list[float], states: list[int], name: str, state: int, number: float | None = None) -> None:
    index = FIELDS.index(name)
    states[index] = state
    if state == KNOWN and number is not None:
        values[index] = float(number)


def _cats(**names: str) -> list[int]:
    found = {key: value for key, value in names.items()}
    return [_code(table, found.get(key, "not_applicable")) for key, table in CATEGORIES]


def east_north_m(lat_deg: float, lon_deg: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    north = (lat_deg - origin_lat) * 111320.0
    east = (lon_deg - origin_lon) * 111320.0 * math.cos(math.radians(origin_lat))
    return east, north


def _num(text: str, scale: float = 1.0) -> float | None:
    if text == "na" or text == "":
        return None
    return float(text) / scale


def _flag(text: str) -> bool:
    return text == "1"


def load_run(path: Path, run_id: str) -> Record:
    """Read state, pose, and tracks. Truth and target names are not opened."""
    states: list[StateSample] = []
    with (path / "state.csv").open(encoding="utf-8", newline="") as handle:
        for row in __import__("csv").reader(handle):
            if len(row) < 9:
                continue
            states.append(
                StateSample(
                    t_s=int(row[0]) / 1e6,
                    name=row[1],
                    type_name=row[2],
                    lat_deg=int(row[3]) / 1e7,
                    lon_deg=int(row[4]) / 1e7,
                    up_m=int(row[5]) / 1e3,
                    v_north_mps=int(row[6]) / 1e3,
                    v_east_mps=int(row[7]) / 1e3,
                    v_up_mps=-int(row[8]) / 1e3,
                )
            )
    poses: list[PoseSample] = []
    with (path / "sensor_pose.csv").open(encoding="utf-8", newline="") as handle:
        for row in __import__("csv").reader(handle):
            if len(row) < 11:
                continue
            poses.append(
                PoseSample(
                    t_s=int(row[0]) / 1e6,
                    name=row[1],
                    sensor=row[2],
                    on=int(row[3]),
                    yaw_deg=int(row[5]) / 1e6,
                    pitch_deg=int(row[6]) / 1e6,
                    fov_az_min_deg=int(row[7]) / 1e6,
                    fov_az_max_deg=int(row[8]) / 1e6,
                    fov_el_min_deg=int(row[9]) / 1e6,
                    fov_el_max_deg=int(row[10]) / 1e6,
                )
            )
    tracks: list[TrackSample] = []
    with (path / "tracks.csv").open(encoding="utf-8", newline="") as handle:
        for row in __import__("csv").reader(handle):
            if len(row) < 29:
                continue
            vel_valid = _flag(row[11])
            tracks.append(
                TrackSample(
                    kind=row[0],
                    event_s=int(row[1]) / 1e6,
                    update_s=int(row[2]) / 1e6,
                    start_s=int(row[3]) / 1e6,
                    observer=row[4],
                    sensor=row[5],
                    track_name=row[6],
                    track_number=int(row[7]),
                    update_count=int(row[8]),
                    loc_valid=_flag(row[9]),
                    alt_known=_flag(row[10]),
                    vel_valid=vel_valid,
                    range_valid=_flag(row[12]),
                    bearing_valid=_flag(row[13]),
                    elev_valid=_flag(row[14]),
                    snr_valid=_flag(row[15]),
                    lat_deg=_num(row[16], 1e7),
                    lon_deg=_num(row[17], 1e7),
                    up_m=_num(row[18], 1e3),
                    v_north_mps=_num(row[19], 1e3) if vel_valid else None,
                    v_east_mps=_num(row[20], 1e3) if vel_valid else None,
                    v_up_mps=(-_num(row[21], 1e3) if _num(row[21], 1e3) is not None else None) if vel_valid else None,
                    range_m=_num(row[22], 1e3),
                    bearing_deg=_num(row[23], 1e6),
                    elevation_deg=_num(row[24], 1e6),
                    range_sigma_raw=_num(row[25]),
                    bearing_sigma_raw=_num(row[26]),
                    elevation_sigma_raw=_num(row[27]),
                    snr=_num(row[28]),
                )
            )
    return Record(RECORD_SCHEMA, run_id, tuple(states), tuple(poses), tuple(tracks))


def source_times(record: Record, kind: str, name: str) -> tuple[float, ...]:
    """Timestamps of the raw rows. A limit of 32 means 32 rows, not 32 seconds."""
    if kind == "state":
        return tuple(row.t_s for row in record.states if row.name == name)
    if kind == "pose":
        return tuple(row.t_s for row in record.poses if row.name == name)
    if kind == "track":
        return tuple(row.event_s for row in record.tracks if row.observer == name)
    raise KeyError(kind)


def latest_source_times(record: Record, kind: str, name: str, limit: int) -> tuple[float, ...]:
    times = source_times(record, kind, name)
    return times[-limit:]


def known_value(frame: Frame, token: int, field_name: str) -> float | None:
    index = FIELDS.index(field_name)
    if frame.feature_state[token][index] != KNOWN:
        return None
    return frame.cont[token][index]


def _profile(type_name: str, profiles: dict[str, Profile]) -> Profile:
    if type_name in profiles:
        return profiles[type_name]
    return Profile("other", "other", "other", "unknown", "unknown", None, False, False, False, False, False)


def _append(frame: Frame, token_id: str, valid: bool, values: list[float], states: list[int], cats: list[int], source: float | None, parent: int) -> int:
    frame.token_id.append(token_id)
    frame.entity_valid.append(valid)
    frame.cont.append(values)
    frame.feature_state.append(states)
    frame.category.append(cats)
    frame.source_s.append(source)
    frame.part_parent.append(parent)
    return len(frame.token_id) - 1


def _kinematics_from_platform(values, states, row: StateSample, previous: StateSample | None, origin_lat: float, origin_lon: float, t_q: float) -> None:
    east, north = east_north_m(row.lat_deg, row.lon_deg, origin_lat, origin_lon)
    _put(values, states, "east_m", KNOWN, east)
    _put(values, states, "north_m", KNOWN, north)
    _put(values, states, "up_m", KNOWN, row.up_m)
    _put(values, states, "v_north_mps", KNOWN, row.v_north_mps)
    _put(values, states, "v_east_mps", KNOWN, row.v_east_mps)
    _put(values, states, "v_up_mps", KNOWN, row.v_up_mps)
    _put(values, states, "state_age_s", KNOWN, t_q - row.t_s)
    if previous is None:
        _put(values, states, "update_gap_s", UNKNOWN)
        _put(values, states, "delta_east_m", UNKNOWN)
        _put(values, states, "delta_north_m", UNKNOWN)
        _put(values, states, "delta_up_m", UNKNOWN)
        return
    prev_east, prev_north = east_north_m(previous.lat_deg, previous.lon_deg, origin_lat, origin_lon)
    _put(values, states, "update_gap_s", KNOWN, row.t_s - previous.t_s)
    _put(values, states, "delta_east_m", KNOWN, east - prev_east)
    _put(values, states, "delta_north_m", KNOWN, north - prev_north)
    _put(values, states, "delta_up_m", KNOWN, row.up_m - previous.up_m)


def _destination(values, states, row: StateSample) -> None:
    if row.dest_east_m is None:
        return
    _put(values, states, "dest_east_m", KNOWN, row.dest_east_m)
    _put(values, states, "dest_north_m", KNOWN, row.dest_north_m)
    _put(values, states, "dest_up_m", KNOWN, row.dest_up_m)


def pack_frame(
    record: Record,
    query_time_s: float,
    *,
    profiles: dict[str, Profile] | None = None,
    entity_cap: int = N_ENTITY_CAP,
    own_cap: int = N_OWN_CAP,
    intent: bool = False,
) -> Frame:
    catalog = PROFILES if profiles is None else {**PROFILES, **profiles}
    frame = Frame(
        SCHEMA,
        record.run_id,
        query_time_s,
        record.origin_lat_deg,
        record.origin_lon_deg,
        accepted=True,
        own_index=[-1] * own_cap,
        unit_valid=[False] * own_cap,
    )
    context_v, context_s = _blank()
    _append(frame, "context", True, context_v, context_s, _cats(entity_kind="context"), None, -1)
    intent_v, intent_s = _blank()
    _append(frame, "intent", intent, intent_v, intent_s, _cats(entity_kind="intent" if intent else "not_applicable"), None, -1)

    latest_state: dict[str, StateSample] = {}
    previous_state: dict[str, StateSample] = {}
    for row in record.states:
        if row.t_s > query_time_s:
            continue
        if row.name in latest_state:
            previous_state[row.name] = latest_state[row.name]
        latest_state[row.name] = row
    names = [name for name in RED_ORDER if name in latest_state]
    names.extend(sorted(set(latest_state) - set(names)))

    own_tokens: dict[str, int] = {}
    room = entity_cap
    placed_own: list[str] = []
    for name in names:
        if len(placed_own) >= own_cap or room <= 0:
            frame.accepted = False
            reason = "己方槽已满" if len(placed_own) >= own_cap else "实体令牌已满"
            frame.omissions.append(Omission("own_platform", name, reason))
            continue
        row = latest_state[name]
        profile = _profile(row.type_name, catalog)
        values, states = _blank()
        _kinematics_from_platform(values, states, row, previous_state.get(name), record.origin_lat_deg, record.origin_lon_deg, query_time_s)
        _destination(values, states, row)
        if profile.max_range_m is None:
            _put(values, states, "max_range_m", UNKNOWN)
        else:
            _put(values, states, "max_range_m", KNOWN, profile.max_range_m)
        placement = row.placement or profile.placement
        domain = row.domain or profile.domain
        token = _append(
            frame,
            name,
            True,
            values,
            states,
            _cats(
                entity_kind="own_platform",
                side="red",
                role=profile.role,
                sensor_class=profile.sensor_class,
                effector_class=profile.effector_class,
                placement=placement,
                domain=domain,
            ),
            row.t_s,
            -1,
        )
        own_tokens[name] = token
        frame.own_index[len(placed_own)] = token
        frame.unit_valid[len(placed_own)] = True
        placed_own.append(name)
        room -= 1

    latest_pose: dict[tuple[str, str], PoseSample] = {}
    previous_pose: dict[tuple[str, str], PoseSample] = {}
    for row in record.poses:
        if row.t_s > query_time_s or row.name not in own_tokens:
            continue
        key = (row.name, row.sensor)
        if key in latest_pose:
            previous_pose[key] = latest_pose[key]
        latest_pose[key] = row
    for name in placed_own:
        for key, row in sorted((item for item in latest_pose.items() if item[0][0] == name), key=lambda item: item[0][1]):
            if room <= 0:
                frame.accepted = False
                frame.omissions.append(Omission("sensor_part", f"{name}/{row.sensor}", "实体令牌已满"))
                continue
            profile = _profile(latest_state[name].type_name, catalog)
            values, states = _blank()
            _put(values, states, "sensor_on", KNOWN, row.on)
            _put(values, states, "yaw_deg", KNOWN, row.yaw_deg)
            _put(values, states, "pitch_deg", KNOWN, row.pitch_deg)
            _put(values, states, "fov_az_min_deg", KNOWN, row.fov_az_min_deg)
            _put(values, states, "fov_az_max_deg", KNOWN, row.fov_az_max_deg)
            _put(values, states, "fov_el_min_deg", KNOWN, row.fov_el_min_deg)
            _put(values, states, "fov_el_max_deg", KNOWN, row.fov_el_max_deg)
            _put(values, states, "state_age_s", KNOWN, query_time_s - row.t_s)
            previous = previous_pose.get(key)
            if previous is None:
                _put(values, states, "update_gap_s", UNKNOWN)
            else:
                _put(values, states, "update_gap_s", KNOWN, row.t_s - previous.t_s)
            if profile.max_range_m is None:
                _put(values, states, "max_range_m", UNKNOWN)
            else:
                _put(values, states, "max_range_m", KNOWN, profile.max_range_m)
            _append(
                frame,
                f"{name}/{row.sensor}",
                True,
                values,
                states,
                _cats(
                    entity_kind="sensor_part",
                    side="red",
                    role=profile.role,
                    sensor_class=profile.sensor_class,
                    effector_class="none",
                    placement=latest_state[name].placement or profile.placement,
                    domain=latest_state[name].domain or profile.domain,
                ),
                row.t_s,
                own_tokens[name],
            )
            room -= 1

    grouped: dict[tuple[str, str, int], list[TrackSample]] = {}
    for row in record.tracks:
        if row.event_s > query_time_s:
            continue
        grouped.setdefault((row.observer, row.track_name, row.track_number), []).append(row)
    for key in sorted(grouped):
        rows = grouped[key]
        current = rows[-1]
        if current.kind == "drop":
            continue
        if room <= 0:
            frame.accepted = False
            frame.omissions.append(Omission("track", f"{key[0]}:{key[1]}.{key[2]}", "实体令牌已满"))
            continue
        observer = latest_state.get(current.observer)
        profile = _profile(observer.type_name, catalog) if observer is not None else _profile("", catalog)
        values, states = _blank()
        _put(values, states, "observe_age_s", KNOWN, query_time_s - current.update_s)
        _put(values, states, "track_life_s", KNOWN, current.update_s - current.start_s)
        _put(values, states, "update_count", KNOWN, current.update_count)
        if len(rows) >= 2:
            _put(values, states, "update_gap_s", KNOWN, current.event_s - rows[-2].event_s)
        else:
            _put(values, states, "update_gap_s", UNKNOWN)
        if current.loc_valid and current.lat_deg is not None and current.lon_deg is not None:
            east, north = east_north_m(current.lat_deg, current.lon_deg, record.origin_lat_deg, record.origin_lon_deg)
            _put(values, states, "east_m", KNOWN, east)
            _put(values, states, "north_m", KNOWN, north)
            if current.alt_known and current.up_m is not None:
                _put(values, states, "up_m", KNOWN, current.up_m)
            else:
                _put(values, states, "up_m", UNKNOWN)
            if len(rows) >= 2 and rows[-2].loc_valid and rows[-2].lat_deg is not None and rows[-2].lon_deg is not None:
                prev_east, prev_north = east_north_m(rows[-2].lat_deg, rows[-2].lon_deg, record.origin_lat_deg, record.origin_lon_deg)
                _put(values, states, "delta_east_m", KNOWN, east - prev_east)
                _put(values, states, "delta_north_m", KNOWN, north - prev_north)
                if current.alt_known and rows[-2].alt_known and current.up_m is not None and rows[-2].up_m is not None:
                    _put(values, states, "delta_up_m", KNOWN, current.up_m - rows[-2].up_m)
                else:
                    _put(values, states, "delta_up_m", UNKNOWN)
            else:
                _put(values, states, "delta_east_m", UNKNOWN)
                _put(values, states, "delta_north_m", UNKNOWN)
                _put(values, states, "delta_up_m", UNKNOWN)
        if profile.reports_velocity and current.vel_valid and current.v_north_mps is not None:
            _put(values, states, "v_north_mps", KNOWN, current.v_north_mps)
            _put(values, states, "v_east_mps", KNOWN, current.v_east_mps)
            _put(values, states, "v_up_mps", KNOWN, current.v_up_mps)
        elif profile.reports_velocity:
            _put(values, states, "v_north_mps", UNKNOWN)
            _put(values, states, "v_east_mps", UNKNOWN)
            _put(values, states, "v_up_mps", UNKNOWN)
        _measure(values, states, current, profile)
        parent = own_tokens.get(current.observer, -1)
        _append(
            frame,
            f"{current.observer}:{current.track_name}.{current.track_number}",
            True,
            values,
            states,
            _cats(entity_kind="track", side="unknown", sensor_class=profile.sensor_class, lifecycle=current.kind if current.kind in LIFECYCLE else "update"),
            current.event_s,
            parent,
        )
        room -= 1
    return frame


def _measure(values, states, row: TrackSample, profile: Profile) -> None:
    _optional(values, states, "range_m", profile.reports_range, row.range_valid, row.range_m)
    _optional(values, states, "bearing_deg", profile.reports_bearing, row.bearing_valid, row.bearing_deg)
    _optional(values, states, "elevation_deg", profile.reports_elevation, row.elev_valid, row.elevation_deg)
    _optional(values, states, "snr", profile.reports_snr, row.snr_valid, row.snr)
    # Raw sigma stays on TrackSample. The frame does not treat a returned 0 as a known error bar.
    if profile.reports_range and row.range_valid:
        _put(values, states, "range_sigma", UNKNOWN)
    if profile.reports_bearing and row.bearing_valid:
        _put(values, states, "bearing_sigma", UNKNOWN)


def _optional(values, states, name: str, reports: bool, valid: bool, number: float | None) -> None:
    if not reports:
        return
    if valid and number is not None:
        _put(values, states, name, KNOWN, number)
    else:
        _put(values, states, name, UNKNOWN)
