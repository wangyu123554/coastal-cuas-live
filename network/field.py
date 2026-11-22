"""Coastal airport counter-UAS baseline.

Synthetic equipment parameters for the scene contract. This module is the
coordinate and range source for the AFSIM files. It does not change the
encoder or the 38 input fields.
"""

from __future__ import annotations

import math

SCHEMA = "l1.coastal-baseline/1.0"
ORIGIN_LAT = 30.9195
ORIGIN_LON = 121.935
FIELD_ALT_M = 6.0
SPAWN_M = 4500.0
DIVE_M = 800.0
CRUISE_AGL = (350.0, 400.0, 380.0)
BLUE_SPEED = 40.0

# name, type, north_m, east_m, heading_deg, command task
RED = (
    ("radar_1", "CT_RADAR_SECTOR", 500.0, 0.0, 0.0, "observe"),
    ("radar_2", "CT_RADAR_ROT", -500.0, 0.0, 0.0, "observe"),
    ("eo_1", "CT_EO", 40.0, -380.0, 180.0, "observe"),
    ("eo_2", "CT_EO", 40.0, 380.0, 180.0, "observe"),
    ("jam_1", "CT_RF", 0.0, -250.0, 270.0, "soft"),
    ("jam_2", "CT_RF", 0.0, 250.0, 90.0, "soft"),
    ("laser_1", "CT_LASER", 150.0, 0.0, 0.0, "hard"),
    ("laser_2", "CT_LASER", -150.0, 0.0, 180.0, "hard"),
    ("net_1", "CT_NET", 60.0, 0.0, 0.0, "hard"),
    ("net_2", "CT_NET", -40.0, 0.0, 180.0, "hard"),
    ("hunter_1", "CT_HUNTER", 0.0, -650.0, 270.0, "hard"),
    ("hunter_2", "CT_HUNTER", 0.0, 650.0, 90.0, "hard"),
)

PROTECT = (
    ("rw_w", 0.0, -400.0),
    ("rw_e", 0.0, 400.0),
    ("apron", 80.0, 0.0),
    ("terminal", 160.0, 40.0),
)

# Horizontal gates. Radars search by themselves, so observe is not sent.
GATES = {
    "radar_1": {"task": "observe", "send": False},
    "radar_2": {"task": "observe", "send": False},
    "eo_1": {"task": "observe", "send": True, "max_m": 3000.0},
    "eo_2": {"task": "observe", "send": True, "max_m": 3000.0},
    "jam_1": {"task": "soft", "send": True, "max_m": 1000.0},
    "jam_2": {"task": "soft", "send": True, "max_m": 1000.0},
    "laser_1": {"task": "hard", "send": True, "max_m": 600.0},
    "laser_2": {"task": "hard", "send": True, "max_m": 600.0},
    "net_1": {"task": "hard", "send": True, "max_m": 400.0},
    "net_2": {"task": "hard", "send": True, "max_m": 400.0},
    "hunter_1": {"task": "hard", "send": True, "min_m": 600.0, "max_m": 4000.0, "clear_of_laser_m": 600.0},
    "hunter_2": {"task": "hard", "send": True, "min_m": 600.0, "max_m": 4000.0, "clear_of_laser_m": 600.0},
}

LASER_NAMES = ("laser_1", "laser_2")
END_TIME_S = 600


def ll(north_m: float, east_m: float) -> tuple[float, float]:
    lat = ORIGIN_LAT + north_m / 111_320.0
    lon = ORIGIN_LON + east_m / (111_320.0 * math.cos(math.radians(ORIGIN_LAT)))
    return lat, lon


# Plans written before the field moved still store the previous field origin.
LEGACY_ORIGIN_LAT = 34.5478
LEGACY_ORIGIN_LON = 112.6709


def shift_legacy_point(lat: float, lon: float) -> tuple[float, float]:
    """Keep the same north/east offset, but measure it from the current field."""
    if not (34.2 <= lat <= 34.9 and 112.3 <= lon <= 113.1):
        return lat, lon
    north = (lat - LEGACY_ORIGIN_LAT) * 111_320.0
    east = (lon - LEGACY_ORIGIN_LON) * 111_320.0 * math.cos(math.radians(LEGACY_ORIGIN_LAT))
    return ll(north, east)


def ground_m(north_a: float, east_a: float, north_b: float, east_b: float) -> float:
    return math.hypot(north_a - north_b, east_a - east_b)


def red_positions() -> dict[str, tuple[float, float]]:
    return {name: (north, east) for name, _kind, north, east, _heading, _task in RED}


def blues() -> list[dict]:
    axes = {"n": (1.0, 0.0), "e": (0.0, 1.0), "s": (-1.0, 0.0), "w": (0.0, -1.0)}
    aims = {"a": (0.0, 0.0), "b": (160.0, 40.0), "c": (80.0, 0.0)}
    rows = []
    for wave, created in enumerate(range(0, 401, 80), start=1):
        for direction, (north_sign, east_sign) in axes.items():
            for slot_index, slot in enumerate("abc"):
                aim_n, aim_e = aims[slot]
                lateral = (slot_index - 1) * 180.0
                spawn_n = north_sign * SPAWN_M + east_sign * lateral
                spawn_e = east_sign * SPAWN_M - north_sign * lateral
                delta_n = aim_n - spawn_n
                delta_e = aim_e - spawn_e
                length = math.hypot(delta_n, delta_e) or 1.0
                dive_n = aim_n - delta_n / length * DIVE_M
                dive_e = aim_e - delta_e / length * DIVE_M
                cruise = FIELD_ALT_M + CRUISE_AGL[slot_index]
                rows.append(
                    {
                        "name": f"blue_{direction}{wave}_{slot}",
                        "created": created,
                        "spawn_n": spawn_n,
                        "spawn_e": spawn_e,
                        "dive_n": dive_n,
                        "dive_e": dive_e,
                        "aim_n": aim_n,
                        "aim_e": aim_e,
                        "cruise_alt": cruise,
                        "final_alt": FIELD_ALT_M,
                    }
                )
    return rows


def layout_ok() -> list[str]:
    problems = []
    if len(blues()) != 72:
        problems.append("蓝方不是 72 架")
    positions = red_positions()
    sample_n, sample_e = 820.0, 0.0
    jam = positions["jam_1"]
    if ground_m(sample_n, sample_e, jam[0], jam[1]) > 1000:
        problems.append("北向 820 米进场不在干扰圈内")
    laser = positions["laser_1"]
    if ground_m(1000.0, 0.0, laser[0], laser[1]) <= 600:
        problems.append("激光圈盖到了 1 公里外")
    if ground_m(420.0, 0.0, laser[0], laser[1]) > 600:
        problems.append("激光圈盖不住近场")
    return problems
