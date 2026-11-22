"""Read the scenario tables and write one command file per assignment."""

from __future__ import annotations

import csv
import importlib
import json
import threading
import time
from pathlib import Path

import torch

from network.allocate import (
    NAMES,
    ORIGIN_LAT,
    ORIGIN_LON,
    PICK_ORDER,
    POSITION,
    TYPE_NAME,
    _allowed,
    load_allocator,
    predict,
)
from network.geom import _meters
from network.input_v10 import RECORD_SCHEMA, Record, StateSample, TrackSample
from network.input_v11 import pack_v11

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "network" / "weights" / "seed_3.pt"


class _Tail:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = ""

    def rows(self) -> list[list[str]]:
        if not self.path.is_file():
            return []
        with self.path.open(encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return []
        self.pending += chunk
        if not self.pending.endswith("\n"):
            lines, _, self.pending = self.pending.rpartition("\n")
            text = lines
        else:
            text, self.pending = self.pending, ""
        return list(csv.reader(text.splitlines())) if text else []


def _command(row: dict) -> str:
    token = row.get("token") or ""
    if row["task"] not in {"observe", "soft", "hard"} or ":" not in token or "." not in token:
        return ""
    body = token.split(":", 1)[1]
    name, number = body.rsplit(".", 1)
    if not number.isdigit():
        return ""
    verb = {"observe": "OBSERVE", "soft": "SOFT", "hard": "HARD"}[row["task"]]
    return f"{row['platform']} {verb} {name} {number}"


def load_policy(spec: str):
    """builtin uses seed 3. Any other value is module:function and must return the same assignment dict."""
    if spec == "builtin":
        model = load_allocator(WEIGHTS)

        def assign(frame):
            with torch.no_grad():
                return predict(model, frame)

        return assign
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    return function


def serve(work: Path, policy: str, stop: threading.Event) -> None:
    assign = load_policy(policy)
    states_tail = _Tail(work / "state.csv")
    tracks_tail = _Tail(work / "tracks.csv")
    audit_tail = _Tail(work / "audit_link.csv")
    ticks = (work / "ticks.jsonl").open("a", encoding="utf-8")
    states: dict[str, StateSample] = {}
    tracks: dict[tuple[str, int], TrackSample] = {}
    audit: dict[tuple[str, int], str] = {}
    last_t = -1.0
    sim_t = 0.0
    last_laser = {name: -10.0 for name in NAMES}
    command_index = 0
    print("接口网关已接上。第一拍要等周视雷达航迹，大约 8 秒。")
    while not stop.is_set():
        time.sleep(0.2)
        for row in states_tail.rows():
            if len(row) < 6 or row[1] not in TYPE_NAME:
                continue
            try:
                stamp = int(row[0]) / 1e6
                states[row[1]] = StateSample(stamp, row[1], TYPE_NAME[row[1]], int(row[3]) / 1e7, int(row[4]) / 1e7, int(row[5]) / 1e3, 0.0, 0.0, 0.0)
                sim_t = max(sim_t, stamp)
            except ValueError:
                continue
        for row in tracks_tail.rows():
            if len(row) < 8 or row[4] != "radar_2":
                continue
            try:
                number = int(row[7])
            except ValueError:
                continue
            if row[0] == "drop":
                tracks.pop((row[6], number), None)
                continue
            if len(row) < 19:
                continue
            try:
                sample = TrackSample(
                    "update", int(row[1]) / 1e6, int(row[1]) / 1e6, int(row[1]) / 1e6,
                    "radar_2", "search", row[6], number, 1,
                    True, True, False, True, True, True, False,
                    int(row[16]) / 1e7, int(row[17]) / 1e7, int(row[18]) / 1e3,
                    None, None, None, None, None, None, None, None, None, None,
                )
            except ValueError:
                continue
            tracks[(row[6], number)] = sample
        for row in audit_tail.rows():
            if len(row) < 6:
                continue
            try:
                audit[(row[3], int(row[4]))] = row[5]
            except ValueError:
                continue
        if sim_t < 8 or sim_t < last_t + 2 or len(states) < len(NAMES):
            continue
        samples = tuple(sample for sample in tracks.values() if sample.event_s >= sim_t - 3)
        frame, _raw = pack_v11(
            Record(RECORD_SCHEMA, "coastal-live", tuple(states[name] for name in NAMES), (), samples, ORIGIN_LAT, ORIGIN_LON),
            (),
            sim_t,
        )
        view = []
        for sample in samples:
            token = f"radar_2:{sample.track_name}.{sample.track_number}"
            view.append(
                {
                    "token": token,
                    "track": f"{sample.track_name}.{sample.track_number}",
                    "target": audit.get((sample.track_name, sample.track_number), ""),
                    "legal": [name for name in NAMES if _allowed(name, sample.lat_deg or 0.0, sample.lon_deg or 0.0)],
                }
            )
        decision = assign(frame)
        by_platform = {row["platform"]: row for row in decision["assignments"]}
        hard_used: set[str] = set()
        laser_used: set[str] = set()
        jam_used: set[str] = set()
        looked: set[str] = set()
        ordered = []
        for name in list(PICK_ORDER) + [name for name in NAMES if name not in PICK_ORDER]:
            row = by_platform[name]
            token = row["token"]
            if token and row["task"] == "observe":
                if token in looked:
                    row = {**row, "task": "none", "token": ""}
                else:
                    looked.add(token)
            elif token and row["task"] == "soft":
                if token in jam_used or token in laser_used:
                    row = {**row, "task": "none", "token": ""}
                else:
                    jam_used.add(token)
            elif token:
                if token in hard_used:
                    row = {**row, "task": "none", "token": ""}
                else:
                    hard_used.add(token)
                    if name.startswith("laser"):
                        laser_used.add(token)
            ordered.append(row)
        outputs = []
        for row in ordered:
            body = _command(row)
            sent = False
            if body and not (row["platform"].startswith("laser") and sim_t - last_laser[row["platform"]] < 4):
                command_index += 1
                (work / f"net_{command_index}.txt").write_text(body + "\n", encoding="utf-8")
                if row["platform"].startswith("laser"):
                    last_laser[row["platform"]] = sim_t
                sent = True
            outputs.append({"platform": row["platform"], "task": row["task"], "token": row["token"], "sent": sent})
        ticks.write(json.dumps({"t": round(sim_t, 3), "inputs": view, "outputs": outputs}, ensure_ascii=False) + "\n")
        ticks.flush()
        named = [row["platform"] for row in outputs if row["sent"]]
        if named:
            print(f"t={sim_t:.1f}s 点名 {', '.join(named)}")
        last_t = sim_t
    ticks.close()
