"""Twelve-platform assignment for the Coastal baseline.

The scaled encoder is used and not modified. Each platform gets its own
pointer over the rotating radar's tracks. Labels rank tracks by how close
they are to the protected points, then give each weapon a target inside its
own range. Jam does not take a target a laser already has. This is not the
distance-to-weapon gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from network.encoder_v11 import D_MODEL
from network.encoder_v12 import ScaledGlobalEncoder
from network.geom import _batch, _meters
from network.input_v10 import (
    ENTITY_KIND,
    RECORD_SCHEMA,
    RED_ORDER,
    Record,
    StateSample,
    TrackSample,
)
from network.input_v11 import pack_v11
from network.field import (
    FIELD_ALT_M,
    GATES,
    ORIGIN_LAT,
    ORIGIN_LON,
    PROTECT,
    RED,
    ll,
)

SCHEMA = "l1.coastal-allocate/1.2"
TASKS = ("none", "observe", "soft", "hard")
TRACK_KIND = ENTITY_KIND.index("track")
NAMES = tuple(row[0] for row in RED)
# The packer lists oil-depot names first, then any other red name in alphabetical order.
PACK_ORDER = tuple([name for name in RED_ORDER if name in NAMES] + sorted(set(NAMES) - set(RED_ORDER)))
ROW_OF = {name: index for index, name in enumerate(PACK_ORDER)}
POSITION = {name: ll(north, east) for name, _kind, north, east, _heading, _task in RED}
TYPE_NAME = {
    "radar_1": "CUASCAT_RADAR_8",
    "radar_2": "CUASCAT_RADAR_8",
    "eo_1": "CT_IR_TURRET",
    "eo_2": "CT_IR_TURRET",
    "jam_1": "CT_JAM",
    "jam_2": "CT_JAM",
    "laser_1": "CT_LASER",
    "laser_2": "CT_LASER",
    "net_1": "CT_GUN",
    "net_2": "CT_GUN",
    "hunter_1": "CT_GUN",
    "hunter_2": "CT_GUN",
}
PICK_ORDER = (
    "laser_1", "laser_2", "net_1", "net_2",
    "hunter_1", "hunter_2", "jam_1", "jam_2", "eo_1", "eo_2",
)
TASK_OF = {
    "eo_1": "observe", "eo_2": "observe",
    "jam_1": "soft", "jam_2": "soft",
    "laser_1": "hard", "laser_2": "hard",
    "net_1": "hard", "net_2": "hard",
    "hunter_1": "hard", "hunter_2": "hard",
}


class CoastalAllocator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ScaledGlobalEncoder()
        self.threat = nn.Linear(D_MODEL, 1)
        self.geometry = nn.Linear(4, 1)
        self.none = nn.Linear(D_MODEL, 1)
        self.task = nn.Linear(D_MODEL, len(TASKS))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, own = self.encoder(
            batch["cont"], batch["feature_state"], batch["category"], batch["entity_valid"],
            batch["part_parent"], batch["own_index"], batch["unit_valid"],
        )
        own = own[:, [ROW_OF[name] for name in NAMES], :]
        east = batch["cont"][..., 0] / 1000.0
        north = batch["cont"][..., 1] / 1000.0
        known = (batch["feature_state"][..., 0] == 3) & (batch["feature_state"][..., 1] == 3)
        features = torch.stack((east, north, east * east, north * north), dim=-1) * known.unsqueeze(-1).to(encoded.dtype)
        threat = self.geometry(features).squeeze(-1) + 0.1 * self.threat(encoded).squeeze(-1)
        shared = threat.unsqueeze(1).expand(-1, own.shape[1], -1)
        pointer = torch.cat((self.none(own), shared), dim=-1)
        return threat, pointer, self.task(own)


def _track(name: str, number: int, lat: float, lon: float) -> TrackSample:
    return TrackSample(
        "update", 0.0, 0.0, 0.0, "radar_2", "search", name, number, 1,
        True, True, False, True, True, True, False,
        lat, lon, FIELD_ALT_M + 350.0, None, None, None, None, None, None, None, None, None, None,
    )


def _distance(name: str, lat: float, lon: float) -> float:
    plat = POSITION[name]
    return _meters(plat[0], plat[1], lat, lon)[2]


def _threat(lat: float, lon: float) -> float:
    return -min(_meters(lat, lon, *ll(north, east))[2] for _name, north, east in PROTECT)


def _in_laser(lat: float, lon: float) -> bool:
    return _distance("laser_1", lat, lon) <= 600.0 or _distance("laser_2", lat, lon) <= 600.0


def _allowed(name: str, lat: float, lon: float) -> bool:
    gate = GATES[name]
    if not gate.get("send"):
        return False
    distance = _distance(name, lat, lon)
    if distance > float(gate.get("max_m", 0)):
        return False
    if distance < float(gate.get("min_m", 0)):
        return False
    if name.startswith("hunter") and _in_laser(lat, lon):
        return False
    return True


def _best(pool: list[dict], name: str) -> dict:
    return max(pool, key=lambda item: (item["threat"], -_distance(name, item["lat"], item["lon"])))


def _fill_pair(meta: list[dict], choice: dict[str, dict | None], names: tuple[str, str], blocked: set[str]) -> None:
    """Give each platform the closest legal contact that is still free, highest threat first."""
    open_names = set(names)
    while open_names:
        pool = []
        for item in meta:
            if item["token"] is None or item["contact"] in blocked:
                continue
            owners = [name for name in open_names if _allowed(name, item["lat"], item["lon"])]
            if owners:
                pool.append((item, owners))
        if not pool:
            return
        item, owners = max(pool, key=lambda pair: pair[0]["threat"])
        name = min(owners, key=lambda owner: _distance(owner, item["lat"], item["lon"]))
        choice[name] = item
        blocked.add(item["contact"])
        open_names.remove(name)


def _assign(meta: list[dict]) -> dict[str, dict | None]:
    """Highest threat inside each weapon's range. Lasers keep their targets from jam."""
    engaged: set[str] = set()
    laser_targets: set[str] = set()
    choice: dict[str, dict | None] = {name: None for name in NAMES}
    for name in ("laser_1", "laser_2", "hunter_1", "hunter_2"):
        pool = [
            item for item in meta
            if item["token"] is not None and item["contact"] not in engaged and _allowed(name, item["lat"], item["lon"])
        ]
        if not pool:
            continue
        picked = _best(pool, name)
        choice[name] = picked
        engaged.add(picked["contact"])
        if name.startswith("laser"):
            laser_targets.add(picked["contact"])
    _fill_pair(meta, choice, ("net_1", "net_2"), set(engaged))
    _fill_pair(meta, choice, ("jam_1", "jam_2"), set(laser_targets))
    looked: set[str] = set()
    for name in ("eo_1", "eo_2"):
        pool = [
            item for item in meta
            if item["token"] is not None and item["contact"] not in looked and _allowed(name, item["lat"], item["lon"])
        ]
        if not pool:
            continue
        picked = _best(pool, name)
        choice[name] = picked
        looked.add(picked["contact"])
    return choice


def make_scene(contacts: list[tuple[str, float, float]]):
    samples = tuple(_track(name, index + 1, lat, lon) for index, (name, lat, lon) in enumerate(contacts))
    states = tuple(
        StateSample(0.0, name, TYPE_NAME[name], *POSITION[name], FIELD_ALT_M, 0.0, 0.0, 0.0)
        for name in NAMES
    )
    frame, _raw = pack_v11(Record(RECORD_SCHEMA, "coastal", states, (), samples, ORIGIN_LAT, ORIGIN_LON), (), 0.0)
    token_of = {}
    for index, token in enumerate(frame.token_id):
        token_of[token] = index
    meta = []
    for index, (name, lat, lon) in enumerate(contacts):
        token = f"radar_2:{name}.{index + 1}"
        meta.append({
            "contact": name,
            "number": index + 1,
            "lat": lat,
            "lon": lon,
            "threat": _threat(lat, lon),
            "token": token_of.get(token),
            "token_id": token,
        })
    choice = _assign(meta)
    batch = _batch(frame, False)
    mask = legal_mask(batch)
    for slot, name in enumerate(NAMES):
        picked = choice[name]
        if picked is None or picked["token"] is None:
            continue
        if not bool(mask[slot, int(picked["token"]) + 1]):
            choice[name] = None
    pointer = []
    tasks = []
    chosen_rows = []
    for name in NAMES:
        picked = choice[name]
        if picked is None:
            pointer.append(0)
            tasks.append(0)
        else:
            pointer.append(int(picked["token"]) + 1)
            tasks.append(TASKS.index(TASK_OF[name]))
        chosen_rows.append(picked)
    return frame, batch, meta, pointer, tasks, chosen_rows


def legal_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    own = batch["own_index"][0]
    kind = batch["category"][0, :, 0]
    valid = batch["entity_valid"][0]
    mask = torch.zeros(len(NAMES), kind.shape[0] + 1, dtype=torch.bool)
    mask[:, 0] = True
    track = (kind == TRACK_KIND) & valid
    east = batch["cont"][0, :, 0]
    north = batch["cont"][0, :, 1]
    known = (batch["feature_state"][0, :, 0] == 3) & (batch["feature_state"][0, :, 1] == 3)
    for slot, name in enumerate(NAMES):
        gate = GATES[name]
        origin = int(own[ROW_OF[name]])
        if not gate.get("send") or not bool(known[origin]):
            continue
        distance = torch.hypot(east - east[origin], north - north[origin])
        allowed = track & known & (distance <= float(gate.get("max_m", 0))) & (distance >= float(gate.get("min_m", 0)))
        if name.startswith("hunter"):
            outside = torch.ones_like(allowed)
            for laser_name in ("laser_1", "laser_2"):
                laser_at = int(own[ROW_OF[laser_name]])
                if not bool(known[laser_at]):
                    continue
                bubble = torch.hypot(east - east[laser_at], north - north[laser_at]) <= 600.0
                outside = outside & ~bubble
            allowed = allowed & outside
        mask[slot, 1:] = allowed
    return mask


def _local(meta: list[dict]) -> dict[str, dict | None]:
    choice = {name: None for name in NAMES}
    for name in NAMES:
        pool = [item for item in meta if item["token"] is not None and _allowed(name, item["lat"], item["lon"])]
        if pool:
            choice[name] = _best(pool, name)
    return choice


def _losses(model: CoastalAllocator, scene) -> torch.Tensor:
    _frame, batch, meta, _pointer, _tasks, _chosen = scene
    threat, pointer_logits, task_logits = model(batch)
    mask = legal_mask(batch)
    local = _local(meta)
    pointer = []
    tasks = []
    for slot, name in enumerate(NAMES):
        picked = local[name]
        if picked is None or not bool(mask[slot, int(picked["token"]) + 1]):
            pointer.append(0)
            tasks.append(0)
        else:
            pointer.append(int(picked["token"]) + 1)
            tasks.append(TASKS.index(TASK_OF[name]))
    masked = pointer_logits[0].masked_fill(~mask, -1e9)
    loss = nn.functional.cross_entropy(masked, torch.tensor(pointer, dtype=torch.long))
    loss = loss + nn.functional.cross_entropy(task_logits[0], torch.tensor(tasks, dtype=torch.long))
    usable = [item for item in meta if item["token"] is not None]
    if len(usable) >= 2:
        score = torch.stack([threat[0, item["token"]] for item in usable])
        value = torch.tensor([item["threat"] for item in usable], dtype=score.dtype)
        sign = torch.sign(value[:, None] - value[None, :])
        pair = sign != 0
        ordered = -sign[pair] * (score[:, None] - score[None, :])[pair]
        loss = loss + nn.functional.softplus(ordered).mean()
    return loss


def _place(generator: torch.Generator, north: float, east: float, index: int) -> tuple[str, float, float]:
    north += float(torch.empty(1).uniform_(-25, 25, generator=generator))
    east += float(torch.empty(1).uniform_(-25, 25, generator=generator))
    lat, lon = ll(north, east)
    return (f"c{index}", lat, lon)


def _contacts(seed: int, count: int = 4) -> list[tuple[str, float, float]]:
    generator = torch.Generator().manual_seed(seed)
    kind = seed % 10
    if kind in (0, 1):
        rows = []
        for index in range(count):
            north = float(torch.empty(1).uniform_(-4500, 4500, generator=generator))
            east = float(torch.empty(1).uniform_(-4500, 4500, generator=generator))
            lat, lon = ll(north, east)
            rows.append((f"c{index}", lat, lon))
        return rows
    if kind in (2, 3, 4):
        spots = ((180.0, 0.0), (-180.0, 0.0), (520.0, 0.0), (-520.0, 0.0))
    elif kind in (5, 6, 7):
        spots = ((0.0, -1000.0), (0.0, 1000.0), (0.0, -1800.0), (0.0, 1800.0))
    elif kind == 8:
        spots = ((250.0, 80.0), (3500.0, 200.0))
    else:
        spots = ((-250.0, -80.0), (-3500.0, -200.0))
    return [_place(generator, north, east, index) for index, (north, east) in enumerate(spots)]


def _decode(logits: torch.Tensor, mask: torch.Tensor) -> list[int]:
    """Highest remaining threat in range. A laser keeps the target from the jammers."""
    hard: set[int] = set()
    laser: set[int] = set()
    jammed: set[int] = set()
    looked: set[int] = set()
    chosen = {name: 0 for name in NAMES}
    slot_of = {name: index for index, name in enumerate(NAMES)}
    for name in list(PICK_ORDER) + [name for name in NAMES if name not in PICK_ORDER]:
        slot = slot_of[name]
        score = logits[slot].clone().masked_fill(~mask[slot], -1e9)
        kind = TASK_OF.get(name, "none")
        if kind == "hard":
            ban = hard
        elif kind == "soft":
            ban = laser | jammed
        elif kind == "observe":
            ban = looked
        else:
            ban = set()
        for column in ban:
            score[column] = -1e9
        choice = int(score.argmax())
        chosen[name] = choice
        if choice == 0:
            continue
        if kind == "hard":
            hard.add(choice)
            if name.startswith("laser"):
                laser.add(choice)
        elif kind == "soft":
            jammed.add(choice)
        elif kind == "observe":
            looked.add(choice)
    return [chosen[name] for name in NAMES]


@torch.no_grad()
def predict(model: CoastalAllocator, frame) -> dict:
    batch = _batch(frame, False)
    threat, pointer_logits, _task_logits = model(batch)
    mask = legal_mask(batch)
    pointer = _decode(pointer_logits[0], mask)
    order = []
    for index, token in enumerate(frame.token_id):
        if frame.entity_valid[index] and frame.category[index][0] == TRACK_KIND:
            order.append((float(threat[0, index]), token))
    order.sort(reverse=True)
    assignments = []
    for slot, name in enumerate(NAMES):
        choice = int(pointer[slot])
        token = "" if choice == 0 else frame.token_id[choice - 1]
        task = "none" if choice == 0 else TASK_OF.get(name, "none")
        assignments.append({"platform": name, "task": task, "token": token})
    return {"ranking": [token for _score, token in order], "assignments": assignments}


def chosen_indexes(frame, assignments: list[dict]) -> list[int]:
    """Token column chosen for each platform. 0 means the network named nobody."""
    indexes = []
    for row in assignments:
        token = row.get("token") or ""
        indexes.append(0 if not token else frame.token_id.index(token) + 1)
    return indexes


def log_prob_chosen(model: CoastalAllocator, batch: dict[str, torch.Tensor], chosen: list[int]) -> torch.Tensor:
    """Mean log probability of the platforms' chosen columns. This is the reinforcement term."""
    _threat, pointer_logits, _task_logits = model(batch)
    masked = pointer_logits[0].masked_fill(~legal_mask(batch), -1e9)
    logp = torch.log_softmax(masked, dim=-1)
    index = torch.tensor(chosen, dtype=torch.long, device=logp.device)
    return logp.gather(1, index.unsqueeze(1)).squeeze(1).mean()


def _pointer_hit(model: CoastalAllocator | None, scene) -> tuple[int, int]:
    pointer = scene[3]
    if model is None:
        chosen = [0 for _ in NAMES]
    else:
        predicted = predict(model, scene[0])
        chosen = []
        for row in predicted["assignments"]:
            chosen.append(0 if not row["token"] else scene[0].token_id.index(row["token"]) + 1)
    return sum(a == b for a, b in zip(chosen, pointer)), len(NAMES)


def _accuracy(model: CoastalAllocator | None, seed: int) -> dict:
    pointer_hits = pointer_total = rank_hits = rank_total = 0
    for index in range(16):
        scene = make_scene(_contacts(seed + index))
        hits, total = _pointer_hit(model, scene)
        pointer_hits += hits
        pointer_total += total
        if model is None:
            continue
        predicted = predict(model, scene[0])
        rank = {token: pos for pos, token in enumerate(predicted["ranking"])}
        usable = [item for item in scene[2] if item["token"] is not None]
        for left in range(len(usable)):
            for right in range(left + 1, len(usable)):
                rank_total += 1
                teacher = usable[left]["threat"] > usable[right]["threat"]
                left_token = scene[0].token_id[usable[left]["token"]]
                right_token = scene[0].token_id[usable[right]["token"]]
                network = rank[left_token] < rank[right_token]
                rank_hits += int(teacher == network)
    return {
        "pointer": round(pointer_hits / max(pointer_total, 1), 4),
        "ranking_pairs": None if model is None or rank_total == 0 else round(rank_hits / rank_total, 4),
    }


def _swap_case(model: CoastalAllocator) -> dict:
    near = ll(250, 80)
    far = ll(3500, 200)
    first = make_scene([("near", *near), ("far", *far)])
    second = make_scene([("near", *far), ("far", *near)])
    before = predict(model, first[0])
    after = predict(model, second[0])
    teacher_before = {name: (row["contact"] if row else "") for name, row in zip(NAMES, first[5])}
    teacher_after = {name: (row["contact"] if row else "") for name, row in zip(NAMES, second[5])}
    followed = 0
    rows = []
    for left, right in zip(before["assignments"], after["assignments"]):
        got_before = left["token"].split(":", 1)[-1].rsplit(".", 1)[0] if left["token"] else ""
        got_after = right["token"].split(":", 1)[-1].rsplit(".", 1)[0] if right["token"] else ""
        same = got_before == teacher_before[left["platform"]] and got_after == teacher_after[right["platform"]]
        followed += int(same)
        rows.append({"platform": left["platform"], "teacher_before": teacher_before[left["platform"]], "got_before": got_before, "teacher_after": teacher_after[right["platform"]], "got_after": got_after, "matched": same})
    return {"matched_platforms": followed, "assignments": rows}


def train_one(directory: Path, seed: int, steps: int = 2000) -> dict:
    torch.manual_seed(seed)
    model = CoastalAllocator()
    before = _accuracy(None, 9000 + seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        loss = _losses(model, make_scene(_contacts(20000 * seed + step, 4 if step % 2 == 0 else 3)))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(round(float(loss.detach()), 4))
    model.eval()
    after = _accuracy(model, 9000 + seed)
    swap = _swap_case(model)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed_{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "schema": SCHEMA, "tasks": TASKS, "names": NAMES}, path)
    passed = bool(
        after["pointer"] >= 0.75
        and (after["ranking_pairs"] or 0) >= 0.75
        and swap["matched_platforms"] >= 9
    )
    return {
        "seed": seed,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "before": before,
        "after": after,
        "swap_matched": swap["matched_platforms"],
        "passed": passed,
    }


def load_allocator(path: Path) -> CoastalAllocator:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("权重不是某临海城市 12 平台分配模型")
    model = CoastalAllocator()
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
