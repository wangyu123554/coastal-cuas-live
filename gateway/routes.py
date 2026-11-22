"""Write the selected blue routes into the scenario text."""

from __future__ import annotations

import re

POLLER = """
execute at_interval_of 0.2 sec
   FileIO net = FileIO();
   if (net.Open(write_str("net_", gNetSeen + 1, ".txt"), "in"))
   {
      string actor = net.ReadToken();
      string verb = net.ReadToken();
      string wantName = net.ReadToken();
      double wantRaw = (double) net.ReadToken();
      net.Close();
      gNetSeen = gNetSeen + 1;
      if (actor != "" && verb != "") CT_Apply(actor, verb, wantName, wantRaw);
   }
end_execute
"""

_IMPACT_TYPES = """weapon_effects OPSIM_IMPACT_EFFECT WSF_GRADUATED_LETHALITY
   radius_and_pk 250 m 1.0
end_weapon_effects
weapon OPSIM_IMPACT WSF_IMPLICIT_WEAPON
   quantity 1
   update_interval 1 sec
   weapon_effects OPSIM_IMPACT_EFFECT
end_weapon
"""


def install_poller(text: str) -> str | None:
    """Replace the distance-gate block with the file gateway. None means the gate was not found."""
    start = text.find("execute at_interval_of 2 sec")
    end = text.find("end_execute", start)
    if start < 0 or end < 0 or "CT_Pick" not in text[start:end]:
        return None
    return text[:start] + POLLER + text[end + len("end_execute") :]


def _signed(value: float, positive: str, negative: str) -> str:
    return f"{abs(float(value)):.6f}{positive if float(value) >= 0 else negative}"


def _waypoint(point: dict) -> str:
    text = f"      position {_signed(point['lat'], 'n', 's')} {_signed(point['lon'], 'e', 'w')}"
    if point.get("altitude_m") is not None:
        text += f" altitude {float(point['altitude_m']):g} m"
    if point.get("speed_mps") is not None:
        text += f" speed {float(point['speed_mps']):g} m/s"
    return text


def _route_block(points: list[dict], impact_name: str | None) -> str:
    lines = ["   route"]
    for index, point in enumerate(points):
        line = _waypoint(point)
        if impact_name and index == len(points) - 1:
            line += f"\n         execute {impact_name}"
        lines.append(line)
    lines.append("   end_route")
    return "\n".join(lines)


def _impact_script(name: str) -> list[str]:
    return [
        f"   script void {name}()",
        "      int bestIndex = -1;",
        "      double best = 250.0 * 250.0;",
        "      for (int j = 0; j < WsfSimulation.PlatformCount(); j = j + 1)",
        "      {",
        "         WsfPlatform target = WsfSimulation.PlatformEntry(j);",
        '         if (target.IsValid() && target.Side() == "gold")',
        "         {",
        "            double north = (target.Latitude() - PLATFORM.Latitude()) * 111320.0;",
        "            double east = (target.Longitude() - PLATFORM.Longitude()) * 111320.0 * MATH.Cos(PLATFORM.Latitude());",
        "            double up = target.Altitude() - PLATFORM.Altitude();",
        "            double dist2 = north * north + east * east + up * up;",
        "            if (dist2 <= best)",
        "            {",
        "               best = dist2;",
        "               bestIndex = j;",
        "            }",
        "         }",
        "      }",
        "      if (bestIndex >= 0)",
        "      {",
        "         WsfPlatform hit = WsfSimulation.PlatformEntry(bestIndex);",
        '         WsfWeapon impact = PLATFORM.Weapon("impact");',
        "         if (hit.IsValid() && impact.IsValid() && impact.QuantityRemaining() > 0) impact.Fire(hit.MakeTrack());",
        "      }",
        "      PLATFORM.SetDamageFactor(1.0);",
        "      PLATFORM.DeletePlatform();",
        "   end_script",
    ]


def apply_blue(text: str, plan: dict) -> str:
    """Replace baked routes with the selected blue plan. Later waves reuse the same direction and slot."""
    routes: dict[str, dict] = {}
    delays: dict[str, float] = {}
    for task in plan.get("tasks") or []:
        if task.get("primitive") == "MOVE_ROUTE" and task.get("waypoints"):
            routes[task["actor"]] = task
        elif task.get("primitive") == "DELAYED_START" and float(task.get("start_sec") or 0) > 0:
            delays[task["actor"]] = float(task["start_sec"])
    if not routes:
        return text
    if any(task.get("end") == "impact" for task in routes.values()) and "weapon OPSIM_IMPACT " not in text:
        text = _IMPACT_TYPES + "\n" + text
    expanded = dict(routes)
    for name, task in list(routes.items()):
        match = re.fullmatch(r"blue_([nsew])1_([abc])", name)
        if not match:
            continue
        direction, slot = match.groups()
        for wave in range(2, 7):
            expanded.setdefault(f"blue_{direction}{wave}_{slot}", task)
    for direction in "nsew":
        donor = expanded.get(f"blue_{direction}1_a")
        if donor is None:
            continue
        for wave in range(1, 7):
            expanded.setdefault(f"blue_{direction}{wave}_c", donor)
    for name, task in expanded.items():
        found = re.search(rf"(platform {re.escape(name)} CT_UAV\n)(.*?\nend_platform)", text, re.S)
        if not found:
            continue
        body = found.group(2)
        points = list(task.get("waypoints") or [])
        impact_name = f"opsim_impact_{name}" if task.get("end") == "impact" else None
        body = re.sub(r"   route\n.*?   end_route", _route_block(points, impact_name), body, count=1, flags=re.S)
        if impact_name and f"script void {impact_name}" not in body:
            body = body.replace(
                "\nend_platform",
                "\n   add weapon impact OPSIM_IMPACT\n      quantity 1\n   end_weapon\n" + "\n".join(_impact_script(impact_name)) + "\nend_platform",
                1,
            )
        if name in delays:
            body = re.sub(r"creation_time [0-9.]+ sec", f"creation_time {delays[name]:g} sec", body, count=1)
        last = points[-1]
        body = re.sub(r'SetAuxData\("dest_lat", [^)]+\)', f'SetAuxData("dest_lat", {float(last["lat"]):.6f})', body, count=1)
        body = re.sub(r'SetAuxData\("dest_lon", [^)]+\)', f'SetAuxData("dest_lon", {float(last["lon"]):.6f})', body, count=1)
        if last.get("altitude_m") is not None:
            body = re.sub(r'SetAuxData\("dest_alt", [^)]+\)', f'SetAuxData("dest_alt", {float(last["altitude_m"]):.1f})', body, count=1)
        text = text[: found.start()] + found.group(1) + body + text[found.end() :]
    return text
