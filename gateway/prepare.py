"""Stage one live run: scenario, selected blue routes, and the file gateway."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from gateway.routes import apply_blue, install_poller

ROOT = Path(__file__).resolve().parents[1]


def prepare(red_path: Path, blue_path: Path, work: Path | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = work or (ROOT / "runs" / f"live-{stamp}")
    dest.mkdir(parents=True, exist_ok=True)
    red = json.loads(red_path.read_text(encoding="utf-8"))
    blue = json.loads(blue_path.read_text(encoding="utf-8"))
    text = (ROOT / "scenario" / "main.txt").read_text(encoding="utf-8")
    text = apply_blue(text, blue)
    patched = install_poller(text)
    if patched is None:
        raise SystemExit("场景里没有找到原来的距离门，接口网关没有接上。")
    (dest / "main.txt").write_text(patched, encoding="utf-8")
    shutil.copyfile(ROOT / "scenario" / "coastal_types.txt", dest / "coastal_types.txt")
    (dest / "run.txt").write_text("realtime\ninclude_once main.txt\n", encoding="utf-8")
    (dest / "selected.json").write_text(
        json.dumps({"red": red.get("title"), "blue": blue.get("title")}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return dest
