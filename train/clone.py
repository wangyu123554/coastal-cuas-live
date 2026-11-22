"""Retrain a copy from the rule teacher. Does not replace seed 3."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from network.allocate import train_one


def main() -> None:
    out = ROOT / "train" / "out"
    out.mkdir(parents=True, exist_ok=True)
    card = train_one(out, seed=3, steps=2000)
    kept = card.get("passed")
    print(f"指针 {card['after']['pointer']}  排序对 {card['after']['ranking_pairs']}  对调跟对 {card['swap_matched']}")
    if kept:
        print(f"过了考核桌。新权重在 {out}，上场用的种子 3 没有被替换。")
    else:
        print("没有过考核桌。不要换上这套权重。种子 3 仍在 network/weights/seed_3.pt。")


if __name__ == "__main__":
    main()
