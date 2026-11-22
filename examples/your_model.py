"""Replace only the assignment function. The gateway still writes the command files.

Input: one packed frame. It already holds the 12 posts and the recent rotating-radar
tracks, in relative east/north.

Output: the same dict the built-in network returns.

    {"assignments": [{"platform": "laser_1", "token": "radar_2:blue_n1_a.1", "task": "hard"}, ...]}

An empty token means that post does nothing this tick. Do not invent task kinds.
laser and hunter and net use hard, jam uses soft, eo uses observe, the two radars use none.
"""

from __future__ import annotations

from network.allocate import load_allocator, predict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = load_allocator(_ROOT / "network" / "weights" / "seed_3.pt")


def assign(frame) -> dict:
    # Swap this body for your own model. Keep the return shape.
    return predict(_MODEL, frame)
