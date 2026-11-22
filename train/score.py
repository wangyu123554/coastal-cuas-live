"""Whole-battle score used by the reinforcement sketch. This file does not train and does not replace seed 3."""

from __future__ import annotations


def battle_score(blue_destroyed: int, protect_impacts: int) -> float:
    """Blue aircraft removed, minus ten for each impact on a protected point."""
    return blue_destroyed - 10 * protect_impacts
