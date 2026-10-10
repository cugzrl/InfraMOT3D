"""Backward-compatible D3 gate import for the research scripts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.perception.scene_memory import StaticGate


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())
