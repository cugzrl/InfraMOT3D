from pathlib import Path

from inframot3d.tracking.ab3dmot import MultiClassAB3DMOT
from inframot3d.tracking.immortal import MultiClassImmortalTracker
from inframot3d.tracking.simpletrack import MultiClassSimpleTrack


def create_tracker(tracker_config, root=None):
    name = str(tracker_config["name"])
    if name == "AB3DMOT":
        return MultiClassAB3DMOT(tracker_config)
    if name == "SimpleTrack":
        return MultiClassSimpleTrack(tracker_config)
    if name == "ImmortalTracker":
        return MultiClassImmortalTracker(tracker_config)
    if name == "Fast-Poly":
        from inframot3d.tracking.fastpoly_adapter import FastPolyTracker

        fastpoly_root = Path(tracker_config["fastpoly_root"])
        if not fastpoly_root.is_absolute():
            fastpoly_root = Path(root) / fastpoly_root if root is not None else fastpoly_root
        return FastPolyTracker(fastpoly_root, tracker_config["fastpoly"], tracker_config["class_names"])
    raise ValueError(f"不支持的跟踪器{name}")
