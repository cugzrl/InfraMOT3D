from inframot3d.tracking.ab3dmot import MultiClassAB3DMOT
from inframot3d.tracking.immortal import MultiClassImmortalTracker
from inframot3d.tracking.simpletrack import MultiClassSimpleTrack


def create_tracker(tracker_config):
    name = str(tracker_config["name"])
    if name == "AB3DMOT":
        return MultiClassAB3DMOT(tracker_config)
    if name == "SimpleTrack":
        return MultiClassSimpleTrack(tracker_config)
    if name == "ImmortalTracker":
        return MultiClassImmortalTracker(tracker_config)
    raise ValueError(f"不支持的跟踪器{name}")
