"""从内部训练轨迹重建运动库，并补上车道、模式和匀速残差

重复时间采样或未来点落到锚点及以前的窗口不入库
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research_knowledge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LaneIndex,
    load_cached_tracks,
    load_config,
    make_sample,
    output_dir,
    project_path,
    sample_window,
    save_bank,
    stack_bank,
    to_local,
    write_json,
)
from kv import LaneGraph, augment_bank  # noqa: E402


def window_fault(track, anchor, hist_index, fut_index, fut_ok):
    times = track["time"]
    if len(np.unique(hist_index)) != len(hist_index):
        return "duplicate_history"
    if np.any(times[hist_index] > anchor + 1e-6):
        return "history_after_anchor"
    if np.any(fut_ok & (times[fut_index] <= anchor + 1e-9)):
        return "future_not_after_anchor"
    if np.any(fut_ok) and len(np.unique(fut_index[fut_ok])) != int(fut_ok.sum()):
        return "duplicate_future"
    return ""


def build(config):
    out = output_dir(config)
    tracks = load_cached_tracks(project_path(config, config["source_tracks"]))
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), project_path(config, config["lane_cache"]))
    lane_graph = LaneGraph(project_path(config, config["project"]["map_root"]))
    vehicles = set(config["vehicle_classes"])
    history_s = float(config["history_s"])
    tolerance = float(config["sample_tolerance_s"])
    grouped = defaultdict(list)
    faults = defaultdict(int)
    for track in tracks:
        if track["role"] != "inner_train" or track["class_name"] not in vehicles or track["split"] != "train":
            continue
        times = track["time"]
        for position in range(len(times)):
            anchor = float(times[position])
            if anchor - float(times[0]) < history_s - tolerance:
                continue
            hist_index, hist_ok, fut_index, fut_ok = sample_window(
                track, anchor, history_s, float(config["future_s"]), float(config["dt_s"]), tolerance
            )
            if not np.all(hist_ok) or int(hist_index[-1]) != position or not np.all(fut_ok):
                continue
            fault = window_fault(track, anchor, hist_index, fut_index, fut_ok)
            if fault:
                faults[fault] += 1
                continue
            sample = make_sample(track, position, hist_index, lanes, config)
            future = to_local(track["world_xy"][fut_index], sample["origin"], sample["heading"])
            grouped[(track["fold"], track["intersection"])].append(
                {
                    "origin": sample["origin"],
                    "heading": sample["heading"],
                    "hist": sample["hist"],
                    "future": future.astype(np.float64),
                    "speed": sample["speed"],
                    "yaw_rate": sample["yaw_rate"],
                    "time": anchor,
                    "region": sample["region"],
                    "class_name": track["class_name"],
                    "sequence_id": track["sequence_id"],
                    "uid": track["uid"],
                }
            )
    motion = out / "kb" / "motion"
    motion.mkdir(parents=True, exist_ok=True)
    summary = []
    for (fold_name, intersection), records in sorted(grouped.items()):
        bank = augment_bank(stack_bank(records), intersection, lane_graph, config)
        path = motion / ("fold%s_%s.npz" % (fold_name, intersection))
        save_bank(path, bank)
        summary.append(
            {
                "fold": fold_name,
                "intersection": intersection,
                "samples": int(len(records)),
                "tracks": int(len(set(bank["uid"].tolist()))),
                "modes": {name: int(np.sum(bank["mode"] == code)) for code, name in {0: "static", 1: "straight", 2: "left", 3: "right"}.items()},
            }
        )
        print("运动库 %s 样本 %d 轨迹 %d" % (path.name, len(records), summary[-1]["tracks"]), flush=True)
    write_json(out / "kb" / "build_audit.json", {"dropped": dict(faults), "banks": summary})
    return summary


if __name__ == "__main__":
    build(load_config(Path(__file__).resolve().parent / "config.yaml"))
