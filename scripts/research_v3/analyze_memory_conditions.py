"""Paired fixed-threshold detection analysis for difficult roadside targets."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.matching import STATUS_TP, in_range, label_candidates, vehicle_items

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"


def frame_detection(row, gts, threshold):
    objects = vehicle_items(row["objects"])
    scores = np.asarray([x["score"] for x in objects], dtype=np.float32)
    boxes = np.asarray([x["box"] for x in objects], dtype=np.float32).reshape(-1, 7)
    keep = (scores >= threshold) & in_range(boxes)
    status, gt_index, _, _, _ = label_candidates(boxes[keep], gts)
    matched = set(gt_index[status == STATUS_TP].tolist())
    false_positive = int((status != STATUS_TP).sum())
    return matched, false_positive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.47854848529411764)
    args = ap.parse_args()
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    groups = {key: {"gt": 0, "D0": 0, "D2": 0, "D3": 0}
              for key in ("all", "near_0_30m", "mid_30_60m", "far_60m_plus",
                          "sparse_0_3_returns", "D0_fixed_threshold_miss")}
    fp = {name: 0 for name in ("D0", "D2", "D3")}
    frames = 0
    for seq in args.sequences:
        gt_rows = list(read_jsonl(CONVERTED / paths[seq]))
        sources = {name: {x["frame_id"]: x for x in read_jsonl(ROOT / path / f"{seq}.jsonl")}
                   for name, path in (("D0", args.baseline), ("D2", args.motion),
                                      ("D3", args.memory))}
        for gt_row in gt_rows:
            frame_id = gt_row["frame_id"]
            gts = np.asarray([x["box"] for x in vehicle_items(gt_row["objects"])],
                             dtype=np.float32).reshape(-1, 7)
            selected = np.where(in_range(gts))[0]
            if not len(selected):
                continue
            points = np.load(POINTS / f"{seq}_{frame_id}.npy", mmap_mode="r")[:, :2]
            xy = gts[selected, :2]
            near_points = ((points[:, None, 0] - xy[None, :, 0]) ** 2 +
                           (points[:, None, 1] - xy[None, :, 1]) ** 2 <= 4.0).sum(0)
            hits = {}
            for name in ("D0", "D2", "D3"):
                hits[name], count = frame_detection(sources[name][frame_id], gts,
                                                    args.threshold)
                fp[name] += count
            frames += 1
            for k, j in enumerate(selected):
                distance = float(np.linalg.norm(gts[j, :2]))
                labels = ["all", "near_0_30m" if distance < 30 else
                          "mid_30_60m" if distance < 60 else "far_60m_plus"]
                if near_points[k] <= 3:
                    labels.append("sparse_0_3_returns")
                if j not in hits["D0"]:
                    labels.append("D0_fixed_threshold_miss")
                for label in labels:
                    groups[label]["gt"] += 1
                    for name in ("D0", "D2", "D3"):
                        groups[label][name] += int(j in hits[name])
    summary = {label: {**counts,
                       **{f"{name}_recall": counts[name] / counts["gt"] if counts["gt"] else None
                          for name in ("D0", "D2", "D3")}}
               for label, counts in groups.items()}
    summary["unmatched_candidates"] = {"frames": frames, **fp,
                                 **{f"{name}_per_frame": fp[name] / frames
                                    for name in fp}}
    summary["notes"] = ("Fixed score threshold and IoU>=0.25 one-to-one matches. "
                        "Sparse means <=3 actual LiDAR returns within 2m of GT center. "
                        "D0-miss means no baseline match at the fixed threshold. "
                        "Unmatched candidates are not the official FP metric. "
                        "GT is used for offline stratification, never inference.")
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
