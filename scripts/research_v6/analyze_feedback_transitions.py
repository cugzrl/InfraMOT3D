"""Count detections gained and lost by online history at a frozen threshold."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/research_v3"))

from analyze_memory_conditions import frame_detection
from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.matching import in_range, vehicle_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--feedback-on", required=True)
    parser.add_argument("--feedback-off", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.47854848529411764)
    args = parser.parse_args()
    converted = ROOT / "data/converted/v2x_seq_infrastructure"
    paths = {item["sequence_id"]: item["path"]
             for item in read_json(converted / "manifest.json")["sequences"]}
    names = ("baseline", "feedback_on", "feedback_off")
    counts = {group: {"gt": 0, **{name: 0 for name in names},
                      "on_only_vs_off": 0, "off_only_vs_on": 0,
                      "baseline_miss_rescued_on": 0, "baseline_hit_lost_on": 0}
              for group in ("all", "near_0_30m", "mid_30_60m", "far_60m_plus")}
    for sequence in args.sequences:
        sources = {name: {row["frame_id"]: row
                          for row in read_jsonl(ROOT / path / f"{sequence}.jsonl")}
                   for name, path in (("baseline", args.baseline),
                                      ("feedback_on", args.feedback_on),
                                      ("feedback_off", args.feedback_off))}
        for row in read_jsonl(converted / paths[sequence]):
            boxes = np.asarray([item["box"] for item in vehicle_items(row["objects"])],
                               dtype=np.float32).reshape(-1, 7)
            selected = np.where(in_range(boxes))[0]
            if not len(selected):
                continue
            hits = {name: frame_detection(sources[name][row["frame_id"]],
                                          boxes, args.threshold)[0]
                    for name in names}
            for index in selected:
                distance = float(np.linalg.norm(boxes[index, :2]))
                group = ("near_0_30m" if distance < 30 else
                         "mid_30_60m" if distance < 60 else "far_60m_plus")
                for key in ("all", group):
                    result = counts[key]
                    result["gt"] += 1
                    for name in names:
                        result[name] += int(index in hits[name])
                    result["on_only_vs_off"] += int(
                        index in hits["feedback_on"] and index not in hits["feedback_off"])
                    result["off_only_vs_on"] += int(
                        index in hits["feedback_off"] and index not in hits["feedback_on"])
                    result["baseline_miss_rescued_on"] += int(
                        index not in hits["baseline"] and index in hits["feedback_on"])
                    result["baseline_hit_lost_on"] += int(
                        index in hits["baseline"] and index not in hits["feedback_on"])
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
