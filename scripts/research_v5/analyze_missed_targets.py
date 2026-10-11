"""Partition baseline misses into low-score and absent-candidate GT positions."""

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
from inframot3d.tcpn.matching import in_range, iou3d_matrix, vehicle_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-disabled", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.47854848529411764)
    args = parser.parse_args()

    manifest = read_json(ROOT / "data/converted/v2x_seq_infrastructure/manifest.json")
    paths = {item["sequence_id"]: item["path"] for item in manifest["sequences"]}
    groups = {name: {"gt": 0, "baseline": 0, "candidate": 0,
                     "candidate_disabled": 0}
              for name in ("all", "baseline_low_score", "baseline_no_candidate",
                           "baseline_match_conflict")}
    for sequence_id in args.sequences:
        rows = list(read_jsonl(ROOT / "data/converted/v2x_seq_infrastructure"
                               / paths[sequence_id]))
        sources = {
            name: {row["frame_id"]: row for row in read_jsonl(ROOT / path / f"{sequence_id}.jsonl")}
            for name, path in (("baseline", args.baseline),
                               ("candidate", args.candidate),
                               ("candidate_disabled", args.candidate_disabled))
        }
        for row in rows:
            frame_id = row["frame_id"]
            boxes = np.asarray([item["box"] for item in vehicle_items(row["objects"])],
                               dtype=np.float32).reshape(-1, 7)
            selected = np.where(in_range(boxes))[0]
            if len(selected) == 0:
                continue
            hit = {name: frame_detection(source[frame_id], boxes, args.threshold)[0]
                   for name, source in sources.items()}
            baseline_objects = vehicle_items(sources["baseline"][frame_id]["objects"])
            baseline_scores = np.asarray([item["score"] for item in baseline_objects],
                                         dtype=np.float32)
            baseline_boxes = np.asarray([item["box"] for item in baseline_objects],
                                        dtype=np.float32).reshape(-1, 7)
            keep = (baseline_scores >= 0.01) & in_range(baseline_boxes)
            overlaps = iou3d_matrix(baseline_boxes[keep], boxes)
            candidate_scores = baseline_scores[keep]
            for index in selected:
                labels = ["all"]
                if index not in hit["baseline"]:
                    matching_scores = candidate_scores[overlaps[:, index] >= 0.25]
                    if len(matching_scores) == 0:
                        labels.append("baseline_no_candidate")
                    elif matching_scores.max() < args.threshold:
                        labels.append("baseline_low_score")
                    else:
                        labels.append("baseline_match_conflict")
                for label in labels:
                    group = groups[label]
                    group["gt"] += 1
                    for name in sources:
                        group[name] += int(index in hit[name])
    result = {name: {**counts, **{f"{key}_recall": counts[key] / counts["gt"]
                                   if counts["gt"] else None
                                   for key in ("baseline", "candidate", "candidate_disabled")}}
              for name, counts in groups.items()}
    result["note"] = ("GT is used only for offline paired analysis. A low-score candidate "
                      "has IoU >= 0.25 and score in [0.01, fixed threshold). A no-candidate "
                      "GT has no baseline box with IoU >= 0.25 at score >= 0.01. A match "
                      "conflict has a high-score overlapping box but no one-to-one match.")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
