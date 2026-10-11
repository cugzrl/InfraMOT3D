"""Compare causal detector outputs at fixed 3D IoU and score thresholds."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.det_metrics import PRAccumulator
from inframot3d.tcpn.matching import vehicle_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("NAME", "DETECTIONS"))
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--iou", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--score-threshold", type=float, default=0.47854848529411764)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    converted = ROOT / "data/converted/v2x_seq_infrastructure"
    paths = {item["sequence_id"]: item["path"]
             for item in read_json(converted / "manifest.json")["sequences"]}
    gt_rows = {
        seq: {row["frame_id"]: row for row in read_jsonl(converted / paths[seq])}
        for seq in args.sequences
    }
    results = {}
    for name, directory in args.run:
        accumulators = {threshold: PRAccumulator(iou_thr=threshold)
                        for threshold in args.iou}
        for seq in args.sequences:
            for row in read_jsonl(ROOT / directory / f"{seq}.jsonl"):
                det = vehicle_items(row["objects"])
                gt = vehicle_items(gt_rows[seq][row["frame_id"]]["objects"])
                scores = np.asarray([item["score"] for item in det], dtype=np.float64)
                boxes = np.asarray([item["box"] for item in det], dtype=np.float32).reshape(-1, 7)
                gt_boxes = np.asarray([item["box"] for item in gt], dtype=np.float32).reshape(-1, 7)
                for accumulator in accumulators.values():
                    accumulator.add(scores, boxes, gt_boxes)
        results[name] = {
            str(threshold): {**accumulator.summary(),
                             "fixed": accumulator.recall_at(args.score_threshold)}
            for threshold, accumulator in accumulators.items()
        }
        print(name, {key: {"AP": value["AP"], "recall": value["fixed"]["recall"]}
                     for key, value in results[name].items()}, flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"sequences": args.sequences, "runs": results}, indent=2))


if __name__ == "__main__":
    main()
