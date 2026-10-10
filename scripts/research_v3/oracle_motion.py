"""P0 ideal next-position oracle for GRAE; GT is diagnostic input only.

The zero-motion control uses the same adapter path as the oracle, making a
replay-level equivalence check possible without changing the baseline class.
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))
sys.path.insert(0, str(ROOT / "scripts" / "research_v2"))

from replay_grae import build_tracker
from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tcpn.matching import MATCH_IOU, iou3d_matrix, vehicle_items

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"


def matches(pred, gt):
    iou = iou3d_matrix([x["box"] for x in pred], [x["box"] for x in gt])
    if not iou.size:
        return {}
    cost = np.where(iou >= MATCH_IOU, -iou, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if iou[r, c] >= MATCH_IOU}


def predict_override(self, timestamp):
    self.track.ct = self.track.observed_xy.clone()
    self.track.translation[:, :2] = self.track.observed_xy
    for index in range(len(self.track)):
        track_id = int(self.track.instance_inds[index, 0].item())
        if track_id not in self.oracle_xy:
            continue
        xy = torch.as_tensor(self.oracle_xy[track_id], dtype=self.track.ct.dtype, device=self.device)
        self.track.ct[index] = xy
        self.track.translation[index, :2] = xy
        self.oracle_use_count += 1


def run_sequence(tracker, det_rows, gt_rows, mode):
    tracker.reset()
    tracker.oracle_xy = {}
    tracker.oracle_use_count = 0
    id_to_gt = {}
    output = []
    for det_row, gt_row in zip(det_rows, gt_rows):
        assert det_row["frame_id"] == gt_row["frame_id"]
        gt = vehicle_items(gt_row["objects"])
        if mode == "gt":
            by_id = {str(x["source_track_id"]): x["box"][:2] for x in gt}
            tracker.oracle_xy = {track_id: by_id[gt_id] for track_id, gt_id in id_to_gt.items()
                                 if gt_id in by_id}
        else:
            tracker.oracle_xy = {}
        objects = tracker.update(det_row["objects"], int(det_row["timestamp"]) / 1e6,
                                 det_row["frame_id"])
        pred = vehicle_items(objects)
        for pi, gi in matches(pred, gt).items():
            id_to_gt[int(pred[pi]["track_id"])] = str(gt[gi]["source_track_id"])
        output.append({k: det_row[k] for k in ("sequence_id", "frame_index", "frame_id", "timestamp")}
                      | {"objects": objects})
    return output, tracker.oracle_use_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("zero", "gt"), required=True)
    ap.add_argument("--detections", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/trackers/grae/centerpoint.yaml")
    ap.add_argument("--ckpt", default="outputs/grae_centerpoint/ckpt/checkpoint-best.pth")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tracker = build_tracker(cfg, ROOT / args.ckpt)
    tracker.motion_mode = "constant_velocity"
    tracker._predict_tracks = types.MethodType(predict_override, tracker)
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    out = ROOT / args.output
    used = 0
    frames = 0
    for seq in args.sequences:
        det_rows = list(read_jsonl(ROOT / args.detections / f"{seq}.jsonl"))
        gt_rows = list(read_jsonl(CONVERTED / paths[seq]))
        if len(det_rows) != len(gt_rows):
            raise ValueError(f"Frame count mismatch {seq}")
        rows, count = run_sequence(tracker, det_rows, gt_rows, args.mode)
        write_jsonl(out / f"{seq}.jsonl", rows)
        used += count
        frames += len(rows)
        print(seq, len(rows), "oracle_track_positions", count, flush=True)
    write_json(out / "manifest.json", {"mode": args.mode, "detections": args.detections,
               "sequences": args.sequences, "frames": frames, "oracle_track_positions": used,
               "warning": "GT is used to correct pre-association track positions in gt mode. Diagnostic ceiling only."})


if __name__ == "__main__":
    main()
