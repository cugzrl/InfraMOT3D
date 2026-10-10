"""Training-side candidate bottleneck and deliberately optimistic tracking ceilings.

The oracle outputs are diagnostics only: they use current-frame GT to select
existing detection boxes and assign identities. They must never be treated as
an online tracker or a deployable result.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.evaluation.protocols.v2xseq import V2XSeqProtocol
from inframot3d.tcpn.matching import MATCH_IOU, iou3d_matrix, vehicle_items

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
FROZEN_THRESHOLD = 0.47854848529411764
TRACKER_FLOOR = 0.1
PROTOCOL = V2XSeqProtocol(ROOT)


def best_pairs(boxes_a, boxes_b):
    """One-to-one IoU assignment; keys are indices into the original inputs."""
    iou = iou3d_matrix(boxes_a, boxes_b)
    if not iou.size:
        return {}, iou
    cost = np.where(iou >= MATCH_IOU, -iou, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return {int(c): int(r) for r, c in zip(rows, cols) if iou[r, c] >= MATCH_IOU}, iou


def box_array(items):
    return np.asarray([x["box"] for x in items], dtype=np.float32).reshape(-1, 7)


def distance_bin(box):
    r = float(np.linalg.norm(np.asarray(box[:2], dtype=float)))
    if r < 30:
        return "0-30m"
    if r < 60:
        return "30-60m"
    return "60m+"


def count_run_lengths(rows):
    lengths = defaultdict(list)
    for track_rows in rows.values():
        prev = None
        run = 0
        for frame_index, category in track_rows:
            failure = category != "covered"
            if failure and prev is not None and frame_index == prev + 1:
                run += 1
            elif failure:
                if run:
                    lengths["failure"].append(run)
                run = 1
            elif run:
                lengths["failure"].append(run)
                run = 0
            prev = frame_index
        if run:
            lengths["failure"].append(run)
    values = lengths["failure"]
    return {"runs": len(values), "mean_frames": float(np.mean(values)) if values else 0.0,
            "p90_frames": float(np.percentile(values, 90)) if values else 0.0,
            "max_frames": max(values, default=0)}


def diagnose_sequence(seq, gt_path, det_root, replay_root, out_root):
    gt_rows = list(read_jsonl(CONVERTED / gt_path))
    det_rows = list(read_jsonl(det_root / f"{seq}.jsonl"))
    pred_rows = list(read_jsonl(replay_root / "predictions" / f"{seq}.jsonl"))
    state_rows = list(read_jsonl(replay_root / "states" / f"{seq}.jsonl"))
    assert len(gt_rows) == len(det_rows) == len(pred_rows) == len(state_rows)
    counts = Counter()
    by_distance = defaultdict(Counter)
    by_class = defaultdict(Counter)
    track_slots = defaultdict(list)
    last_identity = {}
    oracle_raw, oracle_unit = [], []
    identity_map = {}
    collision_ids = set()
    collision_frames = 0
    collision_slots = 0
    for row in gt_rows:
        ids = Counter(str(x["source_track_id"]) for x in vehicle_items(row["objects"])
                      if PROTOCOL.inside_range(x["box"]))
        repeated = {tid for tid, n in ids.items() if n > 1}
        collision_ids.update(repeated)
        collision_frames += bool(repeated)
        collision_slots += sum(ids[tid] for tid in repeated)
    for frame_idx, (gt_row, det_row, pred_row, state_row) in enumerate(zip(gt_rows, det_rows, pred_rows, state_rows)):
        fid = gt_row["frame_id"]
        assert fid == det_row["frame_id"] == pred_row["frame_id"] == state_row["frame_id"]
        gts = vehicle_items(gt_row["objects"])
        gts = [g for g in gts if PROTOCOL.inside_range(g["box"])]
        dets = vehicle_items(det_row["objects"])
        dets = [d for d in dets if PROTOCOL.inside_range(d["box"])]
        pred = [p for p in vehicle_items(pred_row["objects"])
                if p["score"] >= FROZEN_THRESHOLD and PROTOCOL.inside_range(p["box"])]
        selected, det_iou = best_pairs(box_array(dets), box_array(gts))
        published, _ = best_pairs(box_array(pred), box_array(gts))
        raw_objects, unit_objects = [], []
        for gi, gt in enumerate(gts):
            tid = str(gt["source_track_id"])
            identity_map.setdefault(tid, len(identity_map))
            candidate_indices = np.where(det_iou[:, gi] >= MATCH_IOU)[0] if det_iou.size else []
            valid_scores = [float(dets[i]["score"]) for i in candidate_indices]
            best_score = max(valid_scores, default=0.0)
            pi = published.get(gi)
            if not len(candidate_indices):
                category = "no_candidate"
            elif best_score < TRACKER_FLOOR:
                category = "below_tracker_floor"
            elif best_score < FROZEN_THRESHOLD and pi is None:
                category = "below_frozen_output_threshold"
            elif pi is None:
                category = "candidate_not_published"
            elif tid in collision_ids:
                category = "ambiguous_gt_identity"
            else:
                observed_id = int(pred[pi]["track_id"])
                previous = last_identity.get(tid)
                if previous is not None and previous[0] != observed_id:
                    if previous[1] < frame_idx - 1:
                        category = "identity_reentry_after_gap"
                    else:
                        assigned = state_row["assign"]
                        stage = next((a[1] for a in assigned.values() if int(a[0]) == observed_id), None)
                        category = "identity_rebirth_adjacent" if stage == "birth" else "identity_change_adjacent"
                else:
                    category = "covered"
                last_identity[tid] = (observed_id, frame_idx)
            counts[category] += 1
            by_distance[distance_bin(gt["box"])][category] += 1
            by_class[gt["class_name"]][category] += 1
            if tid not in collision_ids:
                track_slots[tid].append((frame_idx, category))
            di = selected.get(gi)
            if di is not None:
                d = dets[di]
                base = {"class_name": d["class_name"], "track_id": identity_map[tid], "box": d["box"]}
                raw_objects.append({**base, "score": float(d["score"])})
                unit_objects.append({**base, "score": 1.0})
        head = {k: det_row[k] for k in ("sequence_id", "frame_index", "frame_id", "timestamp")}
        oracle_raw.append({**head, "objects": raw_objects})
        oracle_unit.append({**head, "objects": unit_objects})
    write_jsonl(out_root / "oracle_raw_score" / f"{seq}.jsonl", oracle_raw)
    write_jsonl(out_root / "oracle_unit_score" / f"{seq}.jsonl", oracle_unit)
    return {"slots": sum(counts.values()), "categories": dict(counts),
            "distance": {k: dict(v) for k, v in by_distance.items()},
            "class": {k: dict(v) for k, v in by_class.items()},
            "failure_runs": count_run_lengths(track_slots),
            "gt_identity_collisions": {"ids": sorted(collision_ids),
                                       "frames": collision_frames, "slots": collision_slots},
            "oracle_selected_boxes": sum(len(row["objects"]) for row in oracle_raw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--sequences", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    out = ROOT / args.output
    results = {}
    for seq in args.sequences:
        results[seq] = diagnose_sequence(seq, paths[seq], ROOT / args.detections,
                                         ROOT / args.replay, out)
        print(seq, results[seq]["slots"], results[seq]["categories"], flush=True)
    total = Counter()
    for row in results.values():
        total.update(row["categories"])
    payload = {"git_sha": None, "detections": args.detections, "replay": args.replay,
               "sequences": args.sequences, "frozen_threshold": FROZEN_THRESHOLD,
               "tracker_floor": TRACKER_FLOOR, "total_categories": dict(total),
               "total_slots": sum(total.values()), "per_sequence": results,
               "oracle_note": "GT selects existing candidates and supplies track IDs. Diagnostic upper bound only."}
    write_json(out / "diagnosis.json", payload)


if __name__ == "__main__":
    main()
