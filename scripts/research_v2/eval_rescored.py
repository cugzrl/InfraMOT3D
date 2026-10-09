"""比较原始分数与新分数的检测质量，候选池完全相同

在正式输出阈值处统计：被找回的低分 TP、被压低的高分 TP、新增与去除的 FP，以及按距离分段的召回
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.det_metrics import PRAccumulator, greedy_flags
from inframot3d.tcpn.matching import in_range, vehicle_items

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
THR = 0.47854848529411764
DIST_BINS = [(0, 30), (30, 60), (60, 120)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescored", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=THR)
    args = parser.parse_args()
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {e["sequence_id"]: e["path"] for e in manifest["sequences"]}
    acc_raw, acc_new = PRAccumulator(), PRAccumulator()
    counts = {"recovered_tp": 0, "suppressed_tp": 0, "added_fp": 0, "removed_fp": 0, "kept_tp": 0, "kept_fp": 0}
    by_dist = {"%d-%d" % b: {"gt": 0, "raw_tp": 0, "new_tp": 0, "raw_fp": 0, "new_fp": 0} for b in DIST_BINS}
    per_seq = {}
    for sequence_id in args.sequences:
        gt_rows = {r["frame_id"]: r for r in read_jsonl(CONVERTED / paths[sequence_id])}
        sr, sn = PRAccumulator(), PRAccumulator()
        for row in read_jsonl(ROOT / args.rescored / ("%s.jsonl" % sequence_id)):
            dets = vehicle_items(row["objects"])
            gts = vehicle_items(gt_rows[row["frame_id"]]["objects"])
            boxes = np.array([d["box"] for d in dets], np.float32).reshape(-1, 7)
            gt_boxes = np.array([g["box"] for g in gts], np.float32).reshape(-1, 7)
            raw = np.array([d.get("raw_score", d["score"]) for d in dets], np.float64)
            new = np.array([d["score"] for d in dets], np.float64)
            for acc, s in ((acc_raw, raw), (acc_new, new), (sr, raw), (sn, new)):
                acc.add(s, boxes, gt_boxes)
            keep = in_range(boxes)
            gkeep = in_range(gt_boxes)
            boxes_k, raw_k, new_k, gt_k = boxes[keep], raw[keep], new[keep], gt_boxes[gkeep]
            hr = raw_k >= args.threshold
            hn = new_k >= args.threshold
            fr, _ = greedy_flags(raw_k[hr], boxes_k[hr], gt_k)
            fn, _ = greedy_flags(new_k[hn], boxes_k[hn], gt_k)
            tr = np.zeros(len(boxes_k), bool)
            tn = np.zeros(len(boxes_k), bool)
            tr[np.where(hr)[0][fr == 1]] = True
            tn[np.where(hn)[0][fn == 1]] = True
            counts["recovered_tp"] += int((tn & ~hr).sum())
            counts["suppressed_tp"] += int((tr & ~hn).sum())
            counts["kept_tp"] += int((tr & tn).sum())
            counts["added_fp"] += int((hn & ~tn & ~hr).sum())
            counts["removed_fp"] += int((hr & ~tr & ~hn).sum())
            counts["kept_fp"] += int((hr & ~tr & hn & ~tn).sum())
            dist_d = np.hypot(boxes_k[:, 0], boxes_k[:, 1])
            dist_g = np.hypot(gt_k[:, 0], gt_k[:, 1])
            for lo, hi in DIST_BINS:
                key = "%d-%d" % (lo, hi)
                m = (dist_d >= lo) & (dist_d < hi)
                by_dist[key]["gt"] += int(((dist_g >= lo) & (dist_g < hi)).sum())
                by_dist[key]["raw_tp"] += int((tr & m).sum())
                by_dist[key]["new_tp"] += int((tn & m).sum())
                by_dist[key]["raw_fp"] += int((hr & ~tr & m).sum())
                by_dist[key]["new_fp"] += int((hn & ~tn & m).sum())
        per_seq[sequence_id] = {"raw_AP": sr.summary()["AP"], "new_AP": sn.summary()["AP"], "raw_R@0.25": sr.summary()["recall@fp0.25"], "new_R@0.25": sn.summary()["recall@fp0.25"]}
    for v in by_dist.values():
        v["raw_recall"] = v["raw_tp"] / max(v["gt"], 1)
        v["new_recall"] = v["new_tp"] / max(v["gt"], 1)
    result = {
        "raw": acc_raw.summary(),
        "new": acc_new.summary(),
        "raw_at_thr": acc_raw.recall_at(args.threshold),
        "new_at_thr": acc_new.recall_at(args.threshold),
        "changes_at_thr": counts,
        "by_distance_at_thr": by_dist,
        "per_sequence": per_seq,
        "curve_raw": acc_raw.curve(),
        "curve_new": acc_new.curve(),
    }
    out = ROOT / args.output
    data = json.loads(out.read_text()) if out.exists() else {}
    data[args.name] = result
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1))
    print(args.name, json.dumps({k: result[k] for k in ("raw", "new", "raw_at_thr", "new_at_thr", "changes_at_thr")}, indent=1))


if __name__ == "__main__":
    main()
