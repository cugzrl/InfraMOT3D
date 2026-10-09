"""检测候选诊断：样本内外差距、分数区间召回、定位误差上限、重复框"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.geometry import wrap_angle
from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.det_metrics import PRAccumulator
from inframot3d.tcpn.matching import STATUS_NAMES, in_range, label_candidates, vehicle_items

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"


def load_gt(sequence_id):
    manifest = read_json(CONVERTED / "manifest.json")
    entry = next(item for item in manifest["sequences"] if item["sequence_id"] == sequence_id)
    return list(read_jsonl(CONVERTED / entry["path"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", default="outputs/centerpoint/detections")
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    det_root = ROOT / args.detections
    acc = PRAccumulator()
    status_count = {name: 0 for name in STATUS_NAMES}
    status_by_bin = {}
    bins = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.01]
    loc = {"center": [], "yaw": [], "iou": [], "size": [], "score": []}
    cross_class_dup = 0
    per_frame_candidates = []
    gt_best_score = []
    for sequence_id in args.sequences:
        gt_rows = load_gt(sequence_id)
        det_rows = {row["frame_id"]: row for row in read_jsonl(det_root / ("%s.jsonl" % sequence_id))}
        for gt_row in gt_rows:
            det_row = det_rows[gt_row["frame_id"]]
            dets = vehicle_items(det_row["objects"])
            gts = vehicle_items(gt_row["objects"])
            det_boxes = np.array([d["box"] for d in dets], dtype=np.float32).reshape(-1, 7)
            gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32).reshape(-1, 7)
            scores = np.array([d["score"] for d in dets], dtype=np.float64)
            acc.add(scores, det_boxes, gt_boxes)
            keep = in_range(det_boxes)
            per_frame_candidates.append(int(keep.sum()))
            status, gt_index, match_iou, max_iou, nearest = label_candidates(det_boxes, gt_boxes)
            names = [d["class_name"] for d in dets]
            for i in np.where(keep)[0]:
                status_count[STATUS_NAMES[status[i]]] += 1
                b = int(np.searchsorted(bins, scores[i], side="right") - 1)
                key = "%.2f-%.2f" % (bins[b], bins[b + 1])
                status_by_bin.setdefault(key, {name: 0 for name in STATUS_NAMES})
                status_by_bin[key][STATUS_NAMES[status[i]]] += 1
                if status[i] == 1 and gt_index[status == 0].size:
                    owner = np.where((status == 0) & (gt_index == nearest[i]))[0]
                    if owner.size and names[owner[0]] != names[i]:
                        cross_class_dup += 1
                if status[i] == 0:
                    g = gt_boxes[gt_index[i]]
                    loc["center"].append(float(np.linalg.norm(det_boxes[i, :2] - g[:2])))
                    yaw = abs(wrap_angle(float(det_boxes[i, 3] - g[3])))
                    loc["yaw"].append(min(yaw, np.pi - yaw))
                    loc["iou"].append(float(match_iou[i]))
                    loc["size"].append(float(np.abs(det_boxes[i, 4:7] - g[4:7]).sum()))
                    loc["score"].append(float(scores[i]))
            gin = in_range(gt_boxes)
            for j in np.where(gin)[0]:
                hits = np.where(gt_index == j)[0]
                gt_best_score.append(float(scores[hits].max()) if hits.size else 0.0)
    summary = acc.summary()
    summary["recall_at"] = [acc.recall_at(t) for t in (0.1, 0.2, 0.3, 0.4, 0.4785, 0.6)]
    loc_np = {k: np.asarray(v) for k, v in loc.items()}
    high = loc_np["score"] >= 0.4785
    summary["localization_tp"] = {
        "count": int(len(loc_np["center"])),
        "center_median": float(np.median(loc_np["center"])),
        "center_p90": float(np.percentile(loc_np["center"], 90)),
        "yaw_median_deg": float(np.degrees(np.median(loc_np["yaw"]))),
        "iou_median": float(np.median(loc_np["iou"])),
        "frac_iou_ge_0.5": float((loc_np["iou"] >= 0.5).mean()),
        "frac_iou_ge_0.7": float((loc_np["iou"] >= 0.7).mean()),
        "frac_center_lt_0.3": float((loc_np["center"] < 0.3).mean()),
        "high_score_center_median": float(np.median(loc_np["center"][high])) if high.any() else None,
        "low_score_center_median": float(np.median(loc_np["center"][~high])) if (~high).any() else None,
        "low_score_iou_median": float(np.median(loc_np["iou"][~high])) if (~high).any() else None,
    }
    best = np.asarray(gt_best_score)
    summary["gt_best_candidate_score"] = {
        "gt": int(len(best)),
        "no_candidate_iou25": float((best == 0).mean()),
        "only_lt_0.1": float(((best > 0) & (best < 0.1)).mean()),
        "0.1_to_thr": float(((best >= 0.1) & (best < 0.4785)).mean()),
        "ge_thr": float((best >= 0.4785).mean()),
    }
    summary["status_in_range"] = status_count
    summary["status_by_score_bin"] = status_by_bin
    summary["cross_class_duplicates"] = int(cross_class_dup)
    summary["candidates_per_frame_mean"] = float(np.mean(per_frame_candidates))
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(out.read_text()) if out.exists() else {}
    payload[args.name] = summary
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(args.name, json.dumps({k: v for k, v in summary.items() if k not in ("status_by_score_bin",)}, indent=1))


if __name__ == "__main__":
    main()
