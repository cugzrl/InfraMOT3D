"""统计 GT Query 与 GRAE 在线 Query 的状态分布、位置误差和标签覆盖率"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.matching import STATUS_TP, in_range
from inframot3d.tcpn.track_labels import TRACK_STATES, gt_queries, label_sequence

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"


def gt_rows_of(sequence_id):
    manifest = read_json(CONVERTED / "manifest.json")
    entry = next(item for item in manifest["sequences"] if item["sequence_id"] == sequence_id)
    return list(read_jsonl(CONVERTED / entry["path"]))


def cv_predict(history, t):
    if len(history) < 2:
        return np.asarray(history[-1]["box"][:2]) if history else None
    a, b = history[-2], history[-1]
    dt = (b["t"] - a["t"]) / 1e6
    if dt <= 1e-3:
        return np.asarray(b["box"][:2])
    v = (np.asarray(b["box"][:2]) - np.asarray(a["box"][:2])) / dt
    speed = np.linalg.norm(v)
    if speed > 40:
        v = v * 40 / speed
    return np.asarray(b["box"][:2]) + v * (t - b["t"]) / 1e6


def summarize(frames_all, kind):
    states = Counter()
    pos_err_zero = []
    pos_err_cv = []
    per_frame = []
    ages = []
    scores = Counter()
    claimed_conflict = 0
    tp_total = 0
    tp_claimed = 0
    near_competitor = 0
    for frames in frames_all:
        for frame in frames:
            tracks = frame["tracks"] if kind == "grae" else frame["gt_tracks"]
            tracks = [t for t in tracks if in_range(np.asarray(t["box"]).reshape(1, 7))[0]]
            per_frame.append(len(tracks))
            targets = Counter()
            for t in tracks:
                states[t["state"]] += 1
                ages.append(t.get("age", 1))
                if t["state"] == "correct":
                    targets[t["target"]] += 1
                    gt_box = frame["cand"]["gt_boxes"][frame["cand"]["gt_ids"].index(t["identity"])]
                    pos_err_zero.append(float(np.linalg.norm(np.asarray(t["box"][:2]) - gt_box[:2])))
                    if kind == "grae" and t.get("history"):
                        pred = cv_predict(t["history"], frame["timestamp"])
                        pos_err_cv.append(float(np.linalg.norm(pred - gt_box[:2])))
            claimed_conflict += sum(1 for v in targets.values() if v > 1)
            cand = frame["cand"]
            tp = [k for k in range(len(cand["status"])) if cand["status"][k] == STATUS_TP and in_range(cand["boxes"][k:k + 1])[0]]
            tp_total += len(tp)
            tp_claimed += sum(1 for k in tp if targets.get(k, 0) > 0)
            centers = np.array([t["box"][:2] for t in tracks]).reshape(-1, 2)
            for t in tracks:
                if t["state"] != "correct" or len(centers) < 2:
                    continue
                d = np.linalg.norm(centers - np.asarray(t["box"][:2]), axis=1)
                near_competitor += int(np.sum((d > 1e-3) & (d < 4.0)) > 0)
    total = sum(states.values())
    definite = total - states["uncertain"]
    pz = np.asarray(pos_err_zero)
    pc = np.asarray(pos_err_cv)
    out = {
        "tracks": int(total),
        "tracks_per_frame": float(np.mean(per_frame)) if per_frame else 0.0,
        "state_fraction": {s: states[s] / max(total, 1) for s in TRACK_STATES},
        "label_coverage": definite / max(total, 1),
        "pos_err_zero_motion": _q(pz),
        "pos_err_const_velocity": _q(pc) if len(pc) else None,
        "age_ge_2_fraction": float(np.mean(np.asarray(ages) >= 2)) if ages else 0.0,
        "tp_candidates_claimed_by_track": tp_claimed / max(tp_total, 1),
        "multi_track_same_target": int(claimed_conflict),
        "correct_with_neighbor_within_4m": near_competitor / max(states["correct"], 1),
    }
    return out


def _q(values):
    if len(values) == 0:
        return None
    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "gt_1m": float(np.mean(values > 1.0)),
        "gt_2m": float(np.mean(values > 2.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames_all = []
    for sequence_id in args.sequences:
        gt_rows = gt_rows_of(sequence_id)
        det_rows = list(read_jsonl(ROOT / args.detections / ("%s.jsonl" % sequence_id)))
        states = list(read_jsonl(ROOT / args.replay / "states" / ("%s.jsonl" % sequence_id)))
        frames = label_sequence(det_rows, gt_rows, states)
        for index, frame in enumerate(frames):
            frame["gt_tracks"] = gt_queries(gt_rows[index - 1], frame["cand"]) if index > 0 else []
        frames_all.append(frames)
    payload = {"grae": summarize(frames_all, "grae"), "gt": summarize(frames_all, "gt")}
    out = ROOT / args.output
    data = json.loads(out.read_text()) if out.exists() else {}
    data[args.name] = payload
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
