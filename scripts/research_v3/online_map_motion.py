"""P1: predict GT future from causal GRAE histories, with honest coverage.

The current GT match only supplies a training/evaluation target. Track input is
exclusively the online GRAE output up to the anchor time. No future GT position
or GT identity enters the predictor feature vector.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))
sys.path.insert(0, str(ROOT / "scripts" / "research_v2"))

import map_motion as mm
from inframot3d.io import read_json, read_jsonl, write_json
from inframot3d.tcpn.matching import MATCH_IOU, iou3d_matrix, vehicle_items

mm.HORIZONS = (0.1, 0.5, 1.0, 2.0)
CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"


def match_objects(pred, gt):
    matrix = iou3d_matrix([x["box"] for x in pred], [x["box"] for x in gt])
    if not matrix.size:
        return {}
    cost = np.where(matrix >= MATCH_IOU, -matrix, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if matrix[r, c] >= MATCH_IOU}


def build_online_samples(sequences, replay_roots, paths, infos, maps):
    samples = []
    coverage = Counter()
    for seq in sequences:
        roots = [root for root in replay_roots if (root / f"{seq}.jsonl").exists()]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one OOF replay for {seq}, got {len(roots)}")
        gt_rows = list(read_jsonl(CONVERTED / paths[seq]))
        pred_rows = list(read_jsonl(roots[0] / f"{seq}.jsonl"))
        if [r["frame_id"] for r in gt_rows] != [r["frame_id"] for r in pred_rows]:
            raise ValueError(f"Frame mismatch in {seq}")
        collision_ids = set()
        for row in gt_rows:
            frame_ids = Counter(str(x["source_track_id"]) for x in vehicle_items(row["objects"]))
            collision_ids.update(tid for tid, n in frame_ids.items() if n > 1)
        gt_tracks, loc, center = mm.load_tracks(seq, paths, infos)
        graph_key = (loc, tuple(np.round(center, 0)))
        if graph_key not in maps:
            maps[graph_key] = mm.LaneGraph(ROOT / "data/v2x_seq_maps" / f"{loc}.json", center)
        graph = maps[graph_key]
        history = defaultdict(list)
        for gt_row, pred_row in zip(gt_rows, pred_rows):
            frame_id = gt_row["frame_id"]
            info = infos[seq][str(frame_id)]
            pose = read_json(ROOT / "data/v2x-seq-infrastructure" / info["calib_virtuallidar_to_world_path"])
            rot = np.asarray(pose["rotation"], dtype=np.float64)
            trans = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
            yaw_offset = np.arctan2(rot[1, 0], rot[0, 0])
            t = int(gt_row["timestamp"]) / 1e6
            gt = vehicle_items(gt_row["objects"])
            pred = vehicle_items(pred_row["objects"])
            matches = match_objects(pred, gt)
            coverage["all_gt_anchors"] += len(gt)
            coverage["all_online_queries"] += len(pred)
            coverage["matched_online_queries"] += len(matches)
            for pi, obj in enumerate(pred):
                track_id = int(obj["track_id"])
                box = obj["box"]
                world = rot @ np.asarray(box[:3], dtype=np.float64) + trans
                history[track_id].append((t, world[0], world[1], mm.wrap(box[3] + yaw_offset)))
                if pi not in matches:
                    continue
                gt_id = str(gt[matches[pi]]["source_track_id"])
                if gt_id in collision_ids:
                    coverage["ambiguous_gt_identity_queries_skipped"] += 1
                    continue
                arr = np.asarray(history[track_id], dtype=np.float64)
                past = mm.interp_track(arr, t + mm.PAST)
                if past is None:
                    continue
                coverage["history_1s_available"] += 1
                truth = gt_tracks.get(gt_id)
                if truth is None:
                    continue
                fut = {}
                for h in mm.HORIZONS:
                    pair = mm.interp_track(truth, np.asarray([t, t + h]))
                    if pair is not None:
                        fut[h] = pair[1]
                        coverage[f"future_{h:.1f}s_available"] += 1
                if not fut:
                    continue
                v = (past[-1] - past[-6]) / 0.5
                speed = float(np.linalg.norm(v))
                yaw = mm.wrap(box[3] + yaw_offset)
                if speed > 1.0 and np.cos(np.arctan2(v[1], v[0]) - yaw) < 0:
                    yaw = mm.wrap(yaw + np.pi)
                heading = float(np.arctan2(v[1], v[0])) if speed > 1.0 else yaw
                samples.append({"seq": seq, "loc": loc, "graph": graph_key,
                                "past": past, "fut": fut, "v": v,
                                "speed": speed, "heading": heading,
                                "junction": graph.in_junction(past[-1]),
                                "query_score": float(obj["score"])})
    for sample in samples:
        sample["lane_feat"], sample["pred_lane"], sample["pred_branches"], sample["lane_ok"] = mm.lane_features(sample, maps)
    return samples, dict(coverage)


def errors_for_heuristics(samples):
    errors = {"stationary": [], "cv": [], "lane_straight": [], "lane_oracle_min": []}
    for sample in samples:
        p0 = sample["past"][-1]
        for name in errors:
            errors[name].append([])
        for j, h in enumerate(mm.HORIZONS):
            if h not in sample["fut"]:
                for name in errors:
                    errors[name][-1].append(None)
                continue
            gt = sample["fut"][h]
            cv = p0 + sample["v"] * h
            candidates = [cv] + [branch[j] for branch in sample["pred_branches"]]
            points = {"stationary": p0, "cv": cv,
                      "lane_straight": sample["pred_lane"][j] if sample["pred_lane"] is not None else cv}
            for name, point in points.items():
                errors[name][-1].append(float(np.linalg.norm(point - gt)))
            errors["lane_oracle_min"][-1].append(float(min(np.linalg.norm(p - gt) for p in candidates)))
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    folds = read_json(ROOT / "configs/research/v2xseq_train_folds.json")
    train = folds["inner_train"]["A"] + folds["inner_train"]["B"]
    holdout = folds["inner_holdout"]["A"] + folds["inner_holdout"]["B"]
    replay_roots = [ROOT / "outputs/research/roadside_joint_perception_v2/replays/detA_oof/predictions",
                    ROOT / "outputs/research/roadside_joint_perception_v2/replays/detB_oof/predictions"]
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    infos = {}
    for item in read_json(ROOT / "data/v2x-seq-infrastructure/data_info.json"):
        infos.setdefault(str(item["sequence_id"]), {})[str(item["frame_id"])] = item
    maps = {}
    train_samples, train_coverage = build_online_samples(train, replay_roots, paths, infos, maps)
    val_samples, val_coverage = build_online_samples(holdout, replay_roots, paths, infos, maps)
    print("samples", len(train_samples), len(val_samples), "coverage", val_coverage, flush=True)
    if not train_samples or not val_samples:
        raise RuntimeError("No online history samples")
    errors = errors_for_heuristics(val_samples)
    ytr, mtr = mm.targets(train_samples)
    yva, mva = mm.targets(val_samples)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    for use_map in (False, True):
        xtr = mm.feature_matrix(train_samples, maps, use_map)
        xva = mm.feature_matrix(val_samples, maps, use_map)
        mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
        net = mm.train_mlp((xtr - mu) / sd, ytr, mtr, args.seed, epochs=args.epochs)
        with torch.no_grad():
            pred = net(torch.from_numpy((xva - mu) / sd)).view(-1, len(mm.HORIZONS), 2).numpy()
        err = np.linalg.norm(pred - yva, axis=-1)
        name = "mlp_map" if use_map else "mlp_nomap"
        errors[name] = [[float(err[i, j]) if mva[i, j] else None for j in range(len(mm.HORIZONS))]
                        for i in range(len(val_samples))]
        torch.save({"state_dict": net.state_dict(), "mean": mu, "std": sd,
                    "seed": args.seed, "epochs": args.epochs, "use_map": use_map,
                    "train_sequences": train, "holdout_sequences": holdout}, out / f"{name}.pth")
    maneuver = [mm.maneuver(s) for s in val_samples]
    groups = {"all": lambda i: True,
              "low_query_score": lambda i: val_samples[i]["query_score"] < 0.47854848529411764,
              "high_query_score": lambda i: val_samples[i]["query_score"] >= 0.47854848529411764,
              "in_junction": lambda i: val_samples[i]["junction"]}
    for kind in ("straight", "left", "right", "static"):
        groups[kind] = (lambda name: lambda i: maneuver[i] == name)(kind)
    payload = {"train_sequences": train, "holdout_sequences": holdout,
               "samples": {"train": len(train_samples), "holdout": len(val_samples)},
               "coverage": {"train": train_coverage, "holdout": val_coverage},
               "horizons": mm.HORIZONS, "seed": args.seed, "epochs": args.epochs,
               "lane_match_rate_holdout": float(np.mean([s["lane_ok"] for s in val_samples])),
               "maneuver_counts_holdout": dict(Counter(maneuver)),
               "errors": mm.summarize(errors, groups),
               "warning": "GT is used to select supervised anchors and future labels. The lane_oracle_min branch choice uses future GT and is diagnostic only. Coverage is conditional on a 1 s online track history."}
    write_json(out / "metrics.json", payload)
    for name, groups_out in payload["errors"].items():
        print(name, {h: groups_out["all"][h]["mean"] for h in groups_out["all"]}, flush=True)


if __name__ == "__main__":
    main()
