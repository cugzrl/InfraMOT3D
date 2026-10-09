"""TCPN 负结果诊断：重复框标签、高分框被压低、校准复用、关联对象是否改变

只读已有样本、开环 dump 和闭环预测，不改 checkpoint
"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import inframot3d.tcpn.matching as matching

matching._GPU_IOU = False

from inframot3d.geometry import bev_corners
from inframot3d.io import read_jsonl
from inframot3d.tcpn.calibration import fit_from_dump
from inframot3d.tcpn.det_metrics import PRAccumulator
from inframot3d.tcpn.matching import STATUS_BG, STATUS_DUP, STATUS_IGNORE, STATUS_TP, iou3d_matrix

THR = 0.47854848529411764
OUT = ROOT / "outputs/research/roadside_motion_interaction"
V2 = ROOT / "outputs/research/roadside_joint_perception_v2"


def load_samples(path, sequences):
    rows = []
    for sequence_id in sequences:
        with (path / ("%s.pkl" % sequence_id)).open("rb") as stream:
            rows.extend(pickle.load(stream))
    return rows


def load_dump(path):
    table = {}
    if not path.exists():
        return table
    for row in read_jsonl(path):
        table[(row["sequence_id"], row["frame_id"])] = row
    return table


def draw_box(ax, box, color, text):
    corners = bev_corners(box)
    loop = np.concatenate([corners, corners[:1]], axis=0)
    ax.plot(loop[:, 0], loop[:, 1], color=color, linewidth=1.2)
    ax.text(box[0], box[1], text, color=color, fontsize=7)


def one_pipeline(name, sequences, sample_dir, pred_name):
    samples = load_samples(V2 / sample_dir, sequences)
    dump = load_dump(V2 / "checkpoints" / name / pred_name / "holdout_grae_epoch8.jsonl")
    mapper = fit_from_dump(V2 / "checkpoints" / name / pred_name / "holdout_grae_epoch8.jsonl") if dump else None
    multi = disagree = gt_with_cand = 0
    label_diff = 0
    paired = 0
    demote_tp = demote_n = 0
    promote_tp = promote_dup = promote_bg = promote_ign = promote_n = 0
    raw_acc, prob_acc, map_acc = PRAccumulator(), PRAccumulator(), PRAccumulator()
    cases = []
    for sample in samples:
        det = np.asarray(sample["cand_box"], np.float32).reshape(-1, 7)
        gt = np.asarray(sample["gt_boxes"], np.float32).reshape(-1, 7)
        score = np.asarray(sample["cand_score"], np.float64)
        status = np.asarray(sample["cand_status"])
        if len(det) == 0 or len(gt) == 0:
            continue
        iou = iou3d_matrix(det, gt)
        winners_iou = set()
        for col in range(len(gt)):
            hit = np.where(iou[:, col] >= 0.25)[0]
            if len(hit) == 0:
                continue
            gt_with_cand += 1
            if len(hit) >= 2:
                multi += 1
                if int(hit[np.argmax(iou[hit, col])]) != int(hit[np.argmax(score[hit])]):
                    disagree += 1
            winners_iou.add(int(np.argmax(np.where(iou[:, col] >= 0.25, iou[:, col], -1.0))))
        order = np.argsort(-score)
        used = np.zeros(len(gt), dtype=bool)
        winners_score = set()
        for row in order:
            cols = np.where((~used) & (iou[row] >= 0.25))[0]
            if len(cols) == 0:
                continue
            col = int(cols[np.argmax(iou[row, cols])])
            used[col] = True
            winners_score.add(int(row))
        label_diff += len(winners_iou.symmetric_difference(winners_score))
        raw_acc.add(score, det, gt)
        row = dump.get((sample["sequence_id"], sample["frame_id"]))
        if row is None or mapper is None or len(row["prob"]) != len(score):
            continue
        paired += 1
        prob = np.asarray(row["prob"], np.float64)
        mapped = mapper.to_raw_scale(prob)
        prob_acc.add(prob, det, gt)
        map_acc.add(mapped, det, gt)
        demote = (score >= THR) & (mapped < THR)
        promote = (score < THR) & (mapped >= THR)
        demote_n += int(demote.sum())
        demote_tp += int((demote & (status == STATUS_TP)).sum())
        promote_n += int(promote.sum())
        promote_tp += int((promote & (status == STATUS_TP)).sum())
        promote_dup += int((promote & (status == STATUS_DUP)).sum())
        promote_bg += int((promote & (status == STATUS_BG)).sum())
        promote_ign += int((promote & (status == STATUS_IGNORE)).sum())
        if demote.any() and (status[demote] == STATUS_TP).any():
            k = int(np.where(demote & (status == STATUS_TP))[0][np.argmax(score[demote & (status == STATUS_TP)] - mapped[demote & (status == STATUS_TP)])])
            cases.append({"drop": float(score[k] - mapped[k]), "sequence_id": sample["sequence_id"], "frame_id": sample["frame_id"], "det": det, "gt": gt, "score": score, "mapped": mapped, "status": status, "focus": k})
    pred_root = V2 / "predictions" / name / "holdout"
    assoc = compare_tracks(sequences, samples, pred_root / "raw/predictions", pred_root / pred_name / "predictions")
    metrics = {}
    for key in ("raw", pred_name):
        path = pred_root / key / "tracking/metrics.json"
        if path.exists():
            payload = json.loads(path.read_text())
            metrics[key] = {"HOTA": payload["hota"]["HOTA"], "DetA": payload["hota"]["DetA"], "AssA": payload["hota"]["AssA"], "IDF1": payload["official"]["IDF1"], "IDSW": payload["official"]["IDSW"]}
    cases = sorted(cases, key=lambda item: -item["drop"])[:8]
    return {
        "frames": len(samples),
        "dump_frames": paired,
        "gt_with_candidate": gt_with_cand,
        "multi_candidate_gt": multi,
        "multi_fraction": multi / max(gt_with_cand, 1),
        "iou_vs_score_winner_disagree": disagree,
        "disagree_fraction_of_multi": disagree / max(multi, 1),
        "label_set_disagreements": label_diff,
        "demote_below_thr": demote_n,
        "demote_tp": demote_tp,
        "promote_above_thr": promote_n,
        "promote_tp": promote_tp,
        "promote_dup": promote_dup,
        "promote_bg": promote_bg,
        "promote_ignore": promote_ign,
        "pr": {"raw": raw_acc.summary(), "prob": prob_acc.summary(), "mapped_on_holdout": map_acc.summary()},
        "association": assoc,
        "tracking": metrics,
        "cases": cases,
    }


def match_tracks(objects, gt):
    boxes = np.array([item["box"] for item in objects], np.float32).reshape(-1, 7) if objects else np.zeros((0, 7), np.float32)
    gt = np.asarray(gt, np.float32).reshape(-1, 7)
    if len(boxes) == 0 or len(gt) == 0:
        return {}
    iou = iou3d_matrix(boxes, gt)
    cost = np.where(iou >= 0.25, -iou, 1.0)
    rows, cols = linear_sum_assignment(cost)
    return {int(col): int(objects[row]["track_id"]) for row, col in zip(rows, cols) if iou[row, col] >= 0.25}


def compare_tracks(sequences, samples, raw_root, new_root):
    by_frame = {(s["sequence_id"], s["frame_id"]): s for s in samples}
    changed = compared = 0
    raw_ids, new_ids = set(), set()
    raw_births = new_births = 0
    seen_raw, seen_new = {}, {}
    for sequence_id in sequences:
        raw_path = raw_root / ("%s.jsonl" % sequence_id)
        new_path = new_root / ("%s.jsonl" % sequence_id)
        if not raw_path.exists() or not new_path.exists():
            continue
        for raw_row, new_row in zip(read_jsonl(raw_path), read_jsonl(new_path)):
            sample = by_frame.get((sequence_id, raw_row["frame_id"]))
            if sample is None:
                continue
            raw_map = match_tracks(raw_row["objects"], sample["gt_boxes"])
            new_map = match_tracks(new_row["objects"], sample["gt_boxes"])
            keys = set(raw_map) | set(new_map)
            compared += len(keys)
            changed += sum(raw_map.get(k) != new_map.get(k) for k in keys)
            for item in raw_row["objects"]:
                tid = int(item["track_id"])
                raw_ids.add((sequence_id, tid))
                if (sequence_id, tid) not in seen_raw:
                    seen_raw[(sequence_id, tid)] = True
                    raw_births += 1
            for item in new_row["objects"]:
                tid = int(item["track_id"])
                new_ids.add((sequence_id, tid))
                if (sequence_id, tid) not in seen_new:
                    seen_new[(sequence_id, tid)] = True
                    new_births += 1
    return {"gt_slots": compared, "association_changed": changed, "changed_fraction": changed / max(compared, 1), "raw_tracks": len(raw_ids), "tcpn_tracks": len(new_ids), "raw_births": raw_births, "tcpn_births": new_births}


def save_cases(name, cases):
    fig_dir = OUT / "figures" / name
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {STATUS_TP: "tab:green", STATUS_DUP: "tab:orange", STATUS_IGNORE: "tab:gray", STATUS_BG: "tab:red"}
    for index, case in enumerate(cases):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for box in case["gt"]:
            draw_box(ax, box, "black", "GT")
        order = np.argsort(-case["score"])[:12]
        for k in order:
            draw_box(ax, case["det"][k], colors[int(case["status"][k])], "%.2f→%.2f" % (case["score"][k], case["mapped"][k]))
        draw_box(ax, case["det"][case["focus"]], "blue", "focus")
        ax.set_aspect("equal")
        ax.set_title("%s %s drop %.2f" % (case["sequence_id"], case["frame_id"], case["drop"]))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.tight_layout()
        fig.savefig(fig_dir / ("case_%02d.png" % index), dpi=120)
        plt.close(fig)


def check_hota_math():
    from inframot3d.evaluation.hota import _sequence_counts

    gt = np.array([0])
    tr = np.array([0])
    sim = np.array([[0.8]])
    empty_gt = np.array([], dtype=np.int64)
    fp_only = np.array([0])
    frames_skip = [(gt, tr, sim)]
    frames_keep = [(gt, tr, sim), (empty_gt, fp_only, np.zeros((0, 1)))]
    # 空 GT 帧若被跳过，不会增加 FP
    skipped = _sequence_counts(frames_skip, 1, 1, [0.5])[0]["FP"]
    # 直接把空帧送进计数，FP 应增加
    kept = _sequence_counts(frames_keep, 1, 1, [0.5])[0]["FP"]
    return {"empty_frame_fp_if_skipped": int(skipped), "empty_frame_fp_if_kept": int(kept), "difference": int(kept - skipped)}


def main():
    folds = json.loads((ROOT / "configs/research/v2xseq_train_folds.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    jobs = {
        "pA": ("detA", folds["inner_holdout"]["B"]),
        "pB": ("detB", folds["inner_holdout"]["A"]),
    }
    summary = {"threshold": THR, "calibration_note": "mapped_on_holdout 使用同一 holdout dump 拟合，属于上一轮的数据复用", "hota_empty_frame": check_hota_math()}
    for name, (sample_dir, sequences) in jobs.items():
        print("diagnose", name, flush=True)
        result = one_pipeline(name, sequences, "samples/" + sample_dir, "hist_bev_qgrae_s0")
        cases = result.pop("cases")
        save_cases(name, cases)
        result["figure_cases"] = len(cases)
        summary[name] = result
        print(json.dumps({k: result[k] for k in ("multi_fraction", "demote_tp", "promote_dup", "association")}, ensure_ascii=False), flush=True)
    (OUT / "diagnostics" / "tcpn_failure.json").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostics" / "tcpn_failure.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "diagnostics/tcpn_failure.json")


if __name__ == "__main__":
    main()
