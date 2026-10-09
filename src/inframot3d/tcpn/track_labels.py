"""为在线 GRAE 轨迹与当前候选生成离线监督，只在训练侧使用 GT

轨迹状态
- correct      身份纯度足够且最近一次观测就是该身份，GT 在当前帧仍存在且有候选
- unobserved   身份可靠，GT 仍存在，但当前帧没有任何达标候选
- terminated   身份可靠，但 GT 已离开
- false        最近观测全部为背景
- uncertain    身份混杂、最近一次观测模糊或刚发生切换，不参与身份监督
- dup_track    与另一条更新近的正确轨迹声明同一 GT
"""

from collections import defaultdict

import numpy as np

from inframot3d.tcpn.matching import STATUS_BG, STATUS_TP, label_candidates, vehicle_items

TRACK_STATES = ("correct", "unobserved", "terminated", "false", "uncertain", "dup_track")
VEHICLE_INDEX = (0, 1, 2, 3)
HISTORY = 5
PURITY = 0.6


def frame_candidates(det_row, gt_row):
    """返回车辆候选在原始检测列表中的下标与标签"""
    vehicle_index = [i for i, item in enumerate(det_row["objects"]) if item["class_name"] in ("Car", "Van", "Bus", "Truck")]
    dets = [det_row["objects"][i] for i in vehicle_index]
    gts = vehicle_items(gt_row["objects"])
    det_boxes = np.array([d["box"] for d in dets], dtype=np.float32).reshape(-1, 7)
    gt_boxes = np.array([g["box"] for g in gts], dtype=np.float32).reshape(-1, 7)
    status, gt_index, match_iou, max_iou, nearest = label_candidates(det_boxes, gt_boxes)
    gt_ids = [str(g["source_track_id"]) for g in gts]
    det_gt = [gt_ids[g] if g >= 0 else None for g in gt_index]
    return {
        "input_index": vehicle_index,
        "boxes": det_boxes,
        "scores": np.array([d["score"] for d in dets], dtype=np.float32),
        "classes": [d["class_name"] for d in dets],
        "status": status,
        "gt_id": det_gt,
        "iou": match_iou,
        "max_iou": max_iou,
        "gt_boxes": gt_boxes,
        "gt_ids": gt_ids,
        "gt_index": gt_index,
        "nearest": nearest,
    }


def label_sequence(det_rows, gt_rows, states):
    """逐帧生成候选标签、轨迹标签与轨迹到候选的目标"""
    obs = defaultdict(list)
    frames = []
    for det_row, gt_row, state in zip(det_rows, gt_rows, states):
        if det_row["frame_id"] != gt_row["frame_id"] or state["frame_id"] != gt_row["frame_id"]:
            raise ValueError("帧未对齐 %s" % gt_row["frame_id"])
        cand = frame_candidates(det_row, gt_row)
        input_to_local = {inp: k for k, inp in enumerate(cand["input_index"])}
        gt_present = set(cand["gt_ids"])
        tp_of_gt = {}
        for k, g in enumerate(cand["gt_id"]):
            if g is not None and cand["status"][k] == STATUS_TP:
                tp_of_gt[g] = k
        gt_box_of = {g: cand["gt_boxes"][j] for j, g in enumerate(cand["gt_ids"])}
        tracks = []
        for item in state["tracks"]:
            if item["class_index"] not in VEHICLE_INDEX:
                continue
            history = obs[item["track_id"]][-HISTORY:]
            label, identity, purity = _track_state(history)
            target = -1
            if label == "correct" or label == "pending":
                if identity in gt_present:
                    if identity in tp_of_gt:
                        label = "correct"
                        target = tp_of_gt[identity]
                    else:
                        label = "unobserved"
                else:
                    label = "terminated"
            last_time = history[-1][0] if history else None
            gt_now = gt_box_of.get(identity) if identity is not None else None
            tracks.append(
                {
                    "track_id": item["track_id"],
                    "box": item["box"],
                    "score": item["score"],
                    "age": item["age"],
                    "class_index": item["class_index"],
                    "state": label,
                    "identity": identity,
                    "purity": purity,
                    "target": target,
                    "last_time": last_time,
                    "gt_box": None if gt_now is None else gt_now.tolist(),
                    "hits": len(obs[item["track_id"]]),
                    "history": [h[2] for h in obs[item["track_id"]][-HISTORY:]],
                }
            )
        _resolve_duplicates(tracks)
        claimed = {t["target"] for t in tracks if t["state"] == "correct"}
        det_role = []
        for k in range(len(cand["status"])):
            if cand["status"][k] == STATUS_TP:
                det_role.append("matched" if k in claimed else "new")
            elif cand["status"][k] == STATUS_BG:
                det_role.append("bg")
            else:
                det_role.append("ignore")
        frames.append({"frame_id": gt_row["frame_id"], "timestamp": int(gt_row["timestamp"]), "cand": cand, "tracks": tracks, "det_role": det_role})
        for inp, (track_id, _) in state["assign"].items():
            k = input_to_local.get(int(inp))
            if k is None:
                continue
            status = int(cand["status"][k])
            gt = cand["gt_id"][k] if status == STATUS_TP else None
            obs[track_id].append((int(gt_row["timestamp"]), gt, {"box": cand["boxes"][k].tolist(), "score": float(cand["scores"][k]), "t": int(gt_row["timestamp"]), "status": status}))
    return frames


def _track_state(history):
    if not history:
        return "uncertain", None, 0.0
    ids = [h[1] for h in history]
    statuses = [h[2]["status"] for h in history]
    if all(s == STATUS_BG for s in statuses):
        return "false", None, 0.0
    tp_ids = [g for g in ids if g is not None]
    if not tp_ids:
        return "uncertain", None, 0.0
    values, counts = np.unique(tp_ids, return_counts=True)
    identity = str(values[np.argmax(counts)])
    purity = float(counts.max() / len(ids))
    if purity < PURITY or ids[-1] != identity:
        return "uncertain", identity, purity
    return "pending", identity, purity


def _resolve_duplicates(tracks):
    by_identity = defaultdict(list)
    for index, track in enumerate(tracks):
        if track["state"] in ("correct", "unobserved", "terminated") and track["identity"] is not None:
            by_identity[track["identity"]].append(index)
    for indices in by_identity.values():
        if len(indices) < 2:
            continue
        keep = max(indices, key=lambda i: (tracks[i]["last_time"] or 0, tracks[i]["hits"]))
        for i in indices:
            if i != keep:
                tracks[i]["state"] = "dup_track"
                tracks[i]["target"] = -1


def gt_queries(prev_gt_row, cand):
    """诊断上限：上一帧 GT 作为轨迹，身份完全正确"""
    tracks = []
    tp_of_gt = {}
    for k, g in enumerate(cand["gt_id"]):
        if g is not None and cand["status"][k] == STATUS_TP:
            tp_of_gt[g] = k
    present = set(cand["gt_ids"])
    for item in vehicle_items(prev_gt_row["objects"]):
        g = str(item["source_track_id"])
        if g in tp_of_gt:
            state, target = "correct", tp_of_gt[g]
        elif g in present:
            state, target = "unobserved", -1
        else:
            state, target = "terminated", -1
        tracks.append({"track_id": g, "box": item["box"], "score": 1.0, "age": 1, "state": state, "identity": g, "target": target})
    return tracks
