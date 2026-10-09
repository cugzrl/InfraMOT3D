"""不学习的重打分基线，候选池与 TCPN 完全相同

nms：车辆超类内类别无关 NMS，被抑制的候选分数置 0 但仍保留在候选池
iso_cls：按类别 isotonic 校准，在训练侧留出序列上拟合，再映射回原始分数尺度
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from sklearn.isotonic import IsotonicRegression

from inframot3d.io import read_jsonl, write_json, write_jsonl
from inframot3d.tcpn.calibration import ScoreMapper
from inframot3d.tcpn.data import CLASS_INDEX
from inframot3d.tcpn.matching import iou3d_matrix


def class_agnostic_nms(objects, iou_thr):
    idx = [i for i, o in enumerate(objects) if o["class_name"] in CLASS_INDEX]
    if len(idx) < 2:
        return
    boxes = np.array([objects[i]["box"] for i in idx], np.float32)
    scores = np.array([objects[i]["score"] for i in idx])
    iou = iou3d_matrix(boxes, boxes)
    order = np.argsort(-scores, kind="stable")
    removed = np.zeros(len(idx), bool)
    for a in order:
        if removed[a]:
            continue
        hit = (iou[a] > iou_thr) & ~removed
        hit[a] = False
        removed |= hit
    for k in np.where(removed)[0]:
        objects[idx[k]]["raw_score"] = objects[idx[k]]["score"]
        objects[idx[k]]["score"] = 0.0


def fit_iso_cls(sample_root, holdout):
    raw, cls, label = [], [], []
    for sequence_id in holdout:
        for s in pickle.load(open(sample_root / ("%s.pkl" % sequence_id), "rb")):
            keep = s["cand_quality"] >= 0
            raw.append(s["cand_score"][keep])
            cls.append(s["cand_cls"][keep])
            label.append(s["cand_quality"][keep])
    raw, cls, label = map(np.concatenate, (raw, cls, label))
    per_class = {}
    for c in np.unique(cls):
        m = cls == c
        per_class[int(c)] = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw[m], label[m])
    prob = np.zeros_like(raw)
    for c, model in per_class.items():
        prob[cls == c] = model.predict(raw[cls == c])
    mapper = ScoreMapper().fit(raw, prob, label)
    return per_class, mapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["nms", "iso_cls"])
    parser.add_argument("--detections", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples")
    parser.add_argument("--holdout", nargs="*")
    parser.add_argument("--nms-iou", type=float, default=0.1)
    args = parser.parse_args()
    out = ROOT / args.output
    if args.mode == "iso_cls":
        per_class, mapper = fit_iso_cls(ROOT / args.samples, args.holdout)
    for sequence_id in args.sequences:
        rows = []
        for row in read_jsonl(ROOT / args.detections / ("%s.jsonl" % sequence_id)):
            objects = [dict(o) for o in row["objects"]]
            if args.mode == "nms":
                class_agnostic_nms(objects, args.nms_iou)
            else:
                for o in objects:
                    c = CLASS_INDEX.get(o["class_name"])
                    if c is None or c not in per_class:
                        continue
                    o["raw_score"] = o["score"]
                    o["score"] = float(mapper.to_raw_scale(per_class[c].predict([o["score"]]))[0])
            rows.append(dict(row, objects=objects))
        write_jsonl(out / ("%s.jsonl" % sequence_id), rows)
    write_json(out.parent / ("%s_meta.json" % out.name), vars(args))


if __name__ == "__main__":
    main()
