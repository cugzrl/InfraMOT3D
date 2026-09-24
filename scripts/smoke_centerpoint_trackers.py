import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation import evaluate
from inframot3d.io import read_json, read_jsonl
from inframot3d.tracking import create_tracker
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint


def _check_rows(name, rows, gt_rows):
    if len(rows) != len(gt_rows):
        raise SystemExit("%s帧数不一致" % name)
    seen_reset = rows[0]["objects"]
    for row, gt_row in zip(rows, gt_rows):
        if row["frame_id"] != gt_row["frame_id"] or int(row["timestamp"]) != int(gt_row["timestamp"]):
            raise SystemExit("%s帧未对齐" % name)
        if row["sequence_id"] != gt_row["sequence_id"] or int(row["frame_index"]) != int(gt_row["frame_index"]):
            raise SystemExit("%s序号未对齐" % name)
        ids = []
        for item in row["objects"]:
            box = np.asarray(item["box"], dtype=np.float64)
            if box.shape != (7,) or not np.isfinite(box).all() or not np.isfinite(item["score"]):
                raise SystemExit("%s输出无效" % name)
            if "source_track_id" in item and item["source_track_id"] not in (None, ""):
                raise SystemExit("%s输出带有source_track_id" % name)
            ids.append(int(item["track_id"]))
        if len(ids) != len(set(ids)):
            raise SystemExit("%s同一帧track_id重复" % name)
    return seen_reset


def _run_classic(config_path, sequence_id, gt_rows):
    config = load_config(config_path)
    tracker = create_tracker(config["tracker"])
    detection_root = Path(config["input"]["detection_root"])
    if not detection_root.is_absolute():
        detection_root = config["_root"] / detection_root
    rows = []
    for det_row, gt_row in zip(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)), gt_rows):
        objects = tracker.update(det_row["objects"], timestamp=det_row["timestamp"])
        rows.append(
            {
                "sequence_id": det_row["sequence_id"],
                "frame_index": det_row["frame_index"],
                "frame_id": det_row["frame_id"],
                "timestamp": det_row["timestamp"],
                "objects": objects,
            }
        )
    _check_rows(config["tracker"]["name"], rows, gt_rows)
    output_root = Path(config["project"]["output_root"])
    from inframot3d.io import write_jsonl

    write_jsonl(output_root / "predictions" / ("%s.jsonl" % sequence_id), rows)
    summary, _ = evaluate(
        config["project"]["converted_root"],
        output_root / "predictions",
        output_root,
        config["evaluation"]["iou_thresholds"],
        [sequence_id],
    )
    print(config["tracker"]["name"], "MOTA", round(summary["overall"]["mota"], 4))
    return rows


def _run_grae(sequence_id, gt_rows):
    config = load_config("configs/grae_centerpoint.yaml")
    root = config["_root"]
    best = Path(config["project"]["output_root"]) / "ckpt" / "checkpoint-best.pth"
    ckpt = [best] if best.is_file() else sorted(
        (Path(config["project"]["output_root"]) / "ckpt").glob("checkpoint-epoch*.pth"),
        key=lambda path: int(path.stem.replace("checkpoint-epoch", "")),
    )
    if not ckpt:
        raise SystemExit("缺少GRAE checkpoint")
    model = build_model(
        root / config["project"]["grae_root"],
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        "cuda",
    )
    load_checkpoint(model, ckpt[-1], "cuda")
    threshold_path = root / config["tracker"]["score_thresholds_file"]
    score_thresholds = yaml.safe_load(threshold_path.read_text(encoding="utf-8"))["score_thresholds"]
    tracker = GraeTracker(
        model,
        config["classes"],
        score_thresholds,
        alpha=config["tracker"]["alpha"],
        age=config["tracker"]["age"],
    )
    detection_root = root / config["input"]["detection_root"]
    rows = []
    tracker.reset()
    for det_row in read_jsonl(detection_root / ("%s.jsonl" % sequence_id)):
        for item in det_row["objects"]:
            if "source_track_id" in item:
                raise SystemExit("检测结果含有source_track_id")
        objects = tracker.update(det_row["objects"], int(det_row["timestamp"]) / 1e6, det_row["frame_id"])
        rows.append(
            {
                "sequence_id": det_row["sequence_id"],
                "frame_index": det_row["frame_index"],
                "frame_id": det_row["frame_id"],
                "timestamp": det_row["timestamp"],
                "objects": objects,
            }
        )
    tracker.reset()
    first_det = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))[0]
    again = tracker.update(first_det["objects"], int(first_det["timestamp"]) / 1e6, first_det["frame_id"])
    if again and min(item["track_id"] for item in again) != 0:
        raise SystemExit("sequence重置后track_id未从0开始")
    _check_rows("GRAE-3DMOT", rows, gt_rows)
    from inframot3d.io import write_jsonl

    output_root = Path(config["project"]["output_root"])
    write_jsonl(output_root / "predictions" / ("%s.jsonl" % sequence_id), rows)
    summary, _ = evaluate(
        config["project"]["converted_root"],
        output_root / "predictions",
        output_root,
        config["evaluation"]["iou_thresholds"],
        [sequence_id],
    )
    print("GRAE-3DMOT", "MOTA", round(summary["overall"]["mota"], 4), "ckpt", ckpt[-1].name)


def main():
    root = Path(__file__).resolve().parents[1]
    split = read_json(root / "configs/v2xseq_sequence_split.json")
    manifest = read_json(root / "data/converted/v2x_seq_infrastructure/manifest.json")
    val_ids = set(split["val"])
    entries = [item for item in manifest["sequences"] if item["sequence_id"] in val_ids]
    entry = min(entries, key=lambda item: item["num_frames"])
    sequence_id = entry["sequence_id"]
    gt_rows = list(read_jsonl(root / "data/converted/v2x_seq_infrastructure" / entry["path"]))
    manifest_det = read_json(root / "outputs/centerpoint/detection_manifest.json")
    if abs(float(manifest_det["score_threshold"]) - 0.01) > 1e-6:
        raise SystemExit("检测导出阈值不是0.01")
    low_score = None
    for item in manifest_det["sequences"]:
        if item["split"] != "val":
            continue
        for row in read_jsonl(root / "outputs/centerpoint" / item["path"]):
            for obj in row["objects"]:
                score = float(obj["score"])
                if low_score is None or score < low_score:
                    low_score = score
                if low_score < 0.1:
                    break
            if low_score is not None and low_score < 0.1:
                break
        if low_score is not None and low_score < 0.1:
            break
    if low_score is None or low_score >= 0.1:
        raise SystemExit("检测没有保留0.1以下分数")
    print("smoke序列", sequence_id, "帧", len(gt_rows), "val最低分", low_score)
    _run_classic(root / "configs/ab3dmot_centerpoint.yaml", sequence_id, gt_rows)
    _run_classic(root / "configs/simpletrack_centerpoint.yaml", sequence_id, gt_rows)
    _run_classic(root / "configs/immortal_centerpoint.yaml", sequence_id, gt_rows)
    _run_grae(sequence_id, gt_rows)
    print("smoke_ok")


if __name__ == "__main__":
    main()
