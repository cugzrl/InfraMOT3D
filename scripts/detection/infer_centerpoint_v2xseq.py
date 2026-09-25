import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.detection.metrics import evaluate_frames
from inframot3d.detection.openpcdet_adapter import load_openpcdet_cfg, prediction_to_object
from inframot3d.detection.v2x_seq_converter import load_sequence_split
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl


class _Logger:
    def info(self, message):
        print(message)


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _latest_checkpoint(openpcdet_root):
    ckpt_dir = openpcdet_root / "output" / "v2x_seq_models" / "centerpoint" / "default" / "ckpt"
    checkpoints = sorted(ckpt_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError("未找到checkpoint %s" % ckpt_dir)
    return checkpoints[-1]


def _predict_split(cfg, checkpoint, split_name, batch_size, workers):
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu

    cfg.DATA_CONFIG.INFO_PATH["test"] = ["infos/v2x_seq_infos_%s.pkl" % split_name]
    logger = _Logger()
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False,
    )
    if len(dataset) == 0:
        return {}
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(checkpoint), logger=logger)
    model.cuda().eval()
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            annos = dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES)
            for anno in annos:
                objects = [
                    prediction_to_object(name, score, box)
                    for name, score, box in zip(anno["name"], anno["score"], anno["boxes_lidar"])
                ]
                predictions[str(anno["frame_id"])] = objects
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detectors/centerpoint/v2xseq.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    openpcdet_root = _resolve(root, config["project"]["openpcdet_root"])
    checkpoint = Path(args.ckpt) if args.ckpt else _latest_checkpoint(openpcdet_root)
    cfg = load_openpcdet_cfg(_resolve(root, config["project"]["openpcdet_cfg"]))
    if args.score_threshold is not None:
        cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = float(args.score_threshold)
        cfg.MODEL.POST_PROCESSING.SCORE_THRESH = float(args.score_threshold)
    split = load_sequence_split(_resolve(root, config["split_file"]))
    allowed = set(args.sequences) if args.sequences else None
    selected = ["train", "val", "test"] if args.split == "all" else [args.split]
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    output_root = Path(config["project"]["output_root"])
    detection_root = output_root / "detections"
    written = []
    for split_name in selected:
        sequence_ids = set(split[split_name])
        if allowed is not None:
            sequence_ids &= allowed
        if not sequence_ids:
            print("%s序列0 帧0" % split_name)
            continue
        predictions = _predict_split(cfg, checkpoint, split_name, args.batch_size, args.workers)
        for entry in manifest["sequences"]:
            sequence_id = entry["sequence_id"]
            if sequence_id not in sequence_ids:
                continue
            rows = []
            for frame in read_jsonl(converted_root / entry["path"]):
                lidar_idx = "%s_%s" % (frame["sequence_id"], frame["frame_id"])
                if lidar_idx not in predictions:
                    raise KeyError("缺少预测 %s" % lidar_idx)
                rows.append(
                    {
                        "sequence_id": frame["sequence_id"],
                        "frame_index": frame["frame_index"],
                        "frame_id": frame["frame_id"],
                        "timestamp": frame["timestamp"],
                        "objects": predictions[lidar_idx],
                    }
                )
            relative = Path("detections") / ("%s.jsonl" % sequence_id)
            write_jsonl(output_root / relative, rows)
            written.append({"sequence_id": sequence_id, "num_frames": len(rows), "path": relative.as_posix(), "split": split_name})
            print("导出序列%s 帧%d" % (sequence_id, len(rows)))
    detection_manifest = {
        "detector": "CenterPoint",
        "checkpoint": str(checkpoint),
        "score_threshold": float(cfg.MODEL.POST_PROCESSING.SCORE_THRESH),
        "coordinate_system": "virtual_lidar",
        "box_order": ["x", "y", "z", "yaw", "length", "width", "height"],
        "sequences": written,
    }
    write_json(output_root / "detection_manifest.json", detection_manifest)
    if args.skip_eval or not args.eval_split:
        return
    eval_ids = {item["sequence_id"] for item in written if item["split"] == args.eval_split}
    frames = []
    for entry in manifest["sequences"]:
        if entry["sequence_id"] not in eval_ids:
            continue
        gt_rows = list(read_jsonl(converted_root / entry["path"]))
        pred_rows = list(read_jsonl(detection_root / ("%s.jsonl" % entry["sequence_id"])))
        if len(gt_rows) != len(pred_rows):
            raise ValueError("评估帧数不一致 %s" % entry["sequence_id"])
        for gt_row, pred_row in zip(gt_rows, pred_rows):
            if gt_row["frame_id"] != pred_row["frame_id"] or int(gt_row["timestamp"]) != int(pred_row["timestamp"]):
                raise ValueError("评估帧未对齐 %s" % entry["sequence_id"])
            frames.append({"gt": gt_row["objects"], "pred": pred_row["objects"]})
    if not frames:
        print("评估split %s 没有帧" % args.eval_split)
        return
    metrics = evaluate_frames(
        frames,
        list(cfg.CLASS_NAMES),
        config["evaluation"]["iou_thresholds"],
        config["evaluation"]["score_threshold"],
    )
    positive = [item["ap"] for item in metrics.values() if item["num_gt"] > 0]
    summary = {"split": args.eval_split, "score_threshold": config["evaluation"]["score_threshold"], "classes": metrics}
    summary["mean_ap"] = float(sum(positive) / len(positive)) if positive else 0.0
    write_json(output_root / "detection_metrics.json", summary)
    print("class precision recall ap")
    for name, item in metrics.items():
        print("%s %.4f %.4f %.4f" % (name, item["precision"], item["recall"], item["ap"]))
    print("mAP %.4f" % summary["mean_ap"])


if __name__ == "__main__":
    main()
