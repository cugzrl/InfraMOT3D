import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_jsonl
from export_v2xseq_official_tracking import (
    OFFICIAL_RANGE,
    _inside_official_range,
    _kitti_line,
    _load_calib,
    _range_tools,
)


def _assert(condition, message):
    if not condition:
        raise SystemExit(message)


def _check_geometry(root, config):
    data_root = Path(config["project"]["data_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    bbox_filter, corner_fn = _range_tools(root)
    outside = [150.0, 0.0, -1.0, 0.0, 4.0, 1.8, 1.5]
    inside = [20.0, 0.0, -1.0, 0.2, 4.5, 1.8, 1.6]
    _assert(not _inside_official_range(outside, bbox_filter, corner_fn), "范围外的框被保留")
    _assert(_inside_official_range(inside, bbox_filter, corner_fn), "范围内的框被过滤")
    sample = None
    for entry in manifest["sequences"]:
        if entry["sequence_id"] not in set(read_json(root / config["split_file"])["val"]):
            continue
        for row in read_jsonl(converted_root / entry["path"]):
            if int(row["frame_index"]) != 0:
                continue
            for item in row["objects"]:
                if item["class_name"] == "Car" and _inside_official_range(item["box"], bbox_filter, corner_fn):
                    sample = (entry["sequence_id"], row, item)
                    break
            if sample:
                break
        if sample:
            break
    _assert(sample is not None, "val中没有范围内的Car")
    sequence_id, row, item = sample
    _assert(row["sequence_id"] == sequence_id, "sequence_id不一致")
    _assert(int(row["frame_index"]) == 0, "frame_index不是从0开始")
    rotation, translation = _load_calib(data_root, row["frame_id"])
    line = _kitti_line(0, item["source_track_id"], item["box"], rotation, translation, 1.0).split()
    box = [float(value) for value in item["box"]]
    _assert(len(line) == 18, "KITTI字段数量错误")
    _assert(abs(float(line[10]) - box[6]) < 1e-4, "高度顺序错误")
    _assert(abs(float(line[11]) - box[5]) < 1e-4, "宽度顺序错误")
    _assert(abs(float(line[12]) - box[4]) < 1e-4, "长度顺序错误")
    _assert(abs(float(line[16])) <= 3.1416, "rotation_y超出范围")
    center = translation.reshape(3) + rotation @ __import__("numpy").array([box[0], box[1], box[2] - box[6] / 2.0])
    _assert(abs(float(line[13]) - float(center[0])) < 1e-4, "相机x转换错误")
    _assert(abs(float(line[14]) - float(center[1])) < 1e-4, "相机y转换错误")
    _assert(abs(float(line[15]) - float(center[2])) < 1e-4, "相机z转换错误")
    _assert(list(OFFICIAL_RANGE) == [0.0, -39.68, -3.0, 100.0, 39.68, 1.0], "extended_range不正确")


def _write_gt_predictions(config, root, output_root):
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    val_ids = set(read_json(root / config["split_file"])["val"])
    prediction_root = output_root / "predictions"
    for entry in manifest["sequences"]:
        if entry["sequence_id"] not in val_ids:
            continue
        rows = []
        for row in read_jsonl(converted_root / entry["path"]):
            objects = []
            for item in row["objects"]:
                box = [float(value) for value in item["box"]]
                # 官方IoU在完全重合多边形上会得到NaN，预测只平移1e-4米
                box[0] += 1e-4
                objects.append(
                    {
                        "class_name": item["class_name"],
                        "score": 1.0,
                        "box": box,
                        "track_id": int(item["source_track_id"]),
                    }
                )
            rows.append(
                {
                    "sequence_id": row["sequence_id"],
                    "frame_index": row["frame_index"],
                    "frame_id": row["frame_id"],
                    "timestamp": row["timestamp"],
                    "objects": objects,
                }
            )
        write_jsonl(prediction_root / ("%s.jsonl" % entry["sequence_id"]), rows)


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    config = load_config("configs/ab3dmot_centerpoint.yaml")
    _check_geometry(root, config)
    output_root = root / "outputs" / "official_parity"
    _write_gt_predictions(config, root, output_root)
    from export_v2xseq_official_tracking import main as export_main
    import evaluate_v2xseq_official

    sys.argv = [
        "export",
        "--config",
        "configs/ab3dmot_centerpoint.yaml",
        "--split",
        "val",
        "--prediction-root",
        "outputs/official_parity/predictions",
        "--output",
        "outputs/official_parity/official_kitti",
        "--protocol",
        "official_v2xseq_range",
    ]
    export_main()
    meta = read_json(output_root / "official_kitti" / "export_meta.json")
    _assert(meta["protocol"] == "official_v2xseq_range", "导出协议不是official_v2xseq_range")
    _assert(meta["extended_range"] == OFFICIAL_RANGE, "导出范围不正确")
    sys.argv = [
        "evaluate",
        "--exported",
        "outputs/official_parity/official_kitti",
        "--output",
        "outputs/official_parity/official_metrics.json",
        "--name",
        "parity_gt",
    ]
    evaluate_v2xseq_official.main()
    metrics = read_json(output_root / "official_metrics.json")
    print("parity MOTA %.4f MOTP %.4f IDS %d" % (metrics["MOTA"], metrics["MOTP"], metrics["IDS"]))
    _assert(metrics["MOTA"] >= 0.99, "MOTA不够接近1")
    _assert(metrics["MOTP"] >= 0.99, "MOTP不够接近1")
    _assert(int(metrics["IDS"]) == 0, "IDS不为0")
    print("parity_ok")


if __name__ == "__main__":
    main()
