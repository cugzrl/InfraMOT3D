import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.evaluation.dair_reference import run_stock_evaluate
from inframot3d.evaluation.protocols.v2xseq import OFFICIAL_RANGE, V2XSeqProtocol, _kitti_line, _load_calib
from inframot3d.io import read_json, read_jsonl, write_jsonl


KEYS = ("MOTA", "MOTP", "AMOTA", "AMOTP", "IDSW", "FP", "FN", "FM")


def _assert(condition, message):
    if not condition:
        raise SystemExit(message)


def _check_geometry(root, config, protocol):
    data_root = Path(config["project"]["data_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    _assert(not protocol.inside_range([150.0, 0.0, -1.0, 0.0, 4.0, 1.8, 1.5]), "范围外的框被保留")
    _assert(protocol.inside_range([20.0, 0.0, -1.0, 0.2, 4.5, 1.8, 1.6]), "范围内的框被过滤")
    sample = None
    val_ids = set(read_json(root / config["split_file"])["val"])
    for entry in manifest["sequences"]:
        if entry["sequence_id"] not in val_ids:
            continue
        for row in read_jsonl(converted_root / entry["path"]):
            if int(row["frame_index"]) != 0:
                continue
            for item in row["objects"]:
                if item["class_name"] == "Car" and protocol.inside_range(item["box"]):
                    sample = (entry["sequence_id"], row, item)
                    break
            if sample:
                break
        if sample:
            break
    _assert(sample is not None, "val中没有范围内的Car")
    sequence_id, row, item = sample
    _assert(row["sequence_id"] == sequence_id, "sequence_id不一致")
    rotation, translation = _load_calib(data_root, row["frame_id"])
    line = _kitti_line(0, item["source_track_id"], item["box"], rotation, translation, 1.0).split()
    box = [float(value) for value in item["box"]]
    _assert(len(line) == 18, "KITTI字段数量错误")
    _assert(abs(float(line[10]) - box[6]) < 1e-4, "高度顺序错误")
    _assert(abs(float(line[11]) - box[5]) < 1e-4, "宽度顺序错误")
    _assert(abs(float(line[12]) - box[4]) < 1e-4, "长度顺序错误")
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
    config = load_config("configs/trackers/ab3dmot/centerpoint.yaml")
    root = config["_root"]
    protocol = V2XSeqProtocol(root)
    _check_geometry(root, config, protocol)
    output_root = root / "outputs" / "official_parity"
    _write_gt_predictions(config, root, output_root)
    metrics = UnifiedMOTEvaluator(root).evaluate(
        config,
        output_root / "predictions",
        output_root / "evaluation",
        split="val",
    )
    official = run_stock_evaluate(root, output_root / "evaluation" / "kitti", name="parity_stock")
    for key in KEYS:
        gap = abs(float(metrics[key]) - float(official[key]))
        print("%s误差 %.8f" % (key, gap))
        _assert(gap < 1e-6, "%s与官方不一致" % key)
    print("parity MOTA %.4f MOTP %.4f IDSW %d" % (metrics["MOTA"], metrics["MOTP"], metrics["IDSW"]))
    _assert(metrics["MOTA"] >= 0.99, "MOTA不够接近1")
    _assert(metrics["MOTP"] >= 0.99, "MOTP不够接近1")
    _assert(int(metrics["IDSW"]) == 0, "IDSW不为0")
    print("parity_ok")


if __name__ == "__main__":
    main()
