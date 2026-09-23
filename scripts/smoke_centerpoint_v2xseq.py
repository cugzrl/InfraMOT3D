import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.detection.config_check import check_centerpoint_configs
from inframot3d.detection.dataset_stats import MAJOR_CLASSES, gt_retention
from inframot3d.detection.metrics import evaluate_frames
from inframot3d.detection.openpcdet_adapter import load_openpcdet_cfg
from inframot3d.detection.v2x_seq_converter import CLASS_NAMES, openpcdet_to_box
from inframot3d.io import read_jsonl


def _check_samples(data_root, converted_root):
    with open(data_root / "infos" / "v2x_seq_infos_train.pkl", "rb") as stream:
        infos = pickle.load(stream)
    by_seq = {}
    for info in infos:
        by_seq.setdefault(info["sequence_id"], []).append(info)
    sequence_ids = sorted(by_seq, key=lambda item: len(by_seq[item]))[:2]
    chosen = []
    point_min = np.full(3, np.inf)
    point_max = np.full(3, -np.inf)
    gt_min = np.full(3, np.inf)
    gt_max = np.full(3, -np.inf)
    intensity_parts = []
    point_cloud_range = [0, -56.0, -5.0, 204.8, 40.0, 3.0]
    for sequence_id in sequence_ids:
        rows = sorted(by_seq[sequence_id], key=lambda item: item["frame_index"])
        converted = list(read_jsonl(converted_root / "sequences" / ("%s.jsonl" % sequence_id)))
        if len(rows) != len(converted):
            raise SystemExit("帧数未对齐 %s" % sequence_id)
        for info, frame in zip(rows, converted):
            if info["frame_id"] != frame["frame_id"] or int(info["timestamp"]) != int(frame["timestamp"]):
                raise SystemExit("frame_id或timestamp未对齐 %s" % sequence_id)
            if info["frame_index"] != frame["frame_index"] or info["sequence_id"] != frame["sequence_id"]:
                raise SystemExit("sequence对齐失败 %s" % sequence_id)
            points = np.load(data_root / "points" / ("%s.npy" % info["point_cloud"]["lidar_idx"]))
            boxes = info["annos"]["gt_boxes_lidar"]
            names = info["annos"]["name"]
            if points.shape[1] != 4 or points.dtype != np.float32 or not np.isfinite(points).all():
                raise SystemExit("点云无效 %s" % info["frame_id"])
            if boxes.ndim != 2 or boxes.shape[1] != 7 or not np.isfinite(boxes).all():
                raise SystemExit("标注框无效 %s" % info["frame_id"])
            if len(boxes) and np.any(boxes[:, 3:6] <= 0):
                raise SystemExit("框尺寸无效 %s" % info["frame_id"])
            if any(name not in CLASS_NAMES for name in names.tolist()):
                raise SystemExit("类别映射错误 %s" % info["frame_id"])
            if points[:, 3].size and (float(points[:, 3].min()) < 0.0 or float(points[:, 3].max()) > 1.0):
                raise SystemExit("intensity未归一化 %s" % info["frame_id"])
            intensity_parts.append(points[:, 3])
            point_min = np.minimum(point_min, points[:, :3].min(0))
            point_max = np.maximum(point_max, points[:, :3].max(0))
            if len(boxes):
                gt_min = np.minimum(gt_min, boxes[:, :3].min(0))
                gt_max = np.maximum(gt_max, boxes[:, :3].max(0))
        chosen.extend(rows)
        frames = []
        for info in rows:
            objects = [
                {"class_name": str(name), "score": 1.0, "box": openpcdet_to_box(box)}
                for name, box in zip(info["annos"]["name"], info["annos"]["gt_boxes_lidar"])
            ]
            frames.append({"gt": objects, "pred": objects})
        metrics = evaluate_frames(
            frames,
            CLASS_NAMES,
            {name: 0.5 if name in ("Car", "Van", "Bus", "Truck") else 0.25 for name in CLASS_NAMES},
            0.1,
        )
        for name, item in metrics.items():
            if item["num_gt"] and (item["precision"] < 0.99 or item["recall"] < 0.99 or item["ap"] < 0.99):
                raise SystemExit("GT自检失败 %s %s" % (sequence_id, name))
    overlap = np.maximum(point_min, gt_min) <= np.minimum(point_max, gt_max)
    if not overlap.all():
        raise SystemExit("点云和GT范围没有重叠")
    intensity = np.concatenate(intensity_parts) if intensity_parts else np.zeros((0,), dtype=np.float32)
    print("intensity min %.6f" % float(intensity.min()))
    print("intensity max %.6f" % float(intensity.max()))
    print("intensity mean %.6f" % float(intensity.mean()))
    for split_name in ("train", "val"):
        with open(data_root / "infos" / ("v2x_seq_infos_%s.pkl" % split_name), "rb") as stream:
            split_infos = pickle.load(stream)
        rows, kept, total = gt_retention(split_infos, point_cloud_range)
        print("%s GT中心范围内 %d / %d %.2f%%" % (split_name, kept, total, 100.0 * kept / max(total, 1)))
        for row in rows:
            print("%s %s %d / %d %.2f%%" % (split_name, row["class_name"], row["kept"], row["total"], row["ratio"] * 100.0))
            if row["class_name"] in MAJOR_CLASSES and row["total"] > 0 and row["ratio"] < 0.99:
                raise SystemExit("GT保留率过低 %s %s" % (split_name, row["class_name"]))
    print("抽检序列", sequence_ids, "帧", len(chosen))
    print("point_minmax", point_min.tolist(), point_max.tolist())
    print("gt_minmax", gt_min.tolist(), gt_max.tolist())
    return sequence_ids, chosen


def _write_smoke_infos(data_root, infos):
    path = data_root / "infos" / "v2x_seq_infos_smoke.pkl"
    with open(path, "wb") as stream:
        pickle.dump(infos, stream)
    cfg_path = ROOT / "third_party" / "OpenPCDet" / "tools" / "cfgs" / "v2x_seq_models" / "centerpoint_smoke.yaml"
    base = (ROOT / "third_party" / "OpenPCDet" / "tools" / "cfgs" / "v2x_seq_models" / "centerpoint.yaml").read_text(encoding="utf-8")
    model = base.split("MODEL:", 1)[1]
    cfg_path.write_text(
        "CLASS_NAMES: ['Car', 'Van', 'Bus', 'Truck', 'Pedestrian', 'Cyclist', 'Motorcyclist', 'Barrowlist']\n\n"
        "DATA_CONFIG:\n"
        "    _BASE_CONFIG_: cfgs/dataset_configs/v2x_seq_dataset.yaml\n"
        "    INFO_PATH: {\n"
        "        'train': [infos/v2x_seq_infos_smoke.pkl],\n"
        "        'test': [infos/v2x_seq_infos_smoke.pkl]\n"
        "    }\n\n"
        "MODEL:" + model,
        encoding="utf-8",
    )
    return cfg_path


def _forward_backward(cfg_file):
    import torch
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, model_fn_decorator

    cfg = load_openpcdet_cfg(cfg_file)

    class _Logger:
        def info(self, message):
            print(message)

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=2,
        dist=False,
        workers=0,
        logger=_Logger(),
        training=True,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.cuda().train()
    batch = next(iter(loader))
    result = model_fn_decorator()(model, batch)
    loss = result.loss
    if not torch.isfinite(loss):
        raise SystemExit("loss不是有限值")
    loss.backward()
    print("forward_backward_loss", float(loss.detach().cpu()))


def _train_one_epoch(cfg_file):
    smoke_output = ROOT / "third_party" / "OpenPCDet" / "output" / "v2x_seq_models" / "centerpoint_smoke"
    if smoke_output.exists():
        shutil.rmtree(smoke_output)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["CUDA_HOME"] = env.get("CUDA_HOME", "/usr/local/cuda-11.8")
    env["PATH"] = env["CUDA_HOME"] + "/bin:" + env.get("PATH", "")
    subprocess.check_call(
        [
            "conda", "run", "-n", "track", "--no-capture-output",
            "python", "train.py",
            "--cfg_file", "cfgs/v2x_seq_models/centerpoint_smoke.yaml",
            "--batch_size", "2",
            "--epochs", "1",
            "--workers", "2",
            "--extra_tag", "smoke",
            "--launcher", "none",
        ],
        cwd=str(ROOT / "third_party" / "OpenPCDet" / "tools"),
        env=env,
    )


def main():
    check_centerpoint_configs(ROOT)
    data_root = ROOT / "data" / "centerpoint_v2xseq"
    converted_root = ROOT / "data" / "converted" / "v2x_seq_infrastructure"
    _, infos = _check_samples(data_root, converted_root)
    cfg_file = _write_smoke_infos(data_root, infos)
    _forward_backward(cfg_file)
    _train_one_epoch(cfg_file)
    print("smoke_ok")


if __name__ == "__main__":
    main()
