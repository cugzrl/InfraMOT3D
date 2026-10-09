import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inframot3d.config import load_config
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols import build_protocol
from inframot3d.io import read_json, read_jsonl, write_json
from inframot3d.analysis.scene_difficulty import _match
from inframot3d.tracking.grae_adapter import build_clips, build_model, load_checkpoint
from inframot3d.tracking.grae_recovery import TrackConditionedRecovery, recovery_clip_loss
from infer_grae_variant import track_sequences


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _track_index(mapping, source_id):
    key = str(source_id)
    if key not in mapping:
        mapping[key] = len(mapping) + 1
    return mapping[key]


def _box(frame, index):
    translation = frame["translation"][index]
    size = frame["size"][index]
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(frame["yaw"][index]),
        float(size[0]),
        float(size[1]),
        float(size[2]),
    ]


def _same_class_pairs(gt_objects, det_index, frame, class_names):
    # 与评估相同的 3D IoU 0.25，并且只连接细粒度类别一致的框
    pairs = []
    gt_groups = {}
    det_groups = {}
    for row, item in enumerate(gt_objects):
        gt_groups.setdefault(item["class_name"], []).append(row)
    for index in det_index:
        det_groups.setdefault(class_names[int(frame["classes"][index])], []).append(index)
    for name, rows in gt_groups.items():
        columns = det_groups.get(name, [])
        if not rows or not columns:
            continue
        matches = _match(
            [gt_objects[row]["box"] for row in rows],
            [_box(frame, index) for index in columns],
            0.25,
            20.0,
        )
        for local_row, local_col, _ in matches:
            pairs.append((rows[local_row], columns[local_col]))
    return pairs


def _attach_low_score_labels(frames, gt_rows, class_names, class_to_index):
    # 先让高分框拿走 3D 匹配，剩余 GT 再给低分框；靠近 GT 但没匹配上的低分框保持未知
    mapping = {}
    added = 0
    background = 0
    for frame, gt_row in zip(frames, gt_rows):
        gt_objects = []
        for item in gt_row["objects"]:
            if item["class_name"] not in class_to_index:
                continue
            gt_objects.append(
                {
                    "class_name": item["class_name"],
                    "box": item["box"],
                    "track_index": _track_index(mapping, item["source_track_id"]),
                }
            )
        frame["tracking_id"][:] = -1
        high_index = [index for index, score in enumerate(frame["score"]) if float(score) >= 0.1]
        low_index = [index for index, score in enumerate(frame["score"]) if 0.01 <= float(score) < 0.1]
        used_rows = set()
        for row, index in _same_class_pairs(gt_objects, high_index, frame, class_names):
            frame["tracking_id"][index] = int(gt_objects[row]["track_index"])
            used_rows.add(row)
        remaining = [item for row, item in enumerate(gt_objects) if row not in used_rows]
        for row, index in _same_class_pairs(remaining, low_index, frame, class_names):
            frame["tracking_id"][index] = int(remaining[row]["track_index"])
            added += 1
        if gt_objects:
            gt_xy = np.asarray([[float(item["box"][0]), float(item["box"][1])] for item in gt_objects], dtype=np.float64)
        else:
            gt_xy = np.zeros((0, 2), dtype=np.float64)
        for index in low_index:
            if int(frame["tracking_id"][index]) >= 0:
                continue
            box = _box(frame, index)
            if len(gt_xy) == 0 or np.hypot(gt_xy[:, 0] - box[0], gt_xy[:, 1] - box[1]).min() > 4.0:
                frame["tracking_id"][index] = -2
                background += 1
    return added, background


def _grad_norm(module):
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/recovery_centerpoint.yaml")
    parser.add_argument("--base-config", default="configs/trackers/grae/centerpoint.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    base = load_config(args.base_config)
    root = config["_root"]
    seed = int(config["project"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    classes = list(config["classes"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    with _resolve(root, config["project"]["grae_data_root"]).joinpath("v2x_seq_train.pkl").open("rb") as stream:
        sequences = pickle.load(stream)
    by_id = {str(frames[0]["sequence_id"]): frames for frames in sequences if frames}
    manifest = read_json(Path(config["project"]["converted_root"]) / "manifest.json")
    calibration_ids = {str(value) for value in config["train"]["calibration_sequences"]}
    added = 0
    background = 0
    train_sequences = []
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in by_id or sequence_id in calibration_ids:
            continue
        frames = by_id[sequence_id]
        gt_rows = list(read_jsonl(Path(config["project"]["converted_root"]) / entry["path"]))
        if len(frames) != len(gt_rows):
            raise RuntimeError("训练帧数不一致 %s" % sequence_id)
        hit, far = _attach_low_score_labels(frames, gt_rows, classes, class_to_index)
        added += hit
        background += far
        train_sequences.append(frames)
        print("低分监督序列%s 累计正样本%d 背景%d" % (sequence_id, added, background), flush=True)
    clips = build_clips(
        train_sequences,
        config["train"]["clip_len"],
        config["train"]["clip_stride"],
        config["train"]["score_threshold"],
    )
    if not clips:
        raise RuntimeError("没有可用训练片段")

    def _has_low_positive(clip):
        for frame in clip:
            score = np.asarray(frame["score"])
            identity = np.asarray(frame["tracking_id"])
            if len(score) and np.any((score >= 0.01) & (score < 0.1) & (identity >= 0)):
                return True
        return False

    low_clips = [clip for clip in clips if _has_low_positive(clip)]
    other_clips = [clip for clip in clips if not _has_low_positive(clip)]
    if not low_clips:
        raise RuntimeError("训练片段里没有低分正样本")
    recoverable = 0
    for clip in low_clips:
        alive = set()
        for frame in clip:
            score = np.asarray(frame["score"], dtype=np.float64)
            identity = np.asarray(frame["tracking_id"])
            low_ids = {int(value) for value, item in zip(identity, score) if int(value) >= 0 and item < 0.1}
            recoverable += len(low_ids & alive)
            for value, item in zip(identity, score):
                if int(value) >= 0 and float(item) >= 0.4:
                    alive.add(int(value))
    print(
        "训练片段%d 含低分框片段%d 低分正样本%d 可接到历史轨迹%d"
        % (len(clips), len(low_clips), added, recoverable),
        flush=True,
    )
    if recoverable == 0:
        raise RuntimeError("低分正样本没有和已有轨迹形成监督对")
    device = torch.device(args.device)
    model = build_model(
        _resolve(root, config["project"]["grae_root"]),
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        device,
    )
    base_checkpoint = Path(base["project"]["output_root"]) / "ckpt" / "checkpoint-best.pth"
    load_checkpoint(model, base_checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    recovery = TrackConditionedRecovery(config["model"]["in_channels"]).to(device)
    optimizer = torch.optim.AdamW(recovery.parameters(), lr=float(config["train"]["lr"]), weight_decay=float(config["train"]["weight_decay"]))
    birth = yaml.safe_load(_resolve(root, config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8"))["score_thresholds"]
    output_root = Path(config["project"]["output_root"])
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    probe_loss = None
    probe_stats = None
    optimizer.zero_grad(set_to_none=True)
    for probe in low_clips[:64]:
        probe_loss, probe_stats = recovery_clip_loss(
            model,
            recovery,
            probe,
            device,
            classes,
            birth,
            association_alpha=config["tracker"]["association_alpha"],
            age=config["tracker"]["age"],
            hard_distance=config["train"]["hard_distance"],
        )
        if probe_loss is not None and int(probe_stats["low_positive"]) > 0:
            break
    if probe_loss is None or int(probe_stats["low_positive"]) == 0:
        raise RuntimeError("梯度检查没有低分正样本")
    probe_loss.backward()
    grad_norm = _grad_norm(recovery)
    print("梯度检查 loss %.6f grad %.6f %s" % (float(probe_loss.detach().cpu()), grad_norm, probe_stats), flush=True)
    if grad_norm <= 0.0:
        raise RuntimeError("新模块梯度为零")
    optimizer.zero_grad(set_to_none=True)
    epochs = int(args.epochs or config["train"]["epochs"])
    history = []
    best_hota = -1.0
    best_epoch = None
    protocol = build_protocol("v2xseq", root)
    frozen_score = float(config["train"]["frozen_score_threshold"])
    for epoch in range(1, epochs + 1):
        repeat = min(8, max(1, int(round(0.25 * len(other_clips) / len(low_clips)))))
        epoch_clips = list(other_clips) + low_clips * repeat
        random.shuffle(epoch_clips)
        total = 0.0
        count = 0
        positive = 0
        low_positive = 0
        for clip in epoch_clips:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = recovery_clip_loss(
                model,
                recovery,
                clip,
                device,
                classes,
                birth,
                association_alpha=config["tracker"]["association_alpha"],
                age=config["tracker"]["age"],
                hard_distance=config["train"]["hard_distance"],
            )
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError("loss不是有限值")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
            positive += int(stats["positive"])
            low_positive += int(stats["low_positive"])
            if count % 200 == 0:
                print("epoch %d clip %d loss %.6f" % (epoch, count, total / count), flush=True)
        if count == 0:
            raise RuntimeError("本轮没有关联损失")
        mean_loss = total / count
        epoch_path = ckpt_dir / ("recovery-epoch%d.pth" % epoch)
        torch.save(
            {
                "epoch": epoch,
                "recovery": recovery.state_dict(),
                "grae_checkpoint": str(base_checkpoint),
                "train_loss": mean_loss,
                "positive_pairs": positive,
                "low_positive_pairs": low_positive,
            },
            epoch_path,
        )
        prediction_root = output_root / "predictions" / ("calib_epoch%d" % epoch)
        runtime = track_sequences(
            config,
            base_checkpoint,
            prediction_root,
            split="train",
            sequences=sorted(calibration_ids),
            score_floor=config["tracker"]["score_floor"],
            class_compat=config["tracker"]["class_compat"],
            motion_mode=config["tracker"]["motion_mode"],
            recovery_path=epoch_path,
            device=str(device),
            weak_association="learned",
        )
        metrics = evaluate_hota(config, prediction_root, sorted(calibration_ids), protocol, frozen_score)
        record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "clips": count,
            "positive_pairs": positive,
            "low_positive_pairs": low_positive,
            "calib_hota": metrics,
            "runtime_seconds": runtime["elapsed_seconds"],
            "checkpoint": str(epoch_path),
        }
        history.append(record)
        if float(metrics["HOTA"]) > best_hota:
            best_hota = float(metrics["HOTA"])
            best_epoch = epoch
            torch.save(torch.load(epoch_path, map_location="cpu", weights_only=False), ckpt_dir / "recovery-best.pth")
        write_json(ckpt_dir / "train_history.json", {"epochs": history, "best_epoch": best_epoch, "best_hota": best_hota, "grad_check": grad_norm, "low_score_labels": added, "low_score_background": background})
        print(
            "epoch %d loss %.6f low_pos %d calib HOTA %.4f"
            % (epoch, mean_loss, low_positive, float(metrics["HOTA"])),
            flush=True,
        )
    print("训练结束 best epoch %s HOTA %.4f" % (best_epoch, best_hota), flush=True)


if __name__ == "__main__":
    main()
