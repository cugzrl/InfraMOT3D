import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inframot3d.analysis.scene_difficulty import _match
from inframot3d.config import load_config
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols import build_protocol
from inframot3d.io import read_json, read_jsonl, write_json
from inframot3d.tracking.grae_adapter import build_clips, build_model, load_checkpoint
from inframot3d.tracking.grae_semantic import VehicleSemanticGate, semantic_clip_loss
from infer_grae_variant import track_sequences
from train_grae_recovery import _box, _grad_norm, _resolve, _track_index


def _group_name(name):
    if name in {"Car", "Van", "Bus", "Truck"}:
        return "Vehicle"
    return name


def _group_pairs(gt_objects, det_index, frame, class_names):
    pairs = []
    gt_groups = {}
    det_groups = {}
    for row, item in enumerate(gt_objects):
        gt_groups.setdefault(_group_name(item["class_name"]), []).append(row)
    for index in det_index:
        det_groups.setdefault(_group_name(class_names[int(frame["classes"][index])]), []).append(index)
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


def _attach_group_labels(frames, gt_rows, class_names, class_to_index):
    # 车辆大类内允许细类不同的框拿到同一个 GT 身份，非车辆仍按原类匹配
    mapping = {}
    cross = 0
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
        index = [item for item, score in enumerate(frame["score"]) if float(score) >= 0.1]
        for row, det_index in _group_pairs(gt_objects, index, frame, class_names):
            frame["tracking_id"][det_index] = int(gt_objects[row]["track_index"])
            if class_names[int(frame["classes"][det_index])] != gt_objects[row]["class_name"]:
                cross += 1
    return cross


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/recovery_centerpoint.yaml")
    parser.add_argument("--base-config", default="configs/trackers/grae/centerpoint.yaml")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
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
        import pickle

        sequences = pickle.load(stream)
    by_id = {str(frames[0]["sequence_id"]): frames for frames in sequences if frames}
    manifest = read_json(Path(config["project"]["converted_root"]) / "manifest.json")
    calibration_ids = {str(value) for value in config["train"]["calibration_sequences"]}
    cross = 0
    train_sequences = []
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if sequence_id not in by_id or sequence_id in calibration_ids:
            continue
        frames = by_id[sequence_id]
        gt_rows = list(read_jsonl(Path(config["project"]["converted_root"]) / entry["path"]))
        cross += _attach_group_labels(frames, gt_rows, classes, class_to_index)
        train_sequences.append(frames)
        print("语义监督序列%s 累计跨类正样本%d" % (sequence_id, cross), flush=True)
    clips = build_clips(train_sequences, config["train"]["clip_len"], config["train"]["clip_stride"], 0.1)
    if not clips or cross == 0:
        raise RuntimeError("没有跨类监督")
    print("训练片段%d 跨类正样本%d" % (len(clips), cross), flush=True)
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
    gate = VehicleSemanticGate(config["model"]["in_channels"], config["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=float(config["train"]["lr"]), weight_decay=0.0)
    birth = yaml.safe_load(_resolve(root, config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8"))["score_thresholds"]
    groups = []
    extra = 1
    for name in classes:
        if name in {"Car", "Van", "Bus", "Truck"}:
            groups.append(0)
        else:
            groups.append(extra)
            extra += 1
    group_table = torch.tensor(groups, dtype=torch.long, device=device)
    output_root = Path(config["project"]["output_root"])
    ckpt_dir = output_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    probe_loss, probe_stats = semantic_clip_loss(
        model, gate, clips[0], device, classes, birth, group_table, association_alpha=0.24, age=12
    )
    if probe_loss is None:
        for clip in clips:
            probe_loss, probe_stats = semantic_clip_loss(
                model, gate, clip, device, classes, birth, group_table, association_alpha=0.24, age=12
            )
            if probe_loss is not None and int(probe_stats["positive"]) > 0:
                break
    if probe_loss is None or int(probe_stats["positive"]) == 0:
        raise RuntimeError("语义门控没有正样本梯度")
    probe_loss.backward()
    grad_norm = _grad_norm(gate)
    print("梯度检查 loss %.6f grad %.6f %s" % (float(probe_loss.detach().cpu()), grad_norm, probe_stats), flush=True)
    if grad_norm <= 0.0:
        raise RuntimeError("语义门控梯度为零")
    optimizer.zero_grad(set_to_none=True)
    protocol = build_protocol("v2xseq", root)
    frozen_score = float(config["train"]["frozen_score_threshold"])
    history = []
    best_hota = -1.0
    best_epoch = None
    for epoch in range(1, int(args.epochs) + 1):
        random.shuffle(clips)
        total = 0.0
        count = 0
        positive = 0
        for clip in clips:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = semantic_clip_loss(
                model,
                gate,
                clip,
                device,
                classes,
                birth,
                group_table,
                association_alpha=config["tracker"]["association_alpha"],
                age=config["tracker"]["age"],
            )
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
            positive += int(stats["positive"])
            if count % 200 == 0:
                print("epoch %d clip %d loss %.6f" % (epoch, count, total / count), flush=True)
        if count == 0:
            raise RuntimeError("本轮没有语义损失")
        mean_loss = total / count
        epoch_path = ckpt_dir / ("semantic-epoch%d.pth" % epoch)
        torch.save(
            {
                "epoch": epoch,
                "semantic": gate.state_dict(),
                "grae_checkpoint": str(base_checkpoint),
                "train_loss": mean_loss,
                "cross_positive_pairs": positive,
            },
            epoch_path,
        )
        prediction_root = output_root / "predictions" / ("semantic_calib_epoch%d" % epoch)
        runtime = track_sequences(
            config,
            base_checkpoint,
            prediction_root,
            split="train",
            sequences=sorted(calibration_ids),
            score_floor=0.1,
            class_compat="exact",
            motion_mode="zero",
            semantic_path=epoch_path,
            device=str(device),
        )
        metrics = evaluate_hota(config, prediction_root, sorted(calibration_ids), protocol, frozen_score)
        record = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "clips": count,
            "cross_positive_pairs": positive,
            "calib_hota": metrics,
            "runtime_seconds": runtime["elapsed_seconds"],
            "checkpoint": str(epoch_path),
        }
        history.append(record)
        if float(metrics["HOTA"]) > best_hota:
            best_hota = float(metrics["HOTA"])
            best_epoch = epoch
            torch.save(torch.load(epoch_path, map_location="cpu", weights_only=False), ckpt_dir / "semantic-best.pth")
        write_json(
            ckpt_dir / "semantic_history.json",
            {"epochs": history, "best_epoch": best_epoch, "best_hota": best_hota, "grad_check": grad_norm, "cross_labels": cross},
        )
        print("epoch %d loss %.6f cross_pos %d calib HOTA %.4f" % (epoch, mean_loss, positive, float(metrics["HOTA"])), flush=True)
    print("语义训练结束 best epoch %s HOTA %.4f" % (best_epoch, best_hota), flush=True)


if __name__ == "__main__":
    main()
