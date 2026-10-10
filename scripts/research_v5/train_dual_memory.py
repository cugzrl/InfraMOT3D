"""Train the dual BEV memory on causal V2X-Seq clips.

The frozen 3D feature extractor initializes from a trained CenterPoint, while
the BEV backbone, CenterHead, and recurrent memory are optimized together.
Gradients traverse every frame inside each truncated-BPTT clip.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/research_v3"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, load_frozen_centerpoint
from inframot3d.perception.dual_bev_memory import DualBEVMemory
from infer_bev_memory import lidar_hit_map
from train_static_gate import targets

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"
CFG = "third_party/OpenPCDet/tools/cfgs/v2x_seq_models/centerpoint_fold{fold}.yaml"
CKPT = "third_party/OpenPCDet/output/v2x_seq_models/centerpoint_fold{fold}/default/ckpt/checkpoint_epoch_30.pth"


def set_head_mode(model, train):
    model.dense_head.train(train)
    for module in model.dense_head.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def labels(row, previous_row, shape, device, max_gap):
    _, _, height, width = shape
    x0, y0 = 0.0, -56.0
    dx, dy = 204.8 / width, 96.0 / height
    yy, xx = torch.meshgrid(
        (torch.arange(height, device=device) + 0.5) * dy + y0,
        (torch.arange(width, device=device) + 0.5) * dx + x0,
        indexing="ij",
    )
    occupied = torch.zeros((height, width), device=device, dtype=torch.bool)
    moving = torch.zeros_like(occupied)
    flow = torch.zeros((2, height, width), device=device)
    previous = {}
    if previous_row is not None:
        dt = (int(row["timestamp"]) - int(previous_row["timestamp"])) / 1e6
        if 0.0 < dt <= max_gap:
            counts = {}
            for old in previous_row["objects"]:
                tid = old["source_track_id"]
                counts[tid] = counts.get(tid, 0) + 1
                previous[tid] = old
            previous = {tid: obj for tid, obj in previous.items() if counts[tid] == 1}
        else:
            previous = {}
    counts = {}
    for obj in row["objects"]:
        tid = obj["source_track_id"]
        counts[tid] = counts.get(tid, 0) + 1
    for obj in row["objects"]:
        x, y, _, yaw, length, box_width, _ = map(float, obj["box"])
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        local_x = (xx - x) * cos_yaw + (yy - y) * sin_yaw
        local_y = -(xx - x) * sin_yaw + (yy - y) * cos_yaw
        mask = (local_x.abs() <= length / 2) & (local_y.abs() <= box_width / 2)
        occupied |= mask
        tid = obj["source_track_id"]
        if counts[tid] != 1 or tid not in previous:
            continue
        old_x, old_y = map(float, previous[tid]["box"][:2])
        shift_x = (x - old_x) / dx
        shift_y = (y - old_y) / dy
        if abs(shift_x) > 8 or abs(shift_y) > 8:
            continue
        moving |= mask
        flow[0, mask] = shift_x
        flow[1, mask] = shift_y
    return occupied[None, None].float(), moving[None, None], flow[None]


def frame_loss(model, memory, dataset, load_gpu, row, previous_row, seq, device, state, flow_weight, mask_weight):
    points = np.load(POINTS / f"{seq}_{row['frame_id']}.npy")
    batch = batch_from_points(dataset, load_gpu, points, device)
    with torch.no_grad():
        for module in model.module_list:
            if module is model.backbone_2d:
                break
            batch = module(batch)
    batch = model.backbone_2d(batch)
    current = batch["spatial_features_2d"]
    hit = lidar_hit_map(points, current.shape[-2], current.shape[-1], device)
    stamp = int(row["timestamp"]) / 1e6
    enhanced, state, aux = memory(current, state, seq, stamp, hit)
    batch["spatial_features_2d"] = enhanced
    batch["gt_boxes"] = targets(row, dataset.class_names, device)
    model.dense_head(batch)
    det_loss, _ = model.dense_head.get_loss()
    occupancy, matched, flow = labels(row, previous_row, aux["flow"].shape, device, memory.max_dynamic_gap)
    dyn_loss = F.binary_cross_entropy_with_logits(
        aux["dynamic_logits"], occupancy,
        pos_weight=occupancy.new_tensor(4.0),
    )
    if matched.any() and aux["dynamic_valid"]:
        flow_loss = F.smooth_l1_loss(aux["flow"][matched.expand_as(flow)], flow[matched.expand_as(flow)])
    else:
        flow_loss = det_loss.new_zeros(())
    total = det_loss + mask_weight * dyn_loss + flow_weight * flow_loss
    stats = {"det": float(det_loss.detach()), "mask": float(dyn_loss.detach()),
             "flow": float(flow_loss.detach()), "total": float(total.detach())}
    return total, state, stats


def run_sequences(model, memory, dataset, load_gpu, sequences, paths, device, optimizer,
                  clip_length, flow_weight, mask_weight, max_frames):
    training = optimizer is not None
    model.backbone_2d.train(training)
    memory.train(training)
    # CenterHead must stay in training mode to construct GT targets during validation.
    set_head_mode(model, True)
    sums = {name: 0.0 for name in ("det", "mask", "flow", "total")}
    frames = 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for seq in sequences:
            rows = list(read_jsonl(CONVERTED / paths[seq]))
            if max_frames:
                rows = rows[:max_frames]
            state = None
            clip_losses = []
            for index, row in enumerate(rows):
                previous_row = rows[index - 1] if index else None
                loss, state, stats = frame_loss(
                    model, memory, dataset, load_gpu, row, previous_row, seq, device,
                    state, flow_weight, mask_weight,
                )
                clip_losses.append(loss)
                for name in sums:
                    sums[name] += stats[name]
                frames += 1
                if training and (len(clip_losses) == clip_length or index == len(rows) - 1):
                    optimizer.zero_grad(set_to_none=True)
                    torch.stack(clip_losses).mean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(memory.parameters()) + list(model.backbone_2d.parameters())
                        + list(model.dense_head.parameters()), 10.0,
                    )
                    optimizer.step()
                    state = state.detach()
                    clip_losses.clear()
            print(("train" if training else "val"), seq, "frames", len(rows),
                  "mean_loss", round(sums["total"] / max(frames, 1), 4), flush=True)
    return {**{name: sums[name] / max(frames, 1) for name in sums},
            "frames": frames, "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=("A", "B"), default="A")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--memory-stride", type=int, default=2)
    parser.add_argument("--lr-memory", type=float, default=3e-4)
    parser.add_argument("--lr-detector", type=float, default=3e-5)
    parser.add_argument("--flow-weight", type=float, default=0.05)
    parser.add_argument("--mask-weight", type=float, default=0.05)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    if args.clip_length < 2:
        parser.error("clip-length must be at least 2 for temporal BPTT")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    fold_data = read_json(ROOT / "configs/research/v2xseq_train_folds.json")
    other = "B" if args.fold == "A" else "A"
    train_sequences = list(fold_data["inner_train"][other])
    val_sequences = list(fold_data["inner_holdout"][other])
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    model, dataset, load_gpu = load_frozen_centerpoint(
        ROOT / "third_party/OpenPCDet", ROOT / CFG.format(fold=args.fold),
        ROOT / CKPT.format(fold=args.fold), device,
    )
    for parameter in model.backbone_2d.parameters():
        parameter.requires_grad_(True)
    for parameter in model.dense_head.parameters():
        parameter.requires_grad_(True)
    memory = DualBEVMemory(
        input_channels=512, hidden_channels=args.hidden_channels,
        memory_stride=args.memory_stride,
    ).to(device)
    optimizer = torch.optim.AdamW([
        {"params": memory.parameters(), "lr": args.lr_memory},
        {"params": model.backbone_2d.parameters(), "lr": args.lr_detector},
        {"params": model.dense_head.parameters(), "lr": args.lr_detector},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_val = float("inf")
    history = []
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        memory.load_state_dict(saved["memory"])
        model.backbone_2d.load_state_dict(saved["backbone_2d"])
        model.dense_head.load_state_dict(saved["dense_head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start_epoch = saved["epoch"]
        best_val = saved.get("best_val", best_val)
        history = saved.get("history", [])
    run_start = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        random.shuffle(train_sequences)
        train = run_sequences(
            model, memory, dataset, load_gpu, train_sequences, paths, device,
            optimizer, args.clip_length, args.flow_weight, args.mask_weight,
            args.max_frames_per_sequence,
        )
        val = None
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            val = run_sequences(
                model, memory, dataset, load_gpu, val_sequences, paths, device,
                None, args.clip_length, args.flow_weight, args.mask_weight,
                args.max_frames_per_sequence,
            )
        scheduler.step()
        history.append({"epoch": epoch + 1, "train": train, "val": val,
                        "lr_memory": optimizer.param_groups[0]["lr"]})
        snapshot = {
            "epoch": epoch + 1, "memory": memory.state_dict(),
            "backbone_2d": model.backbone_2d.state_dict(),
            "dense_head": model.dense_head.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "history": history, "best_val": best_val, "args": vars(args),
            "base_checkpoint": CKPT.format(fold=args.fold),
            "train_sequences": train_sequences, "val_sequences": val_sequences,
        }
        if val is not None and val["total"] < best_val:
            best_val = val["total"]
            snapshot["best_val"] = best_val
            torch.save(snapshot, output / "best.pth")
        torch.save(snapshot, output / "latest.pth")
        with (output / "history.json").open("w", encoding="utf-8") as handle:
            json.dump({"args": vars(args), "history": history,
                       "elapsed_seconds": time.perf_counter() - run_start}, handle, indent=2)
        print("epoch", epoch + 1, "train", train, "val", val,
              "best_val", best_val, "gpu_peak_gb",
              round(torch.cuda.max_memory_allocated(device) / 2**30, 3), flush=True)


if __name__ == "__main__":
    main()
