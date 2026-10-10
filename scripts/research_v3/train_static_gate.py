"""Train a 897-parameter static-memory gate with frozen CenterPoint.

Training is sequence-causal. Current GT supervises CenterHead loss, while the
memory inputs use only current/past LiDAR and online D0 GRAE tracks. The detector
backbone and dense-head parameters and batch-normalization statistics stay fixed.
"""

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.io import read_json, read_jsonl, write_json
from inframot3d.perception.centerpoint_bev import batch_from_points, load_frozen_centerpoint
from inframot3d.perception.scene_memory import LongTermSceneMemory
from infer_bev_memory import (dynamic_mask, lidar_hit_map, motion_velocity,
                              warp_dynamic_feature)
from static_gate import StaticGate, parameter_count

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"


def targets(row, class_names, device):
    indices = {name: i + 1 for i, name in enumerate(class_names)}
    boxes = []
    for item in row["objects"]:
        if item["class_name"] not in indices:
            continue
        x, y, z, yaw, length, width, height = item["box"]
        boxes.append([x, y, z, length, width, height, yaw, indices[item["class_name"]]])
    if not boxes:
        return torch.zeros((1, 0, 8), dtype=torch.float32, device=device)
    return torch.tensor([boxes], dtype=torch.float32, device=device)


def train_sequence(model, dataset, load_gpu, gate, optimizer, seq, rows, tracks, max_frames):
    history = deque(maxlen=2)
    memory_state = None
    scene_memory = None
    total = 0.0
    steps = 0
    for index, row in enumerate(rows[:max_frames] if max_frames else rows):
        stamp = int(row["timestamp"]) / 1e6
        if history and stamp - history[-1][0] > 0.3:
            history.clear()
            memory_state = None
        points = np.load(POINTS / f"{seq}_{row['frame_id']}.npy")
        batch = batch_from_points(dataset, load_gpu, points, torch.device("cuda"))
        with torch.no_grad():
            for module in model.module_list:
                if module is model.dense_head:
                    break
                batch = module(batch)
            current = batch["spatial_features_2d"].detach()
            _, _, height, width = current.shape
            hit = lidar_hit_map(points, height, width, current.device)
            if scene_memory is None:
                scene_memory = LongTermSceneMemory(current.shape[1], gate=gate)
            if history:
                velocities = motion_velocity(tracks, index)
                past = torch.stack([
                    warp_dynamic_feature(feat, current, tracks[old_index]["objects"],
                                         velocities, stamp - old_stamp)
                    for old_stamp, feat, old_index in history
                ]).mean(0)
                dynamic = 0.8 * current + 0.2 * past
            else:
                dynamic = current
        if memory_state is None:
            empty = torch.zeros_like(current)
            invalid = torch.zeros_like(hit, dtype=torch.bool)
            fused, gate_map = gate(current, dynamic, empty, invalid, hit)
        else:
            fused, memory_state, gate_map = scene_memory.read(
                current, dynamic, memory_state, seq, stamp, hit)
        batch["spatial_features_2d"] = fused
        batch["gt_boxes"] = targets(row, dataset.class_names, current.device)
        model.dense_head(batch)
        loss, _ = model.dense_head.get_loss()
        loss = loss + 1e-4 * gate_map.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        steps += 1
        history.append((stamp, current, index))
        with torch.no_grad():
            mask = dynamic_mask(tracks[index]["objects"], height, width,
                                current.device)
            memory_state = scene_memory.write(current, memory_state, seq, stamp,
                                              hit, mask)
        if steps % 100 == 0:
            print(seq, "frames", steps, "loss", round(total / steps, 4), flush=True)
    return total / max(steps, 1), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--track-history-root", required=True)
    ap.add_argument("--train-sequences", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-frames-per-sequence", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, dataset, load_gpu = load_frozen_centerpoint(
        ROOT / "third_party/OpenPCDet", ROOT / args.cfg, ROOT / args.ckpt,
        torch.device("cuda"))
    model.dense_head.train()
    for module in model.dense_head.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    gate = StaticGate().cuda().train()
    optimizer = torch.optim.AdamW(gate.parameters(), lr=args.lr, weight_decay=1e-4)
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {x["sequence_id"]: x["path"] for x in manifest["sequences"]}
    history = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        seqs = list(args.train_sequences)
        random.shuffle(seqs)
        for seq in seqs:
            rows = list(read_jsonl(CONVERTED / paths[seq]))
            tracks = list(read_jsonl(ROOT / args.track_history_root / f"{seq}.jsonl"))
            if [x["frame_id"] for x in rows] != [x["frame_id"] for x in tracks]:
                raise ValueError(f"Track history mismatch {seq}")
            loss, steps = train_sequence(model, dataset, load_gpu, gate, optimizer,
                                         seq, rows, tracks, args.max_frames_per_sequence)
            history.append({"epoch": epoch + 1, "sequence": seq, "frames": steps, "loss": loss})
            print("epoch", epoch + 1, seq, "frames", steps, "loss", round(loss, 4), flush=True)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": gate.state_dict(), "config": vars(args),
                "parameter_count": parameter_count(gate), "history": history}, out / "static_gate.pth")
    write_json(out / "train_manifest.json", {"config": vars(args), "parameter_count": parameter_count(gate),
               "seconds": time.perf_counter() - start, "history": history,
               "note": "Frozen detector; training-side OOF sequences only; D0 online tracks as causal teacher memory."})
    print("saved", out / "static_gate.pth", "parameters", parameter_count(gate), flush=True)


if __name__ == "__main__":
    main()
