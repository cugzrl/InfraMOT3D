"""Full causal training of detector feedback from persistent object queries."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/research_v3"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, load_frozen_centerpoint
from inframot3d.perception.persistent_query_feedback import PersistentQueryFeedback
from inframot3d.perception.attentive_query_feedback import AttentiveQueryFeedback
from train_static_gate import targets
from train_dual_memory import CFG, CKPT, CONVERTED, POINTS, set_head_mode


def query_targets(info, state, row, class_names, device):
    if not info["valid"]:
        zero = state.embedding.new_zeros(())
        return zero, zero, 0
    centers = info["predicted_centers"]
    class_index = {name: i + 1 for i, name in enumerate(class_names)}
    gt = [item for item in row["objects"] if item["class_name"] in class_index]
    exists = centers.new_zeros(len(centers))
    pairs = []
    if gt:
        target_centers = torch.tensor([item["box"][:2] for item in gt], device=device, dtype=centers.dtype)
        target_classes = torch.tensor([class_index[item["class_name"]] for item in gt], device=device)
        distance = torch.cdist(centers, target_centers)
        cost = distance + (state.labels[:, None] != target_classes[None, :]).float() * 1e4
        old_indices, gt_indices = linear_sum_assignment(cost.detach().cpu().numpy())
        pairs = [(int(i), int(j)) for i, j in zip(old_indices, gt_indices) if float(cost[i, j]) < 4.0]
        for i, _ in pairs:
            exists[i] = 1.0
        if pairs:
            source = centers[[i for i, _ in pairs]]
            target = target_centers[[j for _, j in pairs]]
            motion = F.smooth_l1_loss(source, target)
        else:
            motion = centers.new_zeros(())
    else:
        motion = centers.new_zeros(())
    existence = F.binary_cross_entropy_with_logits(info["existence_logits"], exists)
    return motion, existence, len(pairs)


def one_frame(model, query, dataset, load_gpu, row, state, seq, device):
    points = np.load(POINTS / f"{seq}_{row['frame_id']}.npy")
    batch = batch_from_points(dataset, load_gpu, points, device)
    with torch.no_grad():
        for module in model.module_list:
            if module is model.backbone_2d:
                break
            batch = module(batch)
    batch = model.backbone_2d(batch)
    current = batch["spatial_features_2d"]
    timestamp = int(row["timestamp"]) / 1e6
    enhanced, state, info = query.read(current, state, seq, timestamp)
    batch["spatial_features_2d"] = enhanced
    batch["gt_boxes"] = targets(row, dataset.class_names, device)
    model.dense_head(batch)
    # CenterHead.get_loss mutates heatmaps with sigmoid; decode raw logits first.
    with torch.no_grad():
        proposals = model.dense_head.generate_predicted_boxes(
            1, model.dense_head.forward_ret_dict["pred_dicts"]
        )[0]
    detection, _ = model.dense_head.get_loss()
    motion, existence, matched = query_targets(info, state, row, dataset.class_names, device)
    birth_candidates = int((proposals["pred_scores"] >= query.birth_score).sum())
    update_feature = current if isinstance(query, AttentiveQueryFeedback) else enhanced
    state = query.update(update_feature, state, proposals["pred_boxes"],
                         proposals["pred_scores"], proposals["pred_labels"], timestamp)
    loss = detection + 0.2 * motion + 0.05 * existence
    return loss, state, {"total": float(loss.detach()), "detection": float(detection.detach()),
                         "motion": float(motion.detach()), "existence": float(existence.detach()),
                         "queries": len(state.ids), "matched_queries": matched,
                         "birth_candidates": birth_candidates}


def run_sequences(model, query, dataset, load_gpu, sequences, paths, device,
                  optimizer, clip_length, max_frames):
    training = optimizer is not None
    model.backbone_2d.train(training)
    for module in model.backbone_2d.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    query.train(training)
    set_head_mode(model, True)
    stats = {name: 0.0 for name in ("total", "detection", "motion", "existence", "queries", "matched_queries", "birth_candidates")}
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
                loss, state, values = one_frame(model, query, dataset, load_gpu,
                                                row, state, seq, device)
                clip_losses.append(loss)
                for key in stats:
                    stats[key] += values[key]
                frames += 1
                if training and (len(clip_losses) == clip_length or index == len(rows) - 1):
                    optimizer.zero_grad(set_to_none=True)
                    torch.stack(clip_losses).mean().backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(query.parameters()) + list(model.backbone_2d.parameters())
                        + list(model.dense_head.parameters()), 10.0,
                    )
                    optimizer.step()
                    state = state.detach()
                    clip_losses.clear()
            print(("train" if training else "val"), seq, "frames", len(rows),
                  "loss", round(stats["total"] / max(frames, 1), 4),
                  "queries", round(stats["queries"] / max(frames, 1), 1),
                  "birth_candidates", round(stats["birth_candidates"] / max(frames, 1), 1), flush=True)
    return {**{key: value / max(frames, 1) for key, value in stats.items()},
            "frames": frames, "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", choices=("A", "B"), default="A")
    parser.add_argument("--init", help="Optional detector finetuning checkpoint; memory is not loaded")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--architecture", choices=("point", "attention"), default="point")
    parser.add_argument("--clip-length", type=int, default=4)
    parser.add_argument("--lr-query", type=float, default=3e-4)
    parser.add_argument("--lr-detector", type=float, default=1e-5)
    parser.add_argument("--val-every", type=int, default=2)
    parser.add_argument("--max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    folds = read_json(ROOT / "configs/research/v2xseq_train_folds.json")
    other = "B" if args.fold == "A" else "A"
    train_sequences = list(folds["inner_train"][other])
    val_sequences = list(folds["inner_holdout"][other])
    paths = {item["sequence_id"]: item["path"] for item in read_json(CONVERTED / "manifest.json")["sequences"]}
    model, dataset, load_gpu = load_frozen_centerpoint(
        ROOT / "third_party/OpenPCDet", ROOT / CFG.format(fold=args.fold),
        ROOT / CKPT.format(fold=args.fold), device,
    )
    if args.init:
        initial = torch.load(ROOT / args.init, map_location=device, weights_only=False)
        if initial["args"]["fold"] != args.fold:
            raise ValueError("Initial checkpoint fold does not match requested fold")
        model.backbone_2d.load_state_dict(initial["backbone_2d"])
        model.dense_head.load_state_dict(initial["dense_head"])
    for parameter in model.backbone_2d.parameters():
        parameter.requires_grad_(True)
    for parameter in model.dense_head.parameters():
        parameter.requires_grad_(True)
    query_type = AttentiveQueryFeedback if args.architecture == "attention" else PersistentQueryFeedback
    query = query_type(bev_channels=512, query_channels=64).to(device)
    optimizer = torch.optim.AdamW([
        {"params": query.parameters(), "lr": args.lr_query},
        {"params": model.backbone_2d.parameters(), "lr": args.lr_detector},
        {"params": model.dense_head.parameters(), "lr": args.lr_detector},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    first_epoch = 0
    best_val = float("inf")
    history = []
    if args.resume:
        saved = torch.load(ROOT / args.resume, map_location=device, weights_only=False)
        query.load_state_dict(saved["query"])
        model.backbone_2d.load_state_dict(saved["backbone_2d"])
        model.dense_head.load_state_dict(saved["dense_head"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        first_epoch = saved["epoch"]
        best_val = saved.get("best_val", best_val)
        history = saved.get("history", [])
    start = time.perf_counter()
    for epoch in range(first_epoch, args.epochs):
        random.shuffle(train_sequences)
        train = run_sequences(model, query, dataset, load_gpu, train_sequences, paths,
                              device, optimizer, args.clip_length, args.max_frames_per_sequence)
        val = None
        if (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs:
            val = run_sequences(model, query, dataset, load_gpu, val_sequences, paths,
                                device, None, args.clip_length, args.max_frames_per_sequence)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train": train, "val": val,
                        "lr_query": optimizer.param_groups[0]["lr"]})
        snapshot = {
            "epoch": epoch + 1, "query": query.state_dict(),
            "backbone_2d": model.backbone_2d.state_dict(), "dense_head": model.dense_head.state_dict(),
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
                       "elapsed_seconds": time.perf_counter() - start}, handle, indent=2)
        print("epoch", epoch + 1, "train", train, "val", val,
              "best_val", best_val, "gpu_peak_gb",
              round(torch.cuda.max_memory_allocated(device) / 2**30, 3), flush=True)


if __name__ == "__main__":
    main()
