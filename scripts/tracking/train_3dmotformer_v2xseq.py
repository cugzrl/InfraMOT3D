import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import write_json
from inframot3d.tracking.motformer_adapter import build_model, configure_runtime, train_mini_sequence


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/3dmotformer/centerpoint.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    motformer_root = _resolve(root, config["project"]["motformer_root"])
    configure_runtime(
        motformer_root,
        config["classes"],
        config["train"]["lidar_interval"],
        config["train"]["max_velo"],
    )
    from dataset.patchwise_dataset import PatchwiseDataset
    from model.loss import Loss
    from torch_geometric.loader import DataLoader

    data_root = _resolve(root, config["project"]["motformer_data_root"])
    dataset = PatchwiseDataset(
        data_root,
        "training",
        int(config["train"]["sample_length"]),
        "nuscenes",
        True,
        float(config["tracker"]["score_threshold"]),
        float(config["model"]["graph_truncation_dist"]),
    )
    kept = []
    for item in dataset.meta:
        usable = True
        for offset in range(dataset.sample_length):
            frame = dataset.data[item["scene_id"]][item["frame_id"] + offset]
            if frame["dets"]["box"].shape[0] == 0:
                usable = False
                break
        if usable:
            kept.append(item)
    dataset.meta = kept
    if len(dataset) < 2:
        raise SystemExit("训练样本过少")
    device = "cuda"
    model = build_model(motformer_root, config["model"], config["num_classes"], device)
    criterion = Loss(
        gamma=float(config["loss"]["gamma"]),
        velo_loss_weight=float(config["loss"]["velo_loss_weight"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["train"]["lr"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        follow_batch=["x"],
        num_workers=0,
        drop_last=True,
    )
    ckpt_dir = Path(config["project"]["output_root"]) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        model.train()
        total = 0.0
        steps = 0
        for batch in loader:
            data_seq = [item.to(device) for item in batch]
            optimizer.zero_grad(set_to_none=True)
            losses = train_mini_sequence(
                model,
                criterion,
                data_seq,
                float(config["tracker"]["active_track_thresh"]),
                int(config["tracker"]["max_age"]),
                1.0,
                float(config["model"]["graph_truncation_dist"]),
            )
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise SystemExit("训练损失不是有限值")
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        mean_loss = total / max(steps, 1)
        path = ckpt_dir / ("checkpoint-epoch%d.pth" % epoch)
        torch.save({"state_dict": model.state_dict(), "epoch": epoch, "loss": mean_loss}, path)
        history.append({"epoch": epoch, "loss": mean_loss, "steps": steps})
        print("epoch %d loss %.6f steps %d" % (epoch, mean_loss, steps))
    write_json(ckpt_dir / "train_history.json", {"epochs": history, "samples": len(dataset)})
    print("训练完成 样本%d" % len(dataset))


if __name__ == "__main__":
    main()
