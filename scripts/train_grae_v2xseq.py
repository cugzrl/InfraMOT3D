import argparse
import pickle
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.io import write_json
from inframot3d.tracking.grae_adapter import build_clips, build_model, train_clip


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grae_centerpoint.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    random.seed(int(config["project"]["seed"]))
    torch.manual_seed(int(config["project"]["seed"]))
    data_root = _resolve(root, config["project"]["grae_data_root"])
    with (data_root / "v2x_seq_train.pkl").open("rb") as stream:
        sequences = pickle.load(stream)
    clips = build_clips(
        sequences,
        config["train"]["clip_len"],
        config["train"]["clip_stride"],
        config["train"]["score_threshold"],
    )
    if not clips:
        raise RuntimeError("没有可用训练片段")
    print("训练片段%d" % len(clips))
    device = torch.device("cuda")
    model = build_model(
        _resolve(root, config["project"]["grae_root"]),
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    ckpt_dir = Path(config["project"]["output_root"]) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epochs = int(config["train"]["epochs"])
    for epoch in range(1, epochs + 1):
        random.shuffle(clips)
        total = 0.0
        count = 0
        for clip in clips:
            optimizer.zero_grad(set_to_none=True)
            loss = train_clip(model, clip, device)
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("loss不是有限值")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        mean_loss = total / max(count, 1)
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "num_classes": int(config["num_classes"]),
            "classes": list(config["classes"]),
        }
        torch.save(state, ckpt_dir / ("checkpoint-epoch%d.pth" % epoch))
        print("epoch %d loss %.6f" % (epoch, mean_loss))
    write_json(ckpt_dir / "train_done.json", {"epochs": epochs, "clips": len(clips)})


if __name__ == "__main__":
    main()
