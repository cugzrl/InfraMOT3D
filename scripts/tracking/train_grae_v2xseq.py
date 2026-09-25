import argparse
import pickle
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.io import write_json
from inframot3d.tracking.grae_adapter import association_metrics, build_clips, build_model, train_clip


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/centerpoint.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    random.seed(int(config["project"]["seed"]))
    torch.manual_seed(int(config["project"]["seed"]))
    data_root = _resolve(root, config["project"]["grae_data_root"])
    with (data_root / "v2x_seq_train.pkl").open("rb") as stream:
        sequences = pickle.load(stream)
    calibration_ids = {str(value) for value in config["train"]["calibration_sequences"]}
    train_sequences = []
    calibration_sequences = []
    for frames in sequences:
        sequence_id = str(frames[0]["sequence_id"]) if frames else ""
        if sequence_id in calibration_ids:
            calibration_sequences.append(frames)
        else:
            train_sequences.append(frames)
    if len(calibration_sequences) != len(calibration_ids):
        raise RuntimeError("calibration序列没有全部出现在训练数据")
    clips = build_clips(
        train_sequences,
        config["train"]["clip_len"],
        config["train"]["clip_stride"],
        config["train"]["score_threshold"],
    )
    calibration_clips = build_clips(
        calibration_sequences,
        config["train"]["clip_len"],
        config["train"]["clip_stride"],
        config["train"]["score_threshold"],
    )
    if not clips or not calibration_clips:
        raise RuntimeError("没有可用训练片段")
    print("训练片段%d 验证片段%d" % (len(clips), len(calibration_clips)))
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
    history = []
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = None
    for epoch in range(1, epochs + 1):
        random.shuffle(clips)
        total = 0.0
        count = 0
        for clip in clips:
            optimizer.zero_grad(set_to_none=True)
            loss = train_clip(model, clip, device)
            if loss is None:
                continue
            if not torch.isfinite(loss):
                raise RuntimeError("loss不是有限值")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            count += 1
        if count == 0:
            raise RuntimeError("本轮没有关联损失")
        mean_loss = total / count
        metrics = association_metrics(model, calibration_clips, device)
        record = {"epoch": epoch, "train_loss": mean_loss}
        record.update(metrics)
        history.append(record)
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "num_classes": int(config["num_classes"]),
            "classes": list(config["classes"]),
            "association": metrics,
        }
        torch.save(state, ckpt_dir / ("checkpoint-epoch%d.pth" % epoch))
        key = (float(metrics["auc"] or -1.0), float(metrics["positive_accuracy"]), float(metrics["recall"]))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(state, ckpt_dir / "checkpoint-best.pth")
        write_json(
            ckpt_dir / "validation_metrics.json",
            {
                "calibration_sequences": sorted(calibration_ids),
                "loss": {"alpha": -1, "gamma": 1.0},
                "epochs": history,
                "best_epoch": best_epoch,
                "best_metric": "auc",
            },
        )
        auc_text = "none" if metrics["auc"] is None else "%.4f" % metrics["auc"]
        print(
            "epoch %d loss %.6f pos %.4f neg %.4f prec %.4f rec %.4f auc %s"
            % (
                epoch,
                mean_loss,
                metrics["positive_accuracy"],
                metrics["negative_accuracy"],
                metrics["precision"],
                metrics["recall"],
                auc_text,
            )
        )
    write_json(
        ckpt_dir / "train_done.json",
        {"epochs": epochs, "clips": len(clips), "best_epoch": best_epoch, "calibration_sequences": sorted(calibration_ids)},
    )


if __name__ == "__main__":
    main()
