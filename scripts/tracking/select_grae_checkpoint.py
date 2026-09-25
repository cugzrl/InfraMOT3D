import shutil
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking.grae_adapter import GraeTracker, build_model, load_checkpoint


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _run_epoch(config, root, checkpoint, sequences, prediction_root):
    device = "cuda"
    model = build_model(
        _resolve(root, config["project"]["grae_root"]),
        config["model"]["in_channels"],
        config["model"]["layers"],
        config["num_classes"],
        device,
    )
    load_checkpoint(model, checkpoint, device)
    birth_thresholds = yaml.safe_load(_resolve(root, config["tracker"]["birth_thresholds_file"]).read_text(encoding="utf-8"))[
        "score_thresholds"
    ]
    tracker = GraeTracker(
        model,
        config["classes"],
        birth_thresholds,
        association_alpha=config["tracker"]["association_alpha"],
        age=config["tracker"]["age"],
        score_floor=config["tracker"].get("score_floor", 0.1),
    )
    detection_root = _resolve(root, config["input"]["detection_root"])
    for sequence_id in sequences:
        tracker.reset()
        rows = []
        detections = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        for det_row in detections:
            objects = tracker.update(det_row["objects"], int(det_row["timestamp"]) / 1e6, det_row["frame_id"])
            rows.append(
                {
                    "sequence_id": det_row["sequence_id"],
                    "frame_index": det_row["frame_index"],
                    "frame_id": det_row["frame_id"],
                    "timestamp": det_row["timestamp"],
                    "objects": objects,
                }
            )
        write_jsonl(prediction_root / ("%s.jsonl" % sequence_id), rows)
    del model
    torch.cuda.empty_cache()


def main():
    config = load_config("configs/trackers/grae/centerpoint.yaml")
    root = config["_root"]
    sequences = [str(value) for value in config["train"]["calibration_sequences"]]
    val_ids = set(read_json(root / config["split_file"])["val"])
    if set(sequences) & val_ids:
        raise SystemExit("calibration序列不能来自val")
    ckpt_dir = Path(config["project"]["output_root"]) / "ckpt"
    checkpoints = sorted(ckpt_dir.glob("checkpoint-epoch*.pth"), key=lambda path: int(path.stem.replace("checkpoint-epoch", "")))
    if not checkpoints:
        raise SystemExit("缺少GRAE checkpoint")
    work = Path(config["project"]["output_root"]) / "calibration_select"
    records = []
    best = None
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem.replace("checkpoint-epoch", ""))
        prediction_root = work / ("epoch%d" % epoch) / "predictions"
        if prediction_root.exists():
            shutil.rmtree(prediction_root.parent)
        _run_epoch(config, root, checkpoint, sequences, prediction_root)
        metrics = UnifiedMOTEvaluator(root).evaluate(config, prediction_root, prediction_root.parent, sequences=sequences)
        record = {"epoch": epoch, "idf1": float(metrics["IDF1"]), "mota": float(metrics["MOTA"])}
        records.append(record)
        print("epoch %d IDF1 %.4f MOTA %.4f" % (epoch, record["idf1"], record["mota"]))
        key = (record["idf1"], record["mota"])
        if best is None or key > best[0]:
            best = (key, epoch, checkpoint)
    shutil.copy2(best[2], ckpt_dir / "checkpoint-best.pth")
    state = torch.load(ckpt_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    state["selection"] = {
        "metric": "idf1",
        "epoch": best[1],
        "calibration_sequences": sequences,
        "association_alpha": float(config["tracker"]["association_alpha"]),
    }
    torch.save(state, ckpt_dir / "checkpoint-best.pth")
    write_json(
        ckpt_dir / "checkpoint_selection.json",
        {
            "best_epoch": best[1],
            "best_metric": "idf1",
            "association_alpha": float(config["tracker"]["association_alpha"]),
            "calibration_sequences": sequences,
            "epochs": records,
        },
    )
    write_json(
        ckpt_dir / "train_done.json",
        {
            "epochs": int(config["train"]["epochs"]),
            "best_epoch": best[1],
            "best_metric": "idf1",
            "calibration_sequences": sequences,
        },
    )
    print("best epoch %d" % best[1])


if __name__ == "__main__":
    main()
