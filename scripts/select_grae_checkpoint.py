import shutil
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation import evaluate
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
    score_thresholds = yaml.safe_load(_resolve(root, config["tracker"]["score_thresholds_file"]).read_text(encoding="utf-8"))[
        "score_thresholds"
    ]
    tracker = GraeTracker(
        model,
        config["classes"],
        score_thresholds,
        alpha=config["tracker"]["alpha"],
        age=config["tracker"]["age"],
        score_floor=config["tracker"].get("score_floor", 0.01),
    )
    detection_root = _resolve(root, config["input"]["detection_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = {entry["sequence_id"]: entry for entry in read_json(converted_root / "manifest.json")["sequences"]}
    for sequence_id in sequences:
        tracker.reset()
        entry = manifest[sequence_id]
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
    config = load_config("configs/grae_centerpoint.yaml")
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
        summary, _ = evaluate(
            Path(config["project"]["converted_root"]),
            prediction_root,
            prediction_root.parent,
            config["evaluation"]["iou_thresholds"],
            sequences,
        )
        metrics = summary["overall"]
        record = {"epoch": epoch, "idf1": float(metrics["idf1"]), "mota": float(metrics["mota"])}
        records.append(record)
        print("epoch %d IDF1 %.4f MOTA %.4f" % (epoch, record["idf1"], record["mota"]))
        key = (record["idf1"], record["mota"])
        if best is None or key > best[0]:
            best = (key, epoch, checkpoint)
    shutil.copy2(best[2], ckpt_dir / "checkpoint-best.pth")
    state = torch.load(ckpt_dir / "checkpoint-best.pth", map_location="cpu", weights_only=False)
    state["selection"] = {"metric": "idf1", "epoch": best[1], "calibration_sequences": sequences}
    torch.save(state, ckpt_dir / "checkpoint-best.pth")
    write_json(
        ckpt_dir / "checkpoint_selection.json",
        {"best_epoch": best[1], "best_metric": "idf1", "calibration_sequences": sequences, "epochs": records},
    )
    print("best epoch %d" % best[1])


if __name__ == "__main__":
    main()
