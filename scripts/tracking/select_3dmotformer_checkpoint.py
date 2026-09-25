import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.io import read_json, write_json


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/3dmotformer/centerpoint.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    sequences = [str(value) for value in config["train"]["calibration_sequences"]]
    val_ids = set(read_json(root / config["split_file"])["val"])
    if set(sequences) & val_ids:
        raise SystemExit("calibration序列不能来自val")
    ckpt_dir = Path(config["project"]["output_root"]) / "ckpt"
    checkpoints = sorted(
        ckpt_dir.glob("checkpoint-epoch*.pth"),
        key=lambda path: int(path.stem.replace("checkpoint-epoch", "")),
    )
    if not checkpoints:
        raise SystemExit("缺少3DMOTFormer checkpoint")
    infer = Path(__file__).resolve().parent / "infer_3dmotformer_v2xseq.py"
    work = Path(config["project"]["output_root"]) / "calibration_select"
    records = []
    best = None
    evaluator = UnifiedMOTEvaluator(root)
    for checkpoint in checkpoints:
        epoch = int(checkpoint.stem.replace("checkpoint-epoch", ""))
        prediction_root = work / ("epoch%d" % epoch) / "predictions"
        if prediction_root.parent.exists():
            shutil.rmtree(prediction_root.parent)
        command = [
            sys.executable,
            str(infer),
            "--config",
            args.config,
            "--ckpt",
            str(checkpoint),
            "--split",
            "train",
            "--sequences",
            *sequences,
            "--prediction-root",
            str(prediction_root),
        ]
        completed = __import__("subprocess").run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise SystemExit("epoch %d 推理失败" % epoch)
        metrics = evaluator.evaluate(config, prediction_root, prediction_root.parent, sequences=sequences)
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
    }
    torch.save(state, ckpt_dir / "checkpoint-best.pth")
    write_json(
        ckpt_dir / "checkpoint_selection.json",
        {
            "best_epoch": best[1],
            "best_metric": "idf1",
            "calibration_sequences": sequences,
            "epochs": records,
        },
    )
    print("best epoch %d" % best[1])


if __name__ == "__main__":
    main()
