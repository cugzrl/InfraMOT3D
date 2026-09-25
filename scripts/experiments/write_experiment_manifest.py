import argparse
import subprocess
import time
from pathlib import Path

import yaml

from inframot3d.config import load_config
from inframot3d.io import read_json, write_json
from inframot3d.evaluation.protocols.v2xseq import OFFICIAL_RANGE


def _git_commit(root):
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _worktree_dirty(root):
    result = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True)
    return bool(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--experiment", default="configs/experiments/v2xseq_centerpoint.yaml")
    parser.add_argument("--output", default="outputs/benchmark/experiment_manifest.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    experiment = yaml.safe_load((root / args.experiment).read_text(encoding="utf-8"))
    grae_config = load_config("configs/trackers/grae/centerpoint.yaml")
    detection_manifest = read_json(root / "outputs" / "centerpoint" / "detection_manifest.json")
    selection = read_json(Path(grae_config["project"]["output_root"]) / "ckpt" / "checkpoint_selection.json")
    birth_thresholds = yaml.safe_load((root / experiment["birth_thresholds"]).read_text(encoding="utf-8"))["score_thresholds"]
    val_ids = read_json(root / grae_config["split_file"])["val"]
    payload = {
        "inframot3d_commit": _git_commit(root),
        "worktree_dirty": _worktree_dirty(root),
        "centerpoint_checkpoint": detection_manifest["checkpoint"],
        "grae_checkpoint": str(Path(grae_config["project"]["output_root"]) / "ckpt" / "checkpoint-best.pth"),
        "grae_epoch": int(selection["best_epoch"]),
        "birth_thresholds": {str(name): float(value) for name, value in birth_thresholds.items()},
        "association_alpha": float(grae_config["tracker"]["association_alpha"]),
        "score_floor": float(grae_config["tracker"]["score_floor"]),
        "dair_v2x_commit": (root / "third_party" / "DAIR-V2X" / "COMMIT").read_text(encoding="utf-8").strip(),
        "val_sequences": len(val_ids),
        "protocol": experiment["protocol"],
        "evaluation_range": list(OFFICIAL_RANGE),
        "iou_threshold": 0.25,
        "benchmark_csv": experiment["output"],
        "elapsed_seconds": float(args.elapsed_seconds),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, payload)
    print("已写入 %s" % output)


if __name__ == "__main__":
    main()
