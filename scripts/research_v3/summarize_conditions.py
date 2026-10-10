"""Report paired per-sequence HOTA for the frozen D0/D1/D2 comparison."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols import build_protocol
from inframot3d.io import read_json, write_json

THRESHOLD = 0.47854848529411764
BASE = "outputs/research/fixed_view_memory_p2"


def main():
    cfg = load_config("configs/trackers/grae/centerpoint.yaml")
    protocol = build_protocol("v2xseq", ROOT)
    folds = read_json(ROOT / "configs/research/v2xseq_train_folds.json")
    results = {}
    for name, fold in (("pA", "B"), ("pB", "A")):
        seqs = folds["inner_holdout"][fold]
        results[name] = {}
        for seq in seqs:
            row = {}
            for condition in ("D0", "D1_mean3_w020", "D2_motion_w020"):
                prediction_root = f"{BASE}/{name}_{condition}_replay/predictions"
                row[condition] = evaluate_hota(cfg, prediction_root, [seq], protocol, THRESHOLD)
            row["dHOTA_D1_D0"] = row["D1_mean3_w020"]["HOTA"] - row["D0"]["HOTA"]
            row["dHOTA_D2_D0"] = row["D2_motion_w020"]["HOTA"] - row["D0"]["HOTA"]
            row["dHOTA_D2_D1"] = row["D2_motion_w020"]["HOTA"] - row["D1_mean3_w020"]["HOTA"]
            results[name][seq] = row
            print(name, seq, *(f"{row[x]['HOTA']:.6f}" for x in ("D0", "D1_mean3_w020", "D2_motion_w020")), flush=True)
    write_json(ROOT / BASE / "paired_sequence_hota.json", {"threshold": THRESHOLD, "per_sequence": results})


if __name__ == "__main__":
    main()
