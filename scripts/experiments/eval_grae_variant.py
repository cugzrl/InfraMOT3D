import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation.evaluator import UnifiedMOTEvaluator
from inframot3d.evaluation.hota import evaluate_hota
from inframot3d.evaluation.protocols import build_protocol
from inframot3d.io import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/centerpoint.yaml")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--score-threshold", type=float, default=0.47854848529411764)
    parser.add_argument("--amota", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = config["_root"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evaluator = UnifiedMOTEvaluator(root, "v2xseq")
    metrics = evaluator.evaluate(
        config,
        args.predictions,
        output / "official",
        split=args.split,
        sequences=args.sequences,
        score_threshold=None if args.amota else args.score_threshold,
    )
    if args.sequences:
        sequence_ids = list(args.sequences)
    else:
        sequence_ids = list(read_json(root / config["split_file"])[args.split])
    protocol = build_protocol("v2xseq", root)
    hota = evaluate_hota(config, args.predictions, sequence_ids, protocol, args.score_threshold)
    payload = {"official": metrics, "hota": hota, "frozen_score_threshold": None if args.amota else args.score_threshold}
    write_json(output / "metrics.json", payload)
    print(payload)


if __name__ == "__main__":
    main()
