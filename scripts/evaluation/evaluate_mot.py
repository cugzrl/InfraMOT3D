import argparse
from pathlib import Path

from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--prediction-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("split_file", "configs/datasets/v2xseq_sequence_split.json")
    root = config["_root"]
    prediction_root = Path(args.prediction_root) if args.prediction_root else Path(config["project"]["output_root"]) / "predictions"
    if not prediction_root.is_absolute():
        prediction_root = root / prediction_root
    output = Path(args.output) if args.output else Path(config["project"]["output_root"]) / "evaluation"
    if not output.is_absolute():
        output = root / output
    metrics = UnifiedMOTEvaluator(root, config.get("protocol", "v2xseq")).evaluate(
        config, prediction_root, output, split=args.split
    )
    print(
        "MOTA %.4f MOTP %.4f AMOTA %.4f AMOTP %.4f IDSW %d IDF1 %.4f FM %d FP %d FN %d"
        % (
            metrics["MOTA"],
            metrics["MOTP"],
            metrics["AMOTA"],
            metrics["AMOTP"],
            metrics["IDSW"],
            metrics["IDF1"],
            metrics["FM"],
            metrics["FP"],
            metrics["FN"],
        )
    )


if __name__ == "__main__":
    main()
