import argparse
from pathlib import Path

from inframot3d.config import load_config
from inframot3d.evaluation import evaluate
from inframot3d.io import read_json


def _allowed_sequences(config, sequences, split):
    allowed = list(sequences) if sequences else None
    if not split:
        return allowed
    split_file = Path(config.get("split_file", "configs/v2xseq_sequence_split.json"))
    if not split_file.is_absolute():
        split_file = config["_root"] / split_file
    split_ids = set(read_json(split_file)[split])
    if allowed is None:
        return sorted(split_ids)
    return sorted(set(allowed) & split_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ab3dmot_gt.yaml")
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--split", choices=["train", "val", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = config["project"]["output_root"]
    summary, _ = evaluate(
        config["project"]["converted_root"],
        output_root / "predictions",
        output_root,
        config["evaluation"]["iou_thresholds"],
        _allowed_sequences(config, args.sequences, args.split),
    )
    metrics = summary["overall"]
    print(
        f"评估完成 MOTA{metrics['mota']:.4f} IDF1{metrics['idf1']:.4f} "
        f"Recall{metrics['recall']:.4f} IDSW{metrics['id_switches']}"
    )


if __name__ == "__main__":
    main()
