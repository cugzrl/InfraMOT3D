import argparse

from inframot3d.config import load_config
from inframot3d.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ab3dmot_gt.yaml")
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = config["project"]["output_root"]
    summary, _ = evaluate(
        config["project"]["converted_root"],
        output_root / "predictions",
        output_root,
        config["evaluation"]["iou_thresholds"],
        args.sequences,
    )
    metrics = summary["overall"]
    print(
        f"评估完成 MOTA{metrics['mota']:.4f} IDF1{metrics['idf1']:.4f} "
        f"Recall{metrics['recall']:.4f} IDSW{metrics['id_switches']}"
    )


if __name__ == "__main__":
    main()
