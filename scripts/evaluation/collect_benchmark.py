import argparse
import csv
from pathlib import Path

import yaml

from inframot3d.config import load_config
from inframot3d.io import read_json


COLUMNS = ("Method", "MOTA", "MOTP", "AMOTA", "AMOTP", "IDSW", "IDF1", "FM", "FP", "FN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiments/v2xseq_centerpoint.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    experiment = yaml.safe_load((root / args.experiment).read_text(encoding="utf-8"))
    rows = []
    for item in experiment["trackers"]:
        config = load_config(item["config"])
        metrics = read_json(Path(config["project"]["output_root"]) / "evaluation" / "metrics.json")
        rows.append(
            {
                "Method": item["name"],
                "MOTA": metrics["MOTA"],
                "MOTP": metrics["MOTP"],
                "AMOTA": metrics["AMOTA"],
                "AMOTP": metrics["AMOTP"],
                "IDSW": metrics["IDSW"],
                "IDF1": metrics["IDF1"],
                "FM": metrics["FM"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
            }
        )
    output = root / experiment["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: row[key] if key in {"Method", "IDSW", "FM", "FP", "FN"} else "%.4f" % float(row[key])
                    for key in COLUMNS
                }
            )
    print("已写入 %s" % output)


if __name__ == "__main__":
    main()
