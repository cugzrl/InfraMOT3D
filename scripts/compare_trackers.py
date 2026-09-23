import argparse
import csv
from pathlib import Path

from inframot3d.io import read_json


PRESETS = {
    "gt": (
        ("AB3DMOT", "outputs/ab3dmot_gt"),
        ("SimpleTrack", "outputs/simpletrack_gt"),
        ("ImmortalTracker", "outputs/immortal_gt"),
    ),
    "centerpoint": (
        ("AB3DMOT", "outputs/ab3dmot_centerpoint"),
        ("SimpleTrack", "outputs/simpletrack_centerpoint"),
        ("ImmortalTracker", "outputs/immortal_centerpoint"),
        ("GRAE-3DMOT", "outputs/grae_centerpoint"),
    ),
    "centerpoint_classic": (
        ("AB3DMOT", "outputs/ab3dmot_centerpoint"),
        ("SimpleTrack", "outputs/simpletrack_centerpoint"),
        ("ImmortalTracker", "outputs/immortal_centerpoint"),
    ),
}

COLUMNS = ("Tracker", "MOTA", "MOTP", "IDF1", "IDSW", "Frag", "Precision", "Recall", "FPS")


def _row(name, root):
    summary_path = root / "metrics_summary.json"
    runtime_path = root / "runtime.json"
    if not summary_path.is_file() or not runtime_path.is_file():
        raise FileNotFoundError(f"缺少结果 {root}")
    summary = read_json(summary_path)
    runtime = read_json(runtime_path)
    metrics = summary["overall"]
    return {
        "Tracker": name,
        "MOTA": float(metrics["mota"]),
        "MOTP": float(metrics["motp"]),
        "IDF1": float(metrics["idf1"]),
        "IDSW": int(metrics["id_switches"]),
        "Frag": int(metrics["fragmentations"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
        "FPS": float(runtime["fps"]),
    }


def _format(row):
    return [
        row["Tracker"],
        f"{row['MOTA']:.4f}",
        f"{row['MOTP']:.4f}",
        f"{row['IDF1']:.4f}",
        str(row["IDSW"]),
        str(row["Frag"]),
        f"{row['Precision']:.4f}",
        f"{row['Recall']:.4f}",
        f"{row['FPS']:.2f}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="gt")
    parser.add_argument("--output", default="outputs/comparison.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = [_row(name, root / relative) for name, relative in PRESETS[args.preset]]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    rendered = [_format(row) for row in rows]
    widths = [max(len(COLUMNS[index]), *(len(row[index]) for row in rendered)) for index in range(len(COLUMNS))]
    header = "  ".join(COLUMNS[index].ljust(widths[index]) for index in range(len(COLUMNS)))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(COLUMNS))))
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
