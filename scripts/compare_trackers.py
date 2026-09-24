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

COLUMNS = (
    "Tracker",
    "MOTA",
    "MOTP",
    "IDF1",
    "IDSW",
    "Frag",
    "Precision",
    "Recall",
    "FPS",
    "Official_MOTA",
    "Official_MOTP",
    "Official_AMOTA",
    "Official_AMOTP",
    "Official_IDS",
)


def _row(name, root):
    summary_path = root / "metrics_summary.json"
    runtime_path = root / "runtime.json"
    if not summary_path.is_file() or not runtime_path.is_file():
        raise FileNotFoundError(f"缺少结果 {root}")
    summary = read_json(summary_path)
    runtime = read_json(runtime_path)
    metrics = summary["overall"]
    official_path = root / "official_metrics.json"
    official = read_json(official_path) if official_path.is_file() else {}
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
        "Official_MOTA": official.get("MOTA", ""),
        "Official_MOTP": official.get("MOTP", ""),
        "Official_AMOTA": official.get("AMOTA", ""),
        "Official_AMOTP": official.get("AMOTP", ""),
        "Official_IDS": official.get("IDS", ""),
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
        "" if row["Official_MOTA"] == "" else f"{float(row['Official_MOTA']):.4f}",
        "" if row["Official_MOTP"] == "" else f"{float(row['Official_MOTP']):.4f}",
        "" if row["Official_AMOTA"] == "" else f"{float(row['Official_AMOTA']):.4f}",
        "" if row["Official_AMOTP"] == "" else f"{float(row['Official_AMOTP']):.4f}",
        "" if row["Official_IDS"] == "" else str(int(row["Official_IDS"])),
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
