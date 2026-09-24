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

CUSTOM_COLUMNS = ("Tracker", "MOTA", "MOTP", "IDF1", "IDSW", "Frag", "Precision", "Recall")
OFFICIAL_COLUMNS = ("Tracker", "MOTA", "MOTP", "AMOTA", "AMOTP", "IDS")


def _custom_row(name, root):
    summary_path = root / "metrics_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"缺少结果 {root}")
    metrics = read_json(summary_path)["overall"]
    return {
        "Tracker": name,
        "MOTA": float(metrics["mota"]),
        "MOTP": float(metrics["motp"]),
        "IDF1": float(metrics["idf1"]),
        "IDSW": int(metrics["id_switches"]),
        "Frag": int(metrics["fragmentations"]),
        "Precision": float(metrics["precision"]),
        "Recall": float(metrics["recall"]),
    }


def _official_row(name, root):
    official_path = root / "official_metrics.json"
    if not official_path.is_file():
        raise FileNotFoundError(f"缺少官方结果 {root}")
    official = read_json(official_path)
    return {
        "Tracker": name,
        "MOTA": float(official["MOTA"]),
        "MOTP": float(official["MOTP"]),
        "AMOTA": float(official["AMOTA"]),
        "AMOTP": float(official["AMOTP"]),
        "IDS": int(official["IDS"]),
    }


def _format(row, columns):
    rendered = []
    for column in columns:
        value = row[column]
        if column == "Tracker":
            rendered.append(str(value))
        elif column in {"IDSW", "Frag", "IDS"}:
            rendered.append(str(int(value)))
        else:
            rendered.append(f"{float(value):.4f}")
    return rendered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="centerpoint")
    parser.add_argument("--kind", choices=["custom_full_range", "official_v2xseq_car"], default="custom_full_range")
    parser.add_argument("--output", default="outputs/all_class_full_range.csv")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    columns = CUSTOM_COLUMNS if args.kind == "custom_full_range" else OFFICIAL_COLUMNS
    row_fn = _custom_row if args.kind == "custom_full_range" else _official_row
    rows = [row_fn(name, root / relative) for name, relative in PRESETS[args.preset]]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    rendered = [_format(row, columns) for row in rows]
    widths = [max(len(columns[index]), *(len(row[index]) for row in rendered)) for index in range(len(columns))]
    header = "  ".join(columns[index].ljust(widths[index]) for index in range(len(columns)))
    print(header)
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(row[index].ljust(widths[index]) for index in range(len(columns))))
    print(f"已写入 {output}")


if __name__ == "__main__":
    main()
