import argparse
import os
import shutil
import sys
from pathlib import Path

from inframot3d.io import read_json, write_json


def _parse_summary(text):
    mota = motp = ids = None
    amota = amotp = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Multiple Object Tracking Accuracy (MOTA)"):
            mota = float(line.split()[-1])
        elif line.startswith("Multiple Object Tracking Precision (MOTP)"):
            motp = float(line.split()[-1])
        elif line.startswith("ID-switches"):
            ids = int(float(line.split()[-1]))
        elif "AMOTA" in line and "AMOTP" in line and index + 1 < len(lines):
            values = lines[index + 1].split()
            if len(values) >= 3:
                amota = float(values[1])
                amotp = float(values[2])
    if None in (mota, motp, amota, amotp, ids):
        raise RuntimeError("官方摘要解析失败")
    return {"MOTA": mota, "MOTP": motp, "AMOTA": amota, "AMOTP": amotp, "IDS": ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exported", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    exported = Path(args.exported)
    if not exported.is_absolute():
        exported = root / exported
    dair_v2x = root / "third_party" / "DAIR-V2X" / "v2x"
    if not (dair_v2x / "AB3DMOT_plugin" / "scripts" / "KITTI" / "evaluate.py").is_file():
        raise FileNotFoundError("缺少官方评估代码，先运行 scripts/setup_dair_v2x.sh")
    meta = read_json(exported / "export_meta.json")
    stage_root = dair_v2x / "AB3DMOT_plugin"
    label_dir = stage_root / "scripts" / "KITTI" / "label"
    result_dir = stage_root / "results" / "KITTI" / args.name / "data_0"
    if label_dir.exists():
        shutil.rmtree(label_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    label_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    for path in (exported / "label").glob("*.txt"):
        shutil.copy2(path, label_dir / path.name)
    for path in (exported / "pred").glob("*.txt"):
        shutil.copy2(path, result_dir / path.name)
    shutil.copy2(exported / "evaluate_tracking.seqmap.val", stage_root / "scripts" / "KITTI" / "evaluate_tracking.seqmap.val")
    os.chdir(dair_v2x)
    kitti_dir = dair_v2x / "AB3DMOT_plugin" / "scripts" / "KITTI"
    shim_dir = Path(__file__).resolve().parent / "official_import_shims"
    for path in (shim_dir, kitti_dir, dair_v2x):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from AB3DMOT_plugin.scripts.KITTI import evaluate as official_eval
    from AB3DMOT_plugin.scripts.KITTI.mailpy import Mail

    def _skip_plot(self):
        return None

    # 官方绘图不参与指标，避免样本过少时中断
    official_eval.stat.plot = _skip_plot
    ok = official_eval.evaluate(args.name, Mail(""), "1", True, False, 0.25)
    if not ok:
        raise RuntimeError("官方评估失败")
    summary_path = stage_root / "results" / "KITTI" / args.name / "summary_car_average_eval3D.txt"
    parsed = _parse_summary(summary_path.read_text(encoding="utf-8", errors="replace"))
    payload = {
        "evaluator": "DAIR-V2X AB3DMOT_plugin KITTI evaluate",
        "commit": (root / "third_party" / "DAIR-V2X" / "COMMIT").read_text(encoding="utf-8").strip(),
        "supported_classes": meta["supported_classes"],
        "merged_into_car": meta["merged_into_car"],
        "unsupported_classes": meta["unsupported_classes"],
        "iou": meta["iou"],
        "iou_threshold": meta["iou_threshold"],
        "range_filter": "未使用车端extended_range，评估全部路侧标注",
        **parsed,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    write_json(output, payload)
    print(
        "官方Car MOTA %.4f MOTP %.4f AMOTA %.4f AMOTP %.4f IDS %d"
        % (parsed["MOTA"], parsed["MOTP"], parsed["AMOTA"], parsed["AMOTP"], parsed["IDS"])
    )


if __name__ == "__main__":
    main()
