import re
import shutil
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from inframot3d.config import load_config
from inframot3d.evaluation import UnifiedMOTEvaluator
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking.fastpoly_adapter import FastPolyTracker


SEQUENCES = ["0078", "0087", "0093", "0094"]
GROUPS = (
    (("preprocessing", "SF_thre"), "SF_thre"),
    (("association", "first_thre"), "first_thre"),
    (("life_cycle", "score", "delete_thre"), "delete_thre"),
)


def _resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _lookup(tree, path):
    current = tree
    for key in path:
        current = current[key]
    return current


def _scaled(config, multipliers):
    tuned = deepcopy(config)
    fastpoly = tuned["tracker"]["fastpoly"]
    for path, name in GROUPS:
        parent = fastpoly
        for key in path[:-1]:
            parent = parent[key]
        source = parent[path[-1]]
        parent[path[-1]] = {
            int(key): float(value) * float(multipliers[name]) for key, value in source.items()
        }
    return tuned


def _run(config, sequences, prediction_root):
    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    prediction_root.mkdir(parents=True, exist_ok=True)
    root = config["_root"]
    detection_root = _resolve(root, config["input"]["detection_root"])
    converted_root = Path(config["project"]["converted_root"])
    manifest = read_json(converted_root / "manifest.json")
    by_id = {entry["sequence_id"]: entry for entry in manifest["sequences"]}
    for sequence_id in sequences:
        tracker = FastPolyTracker(
            _resolve(root, config["tracker"]["fastpoly_root"]),
            config["tracker"]["fastpoly"],
            config["tracker"]["class_names"],
        )
        detections = list(read_jsonl(detection_root / ("%s.jsonl" % sequence_id)))
        ground_truth = list(read_jsonl(converted_root / by_id[sequence_id]["path"]))
        by_frame = {row["frame_id"]: row for row in detections}
        rows = []
        for frame in ground_truth:
            row = by_frame[frame["frame_id"]]
            objects = [
                {"class_name": item["class_name"], "score": float(item["score"]), "box": item["box"]}
                for item in row["objects"]
            ]
            outputs = tracker.update(objects, timestamp=int(row["timestamp"]) / 1e6)
            rows.append(
                {
                    "sequence_id": row["sequence_id"],
                    "frame_index": row["frame_index"],
                    "frame_id": row["frame_id"],
                    "timestamp": row["timestamp"],
                    "objects": outputs,
                }
            )
        write_jsonl(prediction_root / ("%s.jsonl" % sequence_id), rows)


def _score(config, multipliers, work, evaluator):
    tuned = _scaled(config, multipliers)
    label = "_".join("%s%s" % (name, multipliers[name]) for _, name in GROUPS)
    prediction_root = work / label / "predictions"
    _run(tuned, SEQUENCES, prediction_root)
    metrics = evaluator.evaluate(tuned, prediction_root, prediction_root.parent, sequences=SEQUENCES)
    return (float(metrics["IDF1"]), float(metrics["MOTA"])), metrics


def _render_yaml(text, multipliers, original):
    # 只改校准副本，基础配置文件保持V2X初始适配
    for _, name in GROUPS:
        values = {int(key): float(value) * float(multipliers[name]) for key, value in original[name].items()}
        body = ", ".join("%d: %.4f" % (key, values[key]) for key in sorted(values))
        text, count = re.subn(r"(%s:\s*)\{[^}]*\}" % name, lambda match: match.group(1) + "{%s}" % body, text, count=1)
        if count != 1:
            raise SystemExit("找不到配置项%s" % name)
    return text


def main():
    config = load_config("configs/trackers/fastpoly/centerpoint.yaml")
    root = config["_root"]
    val_ids = set(read_json(root / config["split_file"])["val"])
    if set(SEQUENCES) & val_ids:
        raise SystemExit("calibration序列不能来自val")
    original = {}
    fastpoly = config["tracker"]["fastpoly"]
    for path, name in GROUPS:
        source = _lookup(fastpoly, path)
        original[name] = {int(key): float(value) for key, value in source.items()}
    evaluator = UnifiedMOTEvaluator(root)
    work = Path(config["project"]["output_root"]) / "calibration"
    multipliers = {name: 1.0 for _, name in GROUPS}
    candidates = []
    best_key, _ = _score(config, multipliers, work, evaluator)
    candidates.append({**multipliers, "idf1": best_key[0], "mota": best_key[1]})
    print("baseline IDF1 %.4f MOTA %.4f" % (best_key[0], best_key[1]))
    for _, name in GROUPS:
        chosen = multipliers[name]
        chosen_key = best_key
        for scale in (0.75, 1.25):
            trial = dict(multipliers)
            trial[name] = scale
            key, _ = _score(config, trial, work, evaluator)
            candidates.append({**trial, "idf1": key[0], "mota": key[1]})
            print("%s x%.2f IDF1 %.4f MOTA %.4f" % (name, scale, key[0], key[1]))
            if key > chosen_key:
                chosen = scale
                chosen_key = key
        multipliers[name] = chosen
        best_key = chosen_key
    base_text = (root / "configs/trackers/fastpoly/centerpoint.yaml").read_text(encoding="utf-8")
    best_path = work / "best_config.yaml"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(_render_yaml(base_text, multipliers, original), encoding="utf-8")
    write_json(
        work / "calibration_results.json",
        {
            "calibration_sequences": SEQUENCES,
            "search_parameters": [name for _, name in GROUPS],
            "scales": [0.75, 1.0, 1.25],
            "method": "coordinate_descent",
            "candidates": candidates,
            "final_multipliers": multipliers,
            "idf1": best_key[0],
            "mota": best_key[1],
            "best_config": str(best_path),
        },
    )
    print("选定倍数 %s IDF1 %.4f MOTA %.4f" % (multipliers, best_key[0], best_key[1]))


if __name__ == "__main__":
    main()
