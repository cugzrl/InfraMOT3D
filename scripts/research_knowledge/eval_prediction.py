"""内部留出序列上的离线运动预测"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    LaneIndex,
    assert_bank_isolation,
    blend_futures,
    build_catalog,
    constant_acceleration,
    constant_velocity,
    forbidden_sequences,
    future_pattern,
    horizon_index,
    iter_motion_anchors,
    load_bank,
    load_cached_tracks,
    load_config,
    output_dir,
    project_path,
    select_topk,
    static_future,
    write_csv,
    write_json,
)


def _metrics(errors_ade, errors_fde):
    ade = np.asarray(errors_ade, dtype=np.float64)
    fde = np.asarray(errors_fde, dtype=np.float64)
    if ade.size == 0:
        return {"n": 0, "ade": None, "fde": None}
    return {"n": int(ade.size), "ade": float(ade.mean()), "fde": float(fde.mean())}


def evaluate(config):
    out = output_dir(config)
    catalog, _, _ = build_catalog(config)
    tracks = load_cached_tracks(out / "cache" / "tracks.pkl")
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), out / "cache")
    banks = {}
    motion_dir = out / "kb" / "motion"
    for path in sorted(motion_dir.glob("fold*_*.npz")):
        banks[path.stem] = load_bank(path)
    dt = float(config["dt_s"])
    vehicles = set(config["vehicle_classes"])
    horizons = [float(item) for item in config["horizons_s"]]
    queries = []
    seen_sequence = None
    for track in tracks:
        if track["role"] != "inner_holdout" or track["class_name"] not in vehicles:
            continue
        if track["sequence_id"] != seen_sequence:
            seen_sequence = track["sequence_id"]
            print("预测序列 %s" % seen_sequence, flush=True)
        key = "fold%s_%s" % (track["fold"], track["intersection"])
        if key not in banks:
            continue
        bank = banks[key]
        banned = forbidden_sequences(catalog, track["sequence_id"])
        allowed = [name for name in bank["sequence_names"] if str(name) not in banned]
        assert_bank_isolation(allowed, track["sequence_id"], catalog)
        for position, sample, future, future_ok in iter_motion_anchors(track, config, lanes, False):
            query = {
                "origin": sample["origin"],
                "heading": sample["heading"],
                "hist": sample["hist"],
                "speed": sample["speed"],
                "region": sample["region"],
                "class_name": track["class_name"],
            }
            chosen, costs = select_topk(bank, query, allowed, config)
            kv_top1 = None if chosen.size == 0 else bank["future"][chosen[0]]
            kv_topk = None if chosen.size == 0 else blend_futures(bank["future"][chosen], costs)
            oracle = None
            if np.all(future_ok) and chosen.size:
                candidate = bank["future"][chosen]
                ade = np.mean(np.linalg.norm(candidate - future, axis=-1), axis=1)
                oracle = bank["future"][chosen[int(np.argmin(ade))]]
            pattern = future_pattern(future, config) if np.all(future_ok) else "incomplete"
            queries.append(
                {
                    "fold": track["fold"],
                    "intersection": track["intersection"],
                    "sequence_id": track["sequence_id"],
                    "uid": track["uid"],
                    "pattern": pattern,
                    "future": future,
                    "future_ok": future_ok,
                    "covered": int(chosen.size > 0),
                    "neighbors": int(chosen.size),
                    "preds": {
                        "static": static_future(dt),
                        "cv": constant_velocity(sample["hist"], dt),
                        "ca": constant_acceleration(sample["hist"], dt),
                        "kv_top1": kv_top1,
                        "kv_topk": kv_topk,
                        "kv_topk_oracle": oracle,
                    },
                }
            )
    print("预测查询 %d" % len(queries), flush=True)
    return queries, horizons, dt


def aggregate(queries, horizons, dt):
    methods = ["static", "cv", "ca", "kv_top1", "kv_topk", "kv_topk_oracle"]
    deployable = {"static", "cv", "ca", "kv_top1", "kv_topk"}
    rows = []
    groups = [("all", "all", "all")]
    folds = sorted({item["fold"] for item in queries})
    patterns = ["static", "straight", "turn"]
    for fold_name in folds:
        groups.append((fold_name, "all", "all"))
        for pattern in patterns:
            groups.append((fold_name, pattern, "all"))
            groups.append(("all", pattern, "all"))
    seen = set()
    ordered = []
    for item in groups:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    for fold_name, pattern, _ in ordered:
        subset = []
        for item in queries:
            if fold_name != "all" and item["fold"] != fold_name:
                continue
            if pattern != "all" and item["pattern"] != pattern:
                continue
            subset.append(item)
        for horizon in horizons:
            index = horizon_index(horizon, dt)
            valid = []
            for item in subset:
                if index >= len(item["future_ok"]) or not np.all(item["future_ok"][: index + 1]):
                    continue
                valid.append(item)
            covered = [item for item in valid if item["covered"]]
            for method in methods:
                for scope, pool in (("all_valid", valid), ("kv_covered", covered)):
                    if method.startswith("kv") and scope == "all_valid":
                        pool = [item for item in valid if item["preds"][method] is not None]
                    ades = []
                    fdes = []
                    for item in pool:
                        pred = item["preds"][method]
                        if pred is None:
                            continue
                        gt = item["future"][: index + 1]
                        err = np.linalg.norm(pred[: index + 1] - gt, axis=-1)
                        ades.append(float(err.mean()))
                        fdes.append(float(err[-1]))
                    stat = _metrics(ades, fdes)
                    rows.append(
                        {
                            "fold": fold_name,
                            "pattern": pattern,
                            "horizon_s": horizon,
                            "method": method,
                            "deployable": int(method in deployable),
                            "scope": scope,
                            "n_valid": len(valid),
                            "n_tracks": len({item["uid"] for item in valid}),
                            "coverage": 0.0 if not valid else float(np.mean([item["covered"] for item in valid])),
                            "n": stat["n"],
                            "ade": "" if stat["ade"] is None else round(stat["ade"], 4),
                            "fde": "" if stat["fde"] is None else round(stat["fde"], 4),
                        }
                    )
    return rows


def save_motion_cases(queries, dt, folder):
    index = horizon_index(2.0, dt)
    ranked = []
    for item in queries:
        if item["pattern"] != "turn" or not item["covered"] or item["preds"]["kv_topk"] is None:
            continue
        if index >= len(item["future_ok"]) or not np.all(item["future_ok"][: index + 1]):
            continue
        gt = item["future"][: index + 1]
        cv = float(np.mean(np.linalg.norm(item["preds"]["cv"][: index + 1] - gt, axis=-1)))
        kv = float(np.mean(np.linalg.norm(item["preds"]["kv_topk"][: index + 1] - gt, axis=-1)))
        ranked.append((cv - kv, cv, kv, item))
    ranked.sort(key=lambda row: row[0], reverse=True)
    chosen = [("success", row) for row in ranked[:3]] + [("failure", row) for row in ranked[-3:]]
    folder.mkdir(parents=True, exist_ok=True)
    payload = []
    for kind, (gain, cv, kv, item) in chosen:
        gt = item["future"][: index + 1]
        payload.append(
            {
                "kind": kind,
                "sequence_id": item["sequence_id"],
                "uid": item["uid"],
                "ade_cv": round(cv, 4),
                "ade_kv": round(kv, 4),
                "gain": round(gain, 4),
            }
        )
        figure, axis = plt.subplots(figsize=(5.5, 5.5))
        axis.plot(gt[:, 0], gt[:, 1], c="black", label="GT")
        axis.plot(item["preds"]["cv"][: index + 1, 0], item["preds"]["cv"][: index + 1, 1], c="#1f77b4", label="CV")
        axis.plot(item["preds"]["kv_topk"][: index + 1, 0], item["preds"]["kv_topk"][: index + 1, 1], c="#ff7f0e", label="KV")
        axis.scatter([0], [0], c="black", s=20)
        axis.set_aspect("equal")
        axis.set_title("%s %s ADE CV %.2f KV %.2f" % (kind, item["sequence_id"], cv, kv))
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(folder / ("%s_%02d.png" % (kind, len(payload))), dpi=120)
        plt.close(figure)
    write_json(folder / "cases.json", payload)


def main():
    config = load_config()
    queries, horizons, dt = evaluate(config)
    rows = aggregate(queries, horizons, dt)
    out = output_dir(config) / "motion"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "prediction_metrics.csv", rows)
    by_pattern = [row for row in rows if row["fold"] == "all" and row["scope"] == "kv_covered"]
    write_csv(out / "prediction_by_pattern.csv", by_pattern)
    summary = {
        "queries": len(queries),
        "tracks": len({item["uid"] for item in queries}),
        "coverage": float(np.mean([item["covered"] for item in queries])) if queries else 0.0,
        "patterns": dict(Counter(item["pattern"] for item in queries)),
    }
    write_json(out / "summary.json", summary)
    save_motion_cases(queries, dt, out / "cases")
    print("预测完成 查询 %d 覆盖率 %.3f" % (summary["queries"], summary["coverage"]), flush=True)


if __name__ == "__main__":
    main()
