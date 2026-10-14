"""留出序列上的统一预测对照

Top-K 实际预测不读取未来
minADE、minFDE 和逐时刻最小误差只作为 Oracle
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research_knowledge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    LaneIndex,
    assert_bank_isolation,
    blend_futures,
    build_catalog,
    forbidden_sequences,
    horizon_index,
    load_bank,
    load_cached_tracks,
    load_config,
    make_sample,
    output_dir,
    project_path,
    sample_window,
    static_future,
    write_csv,
    write_json,
)
from build_bank import window_fault  # noqa: E402
from kv import (  # noqa: E402
    BASE_VARIANT,
    LaneGraph,
    ca_path,
    ctrv_path,
    cv_path,
    mode_blend,
    mode_of_future,
    oracle_index,
    pointwise_error,
    residual_forecast,
    sample_hist5,
    select_neighbors,
    speed_tag,
)

VARIANTS = (
    BASE_VARIANT,
    {"name": "hist5", "hist": "hist5", "gate": 60.0, "penalty": "region", "yaw": False},
    {"name": "gate10", "hist": "hist", "gate": 10.0, "penalty": "region", "yaw": False},
    {"name": "gate25", "hist": "hist", "gate": 25.0, "penalty": "region", "yaw": False},
    {"name": "yaw", "hist": "hist", "gate": 60.0, "penalty": "region", "yaw": True},
    {"name": "lane", "hist": "hist", "gate": 60.0, "penalty": "lane", "yaw": False},
)


def method_names():
    names = ["static", "cv", "ca", "ctrv"]
    for variant in VARIANTS:
        names.extend(["%s_top1" % variant["name"], "%s_blend" % variant["name"], "%s_minade" % variant["name"], "%s_minfde" % variant["name"]])
    names.extend(["mode_blend", "base_pointwise", "residual", "residual_always"])
    return names


def _fill_path(ade, fde, method_index, pred, future, future_ok, horizons, dt):
    if pred is None:
        return
    for horizon_i, horizon in enumerate(horizons):
        index = horizon_index(horizon, dt)
        if index >= len(future_ok) or not np.all(future_ok[: index + 1]):
            continue
        if not np.isfinite(pred[: index + 1]).all() or not np.isfinite(future[: index + 1]).all():
            continue
        err = np.linalg.norm(pred[: index + 1] - future[: index + 1], axis=-1)
        ade[method_index, horizon_i] = float(err.mean())
        fde[method_index, horizon_i] = float(err[-1])


def _fill_selected(ade, fde, method_index, candidates, future, future_ok, horizons, dt, by_fde):
    for horizon_i, horizon in enumerate(horizons):
        index = horizon_index(horizon, dt)
        if index >= len(future_ok) or not np.all(future_ok[: index + 1]):
            continue
        piece_c = candidates[:, : index + 1]
        piece_f = future[: index + 1]
        if not np.isfinite(piece_c).all() or not np.isfinite(piece_f).all():
            continue
        pick = oracle_index(piece_c, piece_f, by_fde)
        err = np.linalg.norm(piece_c[pick] - piece_f, axis=-1)
        ade[method_index, horizon_i] = float(err.mean())
        fde[method_index, horizon_i] = float(err[-1])


def _fill_pointwise(ade, fde, method_index, candidates, future, future_ok, horizons, dt):
    for horizon_i, horizon in enumerate(horizons):
        index = horizon_index(horizon, dt)
        if index >= len(future_ok) or not np.all(future_ok[: index + 1]):
            continue
        piece_c = candidates[:, : index + 1]
        piece_f = future[: index + 1]
        if not np.isfinite(piece_c).all() or not np.isfinite(piece_f).all():
            continue
        err = pointwise_error(piece_c, piece_f)
        ade[method_index, horizon_i] = float(err.mean())
        fde[method_index, horizon_i] = float(err[-1])


def evaluate(config):
    out = output_dir(config)
    catalog, _, _ = build_catalog(config)
    tracks = load_cached_tracks(project_path(config, config["source_tracks"]))
    lanes = LaneIndex(project_path(config, config["project"]["map_root"]), project_path(config, config["lane_cache"]))
    lane_graph = LaneGraph(project_path(config, config["project"]["map_root"]))
    banks = {path.stem: load_bank(path) for path in sorted((out / "kb" / "motion").glob("fold*_*.npz"))}
    names = method_names()
    index_of = {name: index for index, name in enumerate(names)}
    horizons = [float(item) for item in config["horizons_s"]]
    dt = float(config["dt_s"])
    steps = int(round(float(config["future_s"]) / dt))
    vehicles = set(config["vehicle_classes"])
    history_s = float(config["history_s"])
    tolerance = float(config["sample_tolerance_s"])
    meta_rows = []
    ades = []
    fdes = []
    applied = []
    faults = {"duplicate_history": 0, "history_after_anchor": 0, "future_not_after_anchor": 0, "duplicate_future": 0}
    quality = {item["name"]: {"windows": 0, "hit": 0, "mode_mismatch": 0.0, "endpoint_std": 0.0} for item in VARIANTS}
    cases = []
    seen = None
    for track in tracks:
        if track["role"] != "inner_holdout" or track["class_name"] not in vehicles:
            continue
        if track["sequence_id"] != seen:
            seen = track["sequence_id"]
            print("预测序列 %s" % seen, flush=True)
        key = "fold%s_%s" % (track["fold"], track["intersection"])
        if key not in banks:
            continue
        bank = banks[key]
        banned = forbidden_sequences(catalog, track["sequence_id"])
        allowed = [str(name) for name in bank["sequence_names"] if str(name) not in banned]
        assert_bank_isolation(allowed, track["sequence_id"], catalog)
        if str(track["sequence_id"]) in {str(name) for name in bank["sequence_names"]}:
            raise AssertionError("查询序列进入了检索库 %s" % track["sequence_id"])
        times = track["time"]
        for position in range(len(times)):
            anchor = float(times[position])
            if anchor - float(times[0]) < history_s - tolerance:
                continue
            hist_index, hist_ok, fut_index, fut_ok = sample_window(
                track, anchor, history_s, float(config["future_s"]), float(dt), tolerance
            )
            if not np.all(hist_ok) or int(hist_index[-1]) != position:
                continue
            fault = window_fault(track, anchor, hist_index, fut_index, fut_ok)
            if fault:
                faults[fault] += 1
                continue
            sample = make_sample(track, position, hist_index, lanes, config)
            future = np.full((fut_index.shape[0], 2), np.nan, dtype=np.float64)
            if np.any(fut_ok):
                from common import to_local

                future[fut_ok] = to_local(track["world_xy"][fut_index[fut_ok]], sample["origin"], sample["heading"])
            query = {
                "origin": sample["origin"],
                "heading": sample["heading"],
                "hist": sample["hist"],
                "hist5": sample_hist5(sample["hist"], dt),
                "speed": sample["speed"],
                "yaw_rate": sample["yaw_rate"],
                "region": sample["region"],
                "class_name": track["class_name"],
                "intersection": track["intersection"],
                "lane": lane_graph.code_at(track["intersection"], sample["origin"], config["retrieval"]["lane_radius_m"]),
            }
            ade = np.full((len(names), len(horizons)), np.nan, dtype=np.float64)
            fde = np.full_like(ade, np.nan)
            query_cv = cv_path(sample["hist"], dt, steps)
            _fill_path(ade, fde, index_of["static"], static_future(dt), future, fut_ok, horizons, dt)
            _fill_path(ade, fde, index_of["cv"], query_cv, future, fut_ok, horizons, dt)
            _fill_path(ade, fde, index_of["ca"], ca_path(sample["hist"], dt, steps), future, fut_ok, horizons, dt)
            _fill_path(ade, fde, index_of["ctrv"], ctrv_path(sample["hist"], dt, steps), future, fut_ok, horizons, dt)
            packs = {}
            for variant in VARIANTS:
                chosen, costs = select_neighbors(bank, query, allowed, variant, config, lane_graph)
                packs[variant["name"]] = (chosen, costs)
                stat = quality[variant["name"]]
                stat["windows"] += 1
                stat["hit"] += int(chosen.size > 0)
                if chosen.size:
                    modes = bank["mode"][chosen]
                    stat["mode_mismatch"] += float(np.mean(modes != modes[0]))
                    ends = bank["future"][chosen, -1]
                    stat["endpoint_std"] += float(np.linalg.norm(np.std(ends, axis=0))) if chosen.size > 1 else 0.0
                    _fill_path(ade, fde, index_of["%s_top1" % variant["name"]], bank["future"][chosen[0]], future, fut_ok, horizons, dt)
                    _fill_path(
                        ade,
                        fde,
                        index_of["%s_blend" % variant["name"]],
                        blend_futures(bank["future"][chosen], costs),
                        future,
                        fut_ok,
                        horizons,
                        dt,
                    )
                    _fill_selected(ade, fde, index_of["%s_minade" % variant["name"]], bank["future"][chosen], future, fut_ok, horizons, dt, False)
                    _fill_selected(ade, fde, index_of["%s_minfde" % variant["name"]], bank["future"][chosen], future, fut_ok, horizons, dt, True)
            base_idx, base_cost = packs["base"]
            used = False
            residual_path = query_cv
            if base_idx.size:
                _fill_path(ade, fde, index_of["mode_blend"], mode_blend(bank["future"][base_idx], base_cost, bank["mode"][base_idx]), future, fut_ok, horizons, dt)
                _fill_pointwise(ade, fde, index_of["base_pointwise"], bank["future"][base_idx], future, fut_ok, horizons, dt)
                residual_path, used = residual_forecast(query_cv, bank["residual"][base_idx], base_cost, bank["mode"][base_idx], config)
                always = query_cv + np.asarray(bank["residual"][base_idx[0]], dtype=np.float64)
            else:
                always = query_cv
            _fill_path(ade, fde, index_of["residual"], residual_path, future, fut_ok, horizons, dt)
            _fill_path(ade, fde, index_of["residual_always"], always, future, fut_ok, horizons, dt)
            pattern = "incomplete"
            if np.all(fut_ok):
                code = mode_of_future(future, config)
                pattern = {0: "static", 1: "straight", 2: "left", 3: "right"}.get(code, "incomplete")
            meta_rows.append(
                {
                    "uid": track["uid"],
                    "sequence_id": track["sequence_id"],
                    "intersection": track["intersection"],
                    "fold": track["fold"],
                    "pattern": pattern,
                    "speed_tag": speed_tag(future, sample["speed"], config) if np.all(fut_ok) else "na",
                }
            )
            ades.append(ade)
            fdes.append(fde)
            applied.append(used)
            if pattern in ("left", "right") and np.all(fut_ok) and base_idx.size:
                gain = float(ade[index_of["cv"], -1] - ade[index_of["residual"], -1])
                if np.isfinite(gain):
                    cases.append((gain, track["uid"], sample["hist"], future, query_cv, bank["future"][base_idx[0]], residual_path, used))
    print("预测查询 %d" % len(meta_rows), flush=True)
    return {
        "names": names,
        "horizons": horizons,
        "meta": meta_rows,
        "ade": np.stack(ades),
        "fde": np.stack(fdes),
        "applied": np.asarray(applied, dtype=bool),
        "faults": faults,
        "quality": quality,
        "cases": cases,
    }


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, 0
    return float(values.mean()), int(values.size)


def _track_mean(uids, values):
    buckets = {}
    for uid, value in zip(uids, values):
        if not np.isfinite(value):
            continue
        buckets.setdefault(uid, []).append(float(value))
    if not buckets:
        return None, 0
    return float(np.mean([float(np.mean(item)) for item in buckets.values()])), len(buckets)


def _oracle_flag(name):
    return int(any(token in name for token in ("minade", "minfde", "pointwise")))


def aggregate(payload):
    names = payload["names"]
    horizons = payload["horizons"]
    meta = payload["meta"]
    ade = payload["ade"]
    fde = payload["fde"]
    rows = []
    slices = [("all", "all", "all", "all", "all")]
    intersections = sorted({item["intersection"] for item in meta})
    sequences = sorted({item["sequence_id"] for item in meta})
    patterns = ["static", "straight", "left", "right"]
    speeds = ["cruise", "accel", "decel"]
    for intersection in intersections:
        slices.append((intersection, "all", "all", "all", intersection))
    for sequence_id in sequences:
        slices.append((sequence_id, "all", "all", sequence_id, "all"))
    for pattern in patterns:
        slices.append(("all", pattern, "all", "all", "all"))
    for pattern in ("straight",):
        for speed in speeds:
            slices.append(("all", pattern, speed, "all", "all"))
    for label, pattern, speed, sequence_id, intersection in slices:
        chosen = []
        for index, item in enumerate(meta):
            if pattern != "all" and item["pattern"] != pattern:
                continue
            if speed != "all" and item["speed_tag"] != speed:
                continue
            if sequence_id != "all" and item["sequence_id"] != sequence_id:
                continue
            if intersection != "all" and item["intersection"] != intersection:
                continue
            chosen.append(index)
        if not chosen:
            continue
        chosen = np.asarray(chosen, dtype=np.int64)
        uids = [meta[int(index)]["uid"] for index in chosen]
        for horizon_i, horizon in enumerate(horizons):
            for method_i, method in enumerate(names):
                method_ade = ade[chosen, method_i, horizon_i]
                method_fde = fde[chosen, method_i, horizon_i]
                cv_ade = ade[chosen, names.index("cv"), horizon_i]
                cv_fde = fde[chosen, names.index("cv"), horizon_i]
                paired = np.isfinite(method_ade) & np.isfinite(cv_ade)
                window_ade, window_n = _mean(method_ade[paired])
                window_fde, _ = _mean(method_fde[paired])
                track_ade, track_n = _track_mean([uid for uid, keep in zip(uids, paired) if keep], method_ade[paired])
                track_fde, _ = _track_mean([uid for uid, keep in zip(uids, paired) if keep], method_fde[paired])
                paired_cv_ade, _ = _track_mean([uid for uid, keep in zip(uids, paired) if keep], cv_ade[paired])
                paired_cv_fde, _ = _track_mean([uid for uid, keep in zip(uids, paired) if keep], cv_fde[paired])
                eligible = np.isfinite(cv_ade)
                coverage = 0.0 if not eligible.any() else float(np.mean(np.isfinite(method_ade[eligible])))
                rows.append(
                    {
                        "slice": label,
                        "pattern": pattern,
                        "speed_tag": speed,
                        "sequence_id": sequence_id,
                        "intersection": intersection,
                        "horizon_s": horizon,
                        "method": method,
                        "deployable": int(_oracle_flag(method) == 0),
                        "oracle": _oracle_flag(method),
                        "n_windows": window_n,
                        "n_tracks": track_n,
                        "coverage": round(coverage, 4),
                        "window_ade": "" if window_ade is None else round(window_ade, 4),
                        "window_fde": "" if window_fde is None else round(window_fde, 4),
                        "track_ade": "" if track_ade is None else round(track_ade, 4),
                        "track_fde": "" if track_fde is None else round(track_fde, 4),
                        "paired_cv_track_ade": "" if paired_cv_ade is None else round(paired_cv_ade, 4),
                        "paired_cv_track_fde": "" if paired_cv_fde is None else round(paired_cv_fde, 4),
                    }
                )
    return rows


def save_cases(cases, folder):
    folder.mkdir(parents=True, exist_ok=True)
    cases = sorted(cases, key=lambda item: item[0])
    picked = [("failure", item) for item in cases[:3]] + [("success", item) for item in cases[-3:]]
    payload = []
    for kind, item in picked:
        gain, uid, hist, future, cv, top1, residual, used = item
        figure, axis = plt.subplots(figsize=(6.2, 5.2))
        axis.plot(hist[:, 0], hist[:, 1], color="0.4", label="history")
        axis.plot(np.r_[0.0, future[:, 0]], np.r_[0.0, future[:, 1]], color="black", label="GT")
        axis.plot(np.r_[0.0, cv[:, 0]], np.r_[0.0, cv[:, 1]], label="CV")
        axis.plot(np.r_[0.0, top1[:, 0]], np.r_[0.0, top1[:, 1]], label="KV Top-1")
        axis.plot(np.r_[0.0, residual[:, 0]], np.r_[0.0, residual[:, 1]], label="residual")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_title("%s %s gain %.2f applied %d" % (kind, uid, gain, int(used)))
        axis.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(folder / ("%s_%02d.png" % (kind, len(payload))), dpi=120)
        plt.close(figure)
        payload.append({"kind": kind, "uid": uid, "gain_ade": round(float(gain), 4), "residual_applied": int(used)})
    write_json(folder / "cases.json", payload)


def main():
    config = load_config(Path(__file__).resolve().parent / "config.yaml")
    payload = evaluate(config)
    out = output_dir(config) / "motion"
    out.mkdir(parents=True, exist_ok=True)
    rows = aggregate(payload)
    write_csv(out / "prediction_metrics.csv", rows)
    quality = {}
    for name, stat in payload["quality"].items():
        hit = max(int(stat["hit"]), 1)
        quality[name] = {
            "windows": int(stat["windows"]),
            "coverage": 0.0 if stat["windows"] == 0 else stat["hit"] / float(stat["windows"]),
            "mode_mismatch": 0.0 if stat["hit"] == 0 else stat["mode_mismatch"] / float(hit),
            "endpoint_std_m": 0.0 if stat["hit"] == 0 else stat["endpoint_std"] / float(hit),
        }
    quality["residual_applied"] = float(payload["applied"].mean()) if len(payload["applied"]) else 0.0
    quality["faults"] = payload["faults"]
    quality["queries"] = len(payload["meta"])
    quality["tracks"] = len({item["uid"] for item in payload["meta"]})
    write_json(out / "retrieval_quality.json", quality)
    save_cases(payload["cases"], out / "cases")
    print("预测完成 查询 %d 轨迹 %d" % (quality["queries"], quality["tracks"]), flush=True)


if __name__ == "__main__":
    main()
