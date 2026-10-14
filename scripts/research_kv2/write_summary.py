"""根据离线表生成中文总结"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research_knowledge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, output_dir  # noqa: E402


def _read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _num(row, key):
    if row is None or row.get(key) in ("", None):
        return None
    return float(row[key])


def _fmt(value):
    if value is None:
        return "无"
    return "%.4f" % value


def _pick(rows, **query):
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in query.items()):
            return row
    return None


def _cell(rows, method, horizon=2.0, pattern="all", speed="all", sequence_id="all", intersection="all"):
    return _pick(
        rows,
        method=method,
        horizon_s=str(horizon) if isinstance(horizon, str) else ("%s" % horizon),
        pattern=pattern,
        speed_tag=speed,
        sequence_id=sequence_id,
        intersection=intersection,
        slice="all" if sequence_id == "all" and intersection == "all" else (sequence_id if sequence_id != "all" else intersection),
    )


def _line(row):
    if row is None:
        return "无结果"
    return "轨迹 ADE %s / FDE %s / 轨迹 %s / 覆盖 %s / 同轨迹 CV ADE %s" % (
        _fmt(_num(row, "track_ade")),
        _fmt(_num(row, "track_fde")),
        row["n_tracks"],
        _fmt(_num(row, "coverage")),
        _fmt(_num(row, "paired_cv_track_ade")),
    )


def main():
    config = load_config(Path(__file__).resolve().parent / "config.yaml")
    out = output_dir(config)
    motion = _read_csv(out / "motion" / "prediction_metrics.csv")
    quality = json.loads((out / "motion" / "retrieval_quality.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "kb" / "build_audit.json").read_text(encoding="utf-8"))
    # DictWriter 会把 2.0 写成 2.0，有的实现写成 2
    horizons = {row["horizon_s"] for row in motion}
    horizon = "2.0" if "2.0" in horizons else "2"
    half = "0.5" if "0.5" in horizons else "0.50"
    one = "1.0" if "1.0" in horizons else "1"

    def cell(method, pattern="all", speed="all", horizon_s=None):
        return _pick(
            motion,
            slice="all",
            pattern=pattern,
            speed_tag=speed,
            sequence_id="all",
            intersection="all",
            horizon_s=horizon if horizon_s is None else horizon_s,
            method=method,
        )

    methods = [
        ("静止", "static"),
        ("匀速 CV", "cv"),
        ("匀加速 CA", "ca"),
        ("CTRV", "ctrv"),
        ("原 KV Top-1", "base_top1"),
        ("原 KV Top-K 平均", "base_blend"),
        ("5 点历史平均", "hist5_blend"),
        ("10 m 范围平均", "gate10_blend"),
        ("25 m 范围平均", "gate25_blend"),
        ("加角速度平均", "yaw_blend"),
        ("车道及后继平均", "lane_blend"),
        ("按未来模式平均", "mode_blend"),
        ("CV 残差", "residual"),
        ("CV 残差始终修正", "residual_always"),
        ("minADE@K Oracle", "base_minade"),
        ("minFDE@K Oracle", "base_minfde"),
        ("逐时刻最小 Oracle", "base_pointwise"),
    ]
    lines = [
        "# 运动检索改进实验",
        "",
        "本轮只做离线检索、预测对照和困难关联诊断，没有修改正式 GRAE 在线跟踪器，也没有训练知识融合网络。官方 val 未参与建库和参数选择。指标以轨迹宏平均为主，先对同一轨迹的窗口取平均，再对轨迹取平均。",
        "",
        "## 和 R39 的差异",
        "",
        "R39 的 Key 是 1 秒内 5 个点加路径长度速度，检索分是位置、形状和朝向的乘积，默认半径 25 m，并有最低分。它的 Value 存在 0.5 到 2 秒四个点。当前公开评估里的 Top-5 Oracle 用未来 ADE 选整条候选，minFDE@5 用 2 秒误差选整条候选，二者都不可部署。若再按每个时刻取不同候选的最小误差，会把多条轨迹拼成一条，仍然是 Oracle。",
        "",
        "上一轮 InfraMOT3D 使用 11 点历史、60 m 范围、粗道路标签，并把 Top-K 不同转向直接平均。本轮把 Top-1、Top-K 平均、按模式平均、minADE@K、minFDE@K 和逐时刻最小分开报告。",
        "",
        "## 采样检查",
        "",
        "虚拟雷达帧间隔并不严格等于 0.1 s。用均匀 0.1 s 网格做最近邻时，历史窗重复采到同一帧 %d 次，未来窗重复 %d 次，未来点落到锚点及以前 %d 次。这些窗口没有进入本轮知识库，也没有进入查询。"
        % (
            int(audit["dropped"].get("duplicate_history", 0)),
            int(audit["dropped"].get("duplicate_future", 0)),
            int(audit["dropped"].get("future_not_after_anchor", 0)),
        ),
        "",
        "查询 %d 个窗口、%d 条轨迹。查询侧重复历史 %d，未来重复采样 %d，未来点落到锚点及以前 %d。60 m 基线覆盖率 %.3f，Top-K 里与 Top-1 模式不一致的比例 %.3f，候选终点标准差 %.3f m。残差实际触发比例 %.3f。"
        % (
            quality["queries"],
            quality["tracks"],
            int(quality["faults"].get("duplicate_history", 0)),
            int(quality["faults"].get("duplicate_future", 0)),
            int(quality["faults"].get("future_not_after_anchor", 0)),
            float(quality["base"]["coverage"]),
            float(quality["base"]["mode_mismatch"]),
            float(quality["base"]["endpoint_std_m"]),
            float(quality["residual_applied"]),
        ),
        "",
        "HiVT 官方实现取前 50 个时间戳做历史，并以 AV 为中心，还依赖交通灯和协同轨迹。TNT 官方观测长度是 20 帧、预测 30 帧，地图是 Argoverse 或 TFD 矢量图。二者都不是 SPD 的 1 秒历史、2 秒未来和路侧折划分，也没有可直接套用的权重。本轮不训练新网络，因此不报告 HiVT 和 TNT 数值，只加入无需训练的 CTRV。",
        "",
        "## 2 秒预测，全部留出轨迹",
        "",
        "下表是同一批有完整 2 秒未来的轨迹。检索方法的覆盖率低于 1 时，ADE 只在该方法有预测的轨迹上计算，并给出这批轨迹上的 CV。残差方法在不可靠时退回 CV，所以覆盖率接近 1。",
        "",
    ]
    for title, method in methods:
        lines.append("- %s：%s" % (title, _line(cell(method))))
    lines.extend(["", "## 0.5 秒和 1 秒", ""])
    for horizon_s, title in ((half, "0.5 s"), (one, "1 s")):
        lines.append(
            "- %s：CV %s；原 KV 平均 %s；模式平均 %s；残差 %s"
            % (title, _line(cell("cv", horizon_s=horizon_s)), _line(cell("base_blend", horizon_s=horizon_s)), _line(cell("mode_blend", horizon_s=horizon_s)), _line(cell("residual", horizon_s=horizon_s)))
        )
    lines.extend(["", "## 按运动模式，2 秒轨迹 ADE", ""])
    for pattern, title in (("straight", "直行"), ("left", "左转"), ("right", "右转"), ("static", "静止")):
        lines.append(
            "- %s：CV %s；原平均 %s；模式平均 %s；残差 %s；minADE Oracle %s"
            % (
                title,
                _fmt(_num(cell("cv", pattern), "track_ade")),
                _fmt(_num(cell("base_blend", pattern), "track_ade")),
                _fmt(_num(cell("mode_blend", pattern), "track_ade")),
                _fmt(_num(cell("residual", pattern), "track_ade")),
                _fmt(_num(cell("base_minade", pattern), "track_ade")),
            )
        )
    lines.extend(["", "## 直行上的加减速", ""])
    for speed, title in (("cruise", "匀速段"), ("accel", "加速"), ("decel", "减速")):
        lines.append(
            "- %s：CV %s；残差 %s；原平均 %s"
            % (
                title,
                _fmt(_num(cell("cv", "straight", speed), "track_ade")),
                _fmt(_num(cell("residual", "straight", speed), "track_ade")),
                _fmt(_num(cell("base_blend", "straight", speed), "track_ade")),
            )
        )
    assoc_path = out / "association" / "association_metrics.csv"
    lines.extend(["", "## 困难关联", ""])
    if not assoc_path.is_file():
        lines.append("关联诊断尚未完成。")
    else:
        assoc = _read_csv(assoc_path)
        events = json.loads((out / "association" / "hard_events.json").read_text(encoding="utf-8"))
        lines.append(
            "真实序列里，范围内车辆 GT %d，其中没有匹配检测 %d，低分过滤 %d。有正样本的轨迹时刻 %d，其中正样本被门控挡住 %d，4 m 内有负样本 %d，分数差小于 0.05 的模糊时刻 %d，GRAE 分配未命中 %d，身份切换 %d，同一 GT 被多条轨迹占用 %d。"
            % (
                int(events.get("gt_vehicle", 0)),
                int(events.get("gt_miss", 0)),
                int(events.get("gt_filtered", 0)),
                int(events.get("positive_tracks", 0)),
                int(events.get("positive_gated", 0)),
                int(events.get("near_tracks", 0)),
                int(events.get("ambiguous_tracks", 0)),
                int(events.get("grae_assignment_miss", 0)),
                int(events.get("id_switch", 0)),
                int(events.get("fragmented_gt", 0)),
            )
        )
        lines.append("")
        for slice_name, title in (
            ("all", "去掉 0066 的全部已知候选"),
            ("near", "4 m 内存在负样本"),
            ("ambiguous", "正负分数差小于 0.05"),
            ("grae_fail", "原 GRAE 分配失败"),
            ("hard_region", "困难区域"),
        ):
            lines.append("**%s**" % title)
            lines.append("")
            for method, label in (
                ("grae", "原 GRAE"),
                ("cv", "GRAE + CV"),
                ("kv", "GRAE + 原 KV"),
                ("kv_mode", "GRAE + 模式 KV"),
                ("kv_res", "GRAE + CV 残差"),
                ("oracle", "真实运动 Oracle"),
            ):
                row = _pick(assoc, slice=slice_name, source="real", method=method)
                if row is None:
                    continue
                lines.append(
                    "- %s：Top-1 %s，分数差 %s，修正 %s，新增错误 %s，正确距离中位数 %s m，轨迹时刻 %s"
                    % (label, row["top1"], row["margin"], row["corrections"], row["new_errors"], row["pos_dist_median"], row["n_tracks"])
                )
            lines.append("")
        lines.append("受控缺测与真实数据分开统计。缺测期间拿掉该 GT 已匹配的检测，恢复帧再看原轨迹是否接回。")
        lines.append("")
        for row in assoc:
            if row["source"] != "synthetic":
                continue
            lines.append(
                "- %s / %s：恢复命中 %s，修正 %s，新增错误 %s，存活回合 %s，轨迹已丢失 %s"
                % (row["slice"], row["method"], row["top1"], row["corrections"], row["new_errors"], row["n_tracks"], row.get("lost_tracks", ""))
            )
    cv = cell("cv")
    blend = cell("base_blend")
    mode = cell("mode_blend")
    residual = cell("residual")
    point = cell("base_pointwise")
    minade = cell("base_minade")
    cv_ade = _num(cv, "track_ade")
    blend_ade = _num(blend, "track_ade")
    mode_ade = _num(mode, "track_ade")
    res_ade = _num(residual, "track_ade")
    always_ade = _num(cell("residual_always"), "track_ade")
    point_ade = _num(point, "track_ade")
    minade_ade = _num(minade, "track_ade")
    lines.extend(["", "## 三个问题", ""])
    lines.append(
        "1. 原 KV 的 2 秒轨迹 ADE 是 %s，同一批轨迹上的 CV 是 %s。5 点与 11 点几乎一样，10 m 和 25 m 比 60 m 更差，角速度和车道后继也没有拉近 CV。按 Top-1 的未来模式重新平均后是 %s，只比原平均 %s 略好，所以把不同转向平均在一起不是主要原因。始终加上 Top-1 残差是 %s，已经明显好于搬运绝对未来，但仍然差于 CV。整条 minADE@K 是 %s，逐时刻最小是 %s，二者都还高于 CV。因此被验证的原因是：检索到的绝对未来，包括 Top-5 里最好的一条，整体上仍不如当前目标自己的匀速；残差去掉了一部分速度误抄，但 Top-1 本身不够准。均匀 0.1 s 网格还会在不规则帧上重复采样，旧的 11 点并没有 11 个独立位置。"
        % (
            _fmt(blend_ade),
            _fmt(_num(blend, "paired_cv_track_ade")),
            _fmt(mode_ade),
            _fmt(blend_ade),
            _fmt(always_ade),
            _fmt(minade_ade),
            _fmt(point_ade),
        )
    )
    lines.append("")
    left_res, left_cv = _num(cell("residual", "left"), "track_ade"), _num(cell("cv", "left"), "track_ade")
    right_res, right_cv = _num(cell("residual", "right"), "track_ade"), _num(cell("cv", "right"), "track_ade")
    straight_res, straight_cv = _num(cell("residual", "straight"), "track_ade"), _num(cell("cv", "straight"), "track_ade")
    lines.append(
        "2. 带可靠性门控的 CV 残差在 2 秒全部轨迹上是 %s，CV 是 %s，触发比例 %.3f。左转 %s 对 CV %s，右转 %s 对 CV %s，直行 %s 对 CV %s，减速直行 %s 对 CV %s。0.5 秒和 1 秒上残差是 %s、%s，CV 是 %s、%s，短时没有优势。残差只在转弯和减速上低于 CV，直行匀速段更差。无条件残差是 %s，不能在所有样本上替代 CV。CTRV 全部轨迹 ADE 是 %s，也没有稳定超过 CV。"
        % (
            _fmt(res_ade),
            _fmt(cv_ade),
            float(quality["residual_applied"]),
            _fmt(left_res),
            _fmt(left_cv),
            _fmt(right_res),
            _fmt(right_cv),
            _fmt(straight_res),
            _fmt(straight_cv),
            _fmt(_num(cell("residual", "straight", "decel"), "track_ade")),
            _fmt(_num(cell("cv", "straight", "decel"), "track_ade")),
            _fmt(_num(cell("residual", horizon_s=half), "track_ade")),
            _fmt(_num(cell("residual", horizon_s=one), "track_ade")),
            _fmt(_num(cell("cv", horizon_s=half), "track_ade")),
            _fmt(_num(cell("cv", horizon_s=one), "track_ade")),
            _fmt(always_ade),
            _fmt(_num(cell("ctrv"), "track_ade")),
        )
    )
    lines.append("")
    if assoc_path.is_file():
        assoc = _read_csv(assoc_path)
        fail = _pick(assoc, slice="grae_fail", source="real", method="kv_res")
        fail_cv = _pick(assoc, slice="grae_fail", source="real", method="cv")
        fail_kv = _pick(assoc, slice="grae_fail", source="real", method="kv")
        full = _pick(assoc, slice="all", source="real", method="kv_res")
        full_grae = _pick(assoc, slice="all", source="real", method="grae")
        full_cv = _pick(assoc, slice="all", source="real", method="cv")
        full_kv = _pick(assoc, slice="all", source="real", method="kv")
        drop_rows = {
            (row["slice"], row["method"]): row
            for row in assoc
            if row["source"] == "synthetic" and row["method"] in ("grae", "cv", "kv", "kv_res", "oracle")
        }

        def _hit(duration, method):
            row = drop_rows.get(("dropout_%s" % duration, method))
            return "无" if row is None else row["top1"]

        lines.append(
            "3. 真实序列有正样本的时刻是 %s，GRAE 分配失败 %s，身份切换 %s。去掉 0066 后全体 Top-1 仍是 %s，正确距离中位数 %s m，所以候选集合本身很简单。全体时刻上残差修正 %s、新增错误 %s，CV 修正 %s、新增错误 %s，原 KV 修正 %s、新增错误 %s。只看失败时刻时残差修正 %s、CV 修正 %s、原 KV 修正 %s，但失败子集不含原本正确的样本，不能把新增错误算成 0。受控缺测里，0.3 秒恢复命中 GRAE %s、CV %s、原 KV %s、残差 %s；0.5 秒是 %s、%s、%s、%s；1 秒是 %s、%s、%s、%s。残差与 CV 几乎相同，原 KV 几乎不改 GRAE。1 秒 Oracle 命中 %s，只说明真实位置还有余量，不能部署。运动知识降低了一部分位置误差，没有让正确关联稳定胜过错误关联。"
            % (
                events.get("positive_tracks"),
                events.get("grae_assignment_miss"),
                events.get("id_switch"),
                None if full_grae is None else full_grae["top1"],
                None if full_grae is None else full_grae["pos_dist_median"],
                None if full is None else full["corrections"],
                None if full is None else full["new_errors"],
                None if full_cv is None else full_cv["corrections"],
                None if full_cv is None else full_cv["new_errors"],
                None if full_kv is None else full_kv["corrections"],
                None if full_kv is None else full_kv["new_errors"],
                None if fail is None else fail["corrections"],
                None if fail_cv is None else fail_cv["corrections"],
                None if fail_kv is None else fail_kv["corrections"],
                _hit(0.3, "grae"),
                _hit(0.3, "cv"),
                _hit(0.3, "kv"),
                _hit(0.3, "kv_res"),
                _hit(0.5, "grae"),
                _hit(0.5, "cv"),
                _hit(0.5, "kv"),
                _hit(0.5, "kv_res"),
                _hit(1.0, "grae"),
                _hit(1.0, "cv"),
                _hit(1.0, "kv"),
                _hit(1.0, "kv_res"),
                _hit(1.0, "oracle"),
            )
        )
    else:
        lines.append("3. 关联诊断完成后，用失败时刻的修正数和新增错误数判断知识是否改变身份，而不是只看预测距离。")
    lines.extend(["", "本轮到此停止，不把检索写进在线跟踪器。", ""])
    text = "\n".join(lines)
    (out / "SUMMARY.md").write_text(text, encoding="utf-8")
    doc = ROOT / "docs" / "RESEARCH_KV2_20261014.md"
    doc.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
