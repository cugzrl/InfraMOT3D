"""根据离线结果写中文实验总结"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, output_dir


def _read_csv(path):
    with Path(path).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _pick(rows, **query):
    found = []
    for row in rows:
        if all(str(row.get(key)) == str(value) for key, value in query.items()):
            found.append(row)
    return found


def _num(row, key):
    value = row.get(key, "")
    if value == "" or value is None:
        return None
    return float(value)


def _fmt(value):
    if value is None:
        return "无"
    return "%.4f" % value


def main():
    config = load_config()
    out = output_dir(config)
    quality = json.loads((out / "audit" / "quality_report.json").read_text(encoding="utf-8"))
    intersections = _read_csv(out / "audit" / "intersection_table.csv")
    motion_summary = _read_csv(out / "kb" / "motion_summary.csv")
    reliability = json.loads((out / "kb" / "reliability_meta.json").read_text(encoding="utf-8"))
    reliability_rows = _read_csv(out / "kb" / "reliability.csv")
    prediction = _read_csv(out / "motion" / "prediction_metrics.csv")
    motion_brief = json.loads((out / "motion" / "summary.json").read_text(encoding="utf-8"))
    association = _read_csv(out / "association" / "association_metrics.csv")
    baseline = json.loads((out / "association" / "baseline_match.json").read_text(encoding="utf-8"))
    anomalies = json.loads((out / "audit" / "id_anomalies.json").read_text(encoding="utf-8"))
    duplicates = [item for item in anomalies["anomalies"] if item["type"] == "duplicate_id_same_frame"]

    def pred_line(method, horizon, pattern="all"):
        rows = _pick(prediction, fold="all", pattern=pattern, horizon_s=str(horizon), method=method, scope="kv_covered", deployable="1" if method != "kv_topk_oracle" else "0")
        if method == "kv_topk_oracle":
            rows = _pick(prediction, fold="all", pattern=pattern, horizon_s=str(horizon), method=method, scope="kv_covered")
        if not rows:
            return "无"
        row = rows[0]
        return "ADE %s / FDE %s / n %s" % (row["ade"], row["fde"], row["n"])

    def asso(method, slice_name="no_0066", covered="0"):
        rows = _pick(association, slice=slice_name, method=method, covered_only=covered)
        return rows[0] if rows else None

    grae = asso("grae")
    kv = asso("kv")
    cv = asso("cv")
    rel = asso("reliability")
    oracle = asso("oracle")
    kv_cov = asso("kv", covered="1")
    grae_cov = asso("grae", covered="1")
    confident = sum(int(row["confident"]) for row in reliability_rows)
    lines = [
        "# 长期时空知识检索实验总结",
        "",
        "本轮只做数据审查、知识库和离线验证，没有修改 GRAE 在线关联，也没有训练新网络。官方 val 未参与建库和参数选择。",
        "",
        "## 数据",
        "",
        "六个亦庄路口共 %d 条序列。同路口虚拟雷达原点最大偏差 %.3f m，序列内最大漂移 %.3f m，车辆轨迹到车道中心线的中位距离 %.2f m，不同序列可以叠到同一张道路地图上。"
        % (
            sum(int(row["sequences"]) for row in intersections),
            max(float(row["origin_gap_max_m"]) for row in intersections),
            max(float(row["pose_drift_max_m"]) for row in intersections),
            quality["lane_distance"]["p50"],
        ),
        "",
        "按采集时间检查，同路口相邻序列没有时间重叠，inner_train 与 inner_holdout 的间隔都大于 0。检索库只使用对应折的 inner_train，并按完整序列排除待评估序列。",
        "",
        "同帧重复 GT ID：%s。这些 ID 不进入运动库，关联主表另给出去掉 0066 的结果。"
        % "；".join("%s 的 %s 共 %s 帧" % (item["sequence_id"], ",".join(item["track_ids"]), item["frames"]) for item in duplicates),
        "",
        "inner_train 中时长不少于 1 s 的车辆轨迹数量："
        + "，".join("%s %s" % (row["intersection"], row["usable_history_tracks"]) for row in intersections)
        + "。yizhuang13 每折只有一条训练序列，没有内部留出，本轮不对其做检索评估。",
        "",
        "## 知识库",
        "",
        "运动库样本：" + "；".join("折 %s %s 样本 %s 条、独立轨迹 %s" % (row["fold"], row["intersection"], row["samples"], row["tracks"]) for row in motion_summary) + "。",
        "",
        "空间可靠性使用交叉折检测：A 的 inner_train 用在 B 上训练的 CenterPoint，B 则相反。固定已发布分数阈值 %.4f，格子样本少于 %d 时不给出漏检率。有结论的格子 %d 个，因时间重叠被排除的序列 %d 条。"
        % (reliability["operating_score"], reliability["min_count"], confident, len(reliability["excluded_overlap_sequences"])),
        "",
        "## 离线运动预测",
        "",
        "留出查询 %d 个、轨迹 %d 条，60 m 内同类检索覆盖率 %.3f。指标在有检索结果的查询上计算。"
        % (motion_brief["queries"], motion_brief["tracks"], motion_brief["coverage"]),
        "",
        "- 0.5 s 匀速 " + pred_line("cv", 0.5) + "；KV Top-K " + pred_line("kv_topk", 0.5) + "。",
        "- 1 s 匀速 " + pred_line("cv", 1.0) + "；KV Top-K " + pred_line("kv_topk", 1.0) + "。",
        "- 2 s 静止：" + pred_line("static", 2.0),
        "- 2 s 匀速：" + pred_line("cv", 2.0),
        "- 2 s 匀加速：" + pred_line("ca", 2.0),
        "- 2 s KV Top-1：" + pred_line("kv_top1", 2.0),
        "- 2 s KV Top-K：" + pred_line("kv_topk", 2.0),
        "- 2 s KV Top-K Oracle（不可部署）：" + pred_line("kv_topk_oracle", 2.0),
        "",
        "转弯子集 2 s：匀速 " + pred_line("cv", 2.0, "turn") + "；KV Top-K " + pred_line("kv_topk", 2.0, "turn") + "。",
        "",
        "## 离线关联",
        "",
        "候选集合来自原始零速度 GRAE，学习分数保持该次前向的结果，CV、KV 和 Oracle 只替换几何距离。已发布预测逐帧一致的序列：%d / %d。"
        % (sum(1 for value in baseline.values() if value), len(baseline)),
        "",
    ]
    if grae and kv and cv and rel and oracle:
        lines.extend(
            [
                "去掉 0066 后，全部已知正负候选：",
                "",
                "- 原 GRAE：AUC %s，4 m 内负样本 AUC %s，正负分数差 %s，Top-1 %s，正确候选距离中位数 %s m"
                % (_fmt(_num(grae, "auc")), _fmt(_num(grae, "hard_auc")), _fmt(_num(grae, "margin")), _fmt(_num(grae, "top1")), _fmt(_num(grae, "pos_dist_median"))),
                "- GRAE + CV：AUC %s，近负样本 AUC %s，分数差 %s，Top-1 %s，修正 %s，新增错误 %s，距离中位数 %s m"
                % (_fmt(_num(cv, "auc")), _fmt(_num(cv, "hard_auc")), _fmt(_num(cv, "margin")), _fmt(_num(cv, "top1")), cv["corrections"], cv["new_errors"], _fmt(_num(cv, "pos_dist_median"))),
                "- GRAE + KV：AUC %s，近负样本 AUC %s，分数差 %s，Top-1 %s，修正 %s，新增错误 %s，距离中位数 %s m"
                % (_fmt(_num(kv, "auc")), _fmt(_num(kv, "hard_auc")), _fmt(_num(kv, "margin")), _fmt(_num(kv, "top1")), kv["corrections"], kv["new_errors"], _fmt(_num(kv, "pos_dist_median"))),
                "- GRAE + 空间可靠性：AUC %s，近负样本 AUC %s，分数差 %s，Top-1 %s，修正 %s，新增错误 %s"
                % (_fmt(_num(rel, "auc")), _fmt(_num(rel, "hard_auc")), _fmt(_num(rel, "margin")), _fmt(_num(rel, "top1")), rel["corrections"], rel["new_errors"]),
                "- 真实运动 Oracle（不可部署）：AUC %s，近负样本 AUC %s，分数差 %s，Top-1 %s，距离中位数 %s m"
                % (_fmt(_num(oracle, "auc")), _fmt(_num(oracle, "hard_auc")), _fmt(_num(oracle, "margin")), _fmt(_num(oracle, "top1")), _fmt(_num(oracle, "pos_dist_median"))),
                "",
            ]
        )
        if kv_cov and grae_cov:
            lines.append(
                "只看 KV 能够检索到近邻的轨迹：GRAE AUC %s、分离比例 %s；KV AUC %s、分离比例 %s。"
                % (_fmt(_num(grae_cov, "auc")), _fmt(_num(grae_cov, "separation")), _fmt(_num(kv_cov, "auc")), _fmt(_num(kv_cov, "separation")))
            )
            lines.append("")
        grae_auc = _num(grae, "hard_auc")
        kv_auc = _num(kv, "hard_auc")
        grae_margin = _num(grae, "margin")
        kv_margin = _num(kv, "margin")
        kv_dist = _num(kv, "pos_dist_median")
        grae_dist = _num(grae, "pos_dist_median")
        cv_dist = _num(cv, "pos_dist_median")
        oracle_dist = _num(oracle, "pos_dist_median")
        answer = (
            "KV 把正确候选距离中位数从 %s m 降到 %s m，正样本相对最强负样本的分数差从 %s 增到 %s。"
            "4 m 内负样本 AUC 仍是 %s，与原 GRAE 的 %s 相同；Top-1 从 %s 变为 %s，修正 %s 个、新增错误 %s 个。"
            "匀速把距离降到 %s m、分数差增到 %s，修正 %s 个、新增错误 %s 个。"
            "因此 KV 有轻微的位置和分数差收益，但小于匀速，也没有让正确关联在可部署决策里稳定胜过错误关联。"
            "空间可靠性按漏检率直接缩放融合分数后，修正 %s 个、新增错误 %s 个，这个冻结公式不能当成有效先验。"
            "真实运动 Oracle 的距离中位数是 %s m，只说明运动上限，不能部署。"
        ) % (
            _fmt(grae_dist),
            _fmt(kv_dist),
            _fmt(_num(grae, "margin")),
            _fmt(_num(kv, "margin")),
            _fmt(_num(kv, "hard_auc")),
            _fmt(_num(grae, "hard_auc")),
            _fmt(_num(grae, "top1")),
            _fmt(_num(kv, "top1")),
            kv["corrections"],
            kv["new_errors"],
            _fmt(cv_dist),
            _fmt(_num(cv, "margin")),
            cv["corrections"],
            cv["new_errors"],
            rel["corrections"],
            rel["new_errors"],
            _fmt(oracle_dist),
        )
        lines.extend(["## 结论", "", answer, "", "中断 0.5 到 1 s 的可评估轨迹只有个位数，不能据此宣称长时中断已被检索解决。下一步若接入在线 GRAE，应先面对匀速已经强于当前 KV 这一事实，而不是直接训练融合网络。", ""])
    text = "\n".join(lines)
    (out / "SUMMARY.md").write_text(text, encoding="utf-8")
    docs = ROOT / "docs" / "RESEARCH_KNOWLEDGE_20261014.md"
    docs.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
