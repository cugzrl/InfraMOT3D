import os
import shutil
import sys
from pathlib import Path

from inframot3d.evaluation.metrics import identity_f1


def _prepare_import(root):
    root = Path(root)
    dair_v2x = root / "third_party" / "DAIR-V2X" / "v2x"
    evaluate_path = dair_v2x / "AB3DMOT_plugin" / "scripts" / "KITTI" / "evaluate.py"
    if not evaluate_path.is_file():
        raise FileNotFoundError("缺少官方评估代码，先运行 scripts/setup/setup_dair_v2x.sh")
    shim_dir = Path(__file__).resolve().parent / "shims"
    kitti_dir = dair_v2x / "AB3DMOT_plugin" / "scripts" / "KITTI"
    for path in (shim_dir, kitti_dir, dair_v2x):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from AB3DMOT_plugin.scripts.KITTI import evaluate as official_eval
    from AB3DMOT_plugin.scripts.KITTI.mailpy import Mail

    def _skip_plot(self):
        return None

    # 官方绘图不参与指标
    official_eval.stat.plot = _skip_plot
    return dair_v2x, official_eval, Mail


def _stage(dair_v2x, exported, name):
    stage_root = dair_v2x / "AB3DMOT_plugin"
    label_dir = stage_root / "scripts" / "KITTI" / "label"
    result_dir = stage_root / "results" / "KITTI" / name / "data_0"
    if label_dir.exists():
        shutil.rmtree(label_dir)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    label_dir.mkdir(parents=True)
    result_dir.mkdir(parents=True)
    for path in Path(exported).joinpath("label").glob("*.txt"):
        shutil.copy2(path, label_dir / path.name)
    for path in Path(exported).joinpath("pred").glob("*.txt"):
        shutil.copy2(path, result_dir / path.name)
    shutil.copy2(Path(exported) / "evaluate_tracking.seqmap.val", stage_root / "scripts" / "KITTI" / "evaluate_tracking.seqmap.val")
    return stage_root


def run_official_metrics(root, exported, name="unified"):
    dair_v2x, official_eval, Mail = _prepare_import(root)
    stage_root = _stage(dair_v2x, exported, name)
    previous = Path.cwd()
    os.chdir(dair_v2x)
    try:
        mail = Mail("")
        evaluation = official_eval.trackingEvaluation(
            t_sha=name,
            mail=mail,
            cls="car",
            eval_3diou=True,
            eval_2diou=False,
            num_hypo=1,
            thres=0.25,
        )
        if not evaluation.loadTracker():
            raise RuntimeError("官方结果读取失败")
        if not evaluation.loadGroundtruth():
            raise RuntimeError("官方真值读取失败")
        evaluation.compute3rdPartyMetrics()
        best_mota, best_threshold = 0, -10000
        thresholds, recalls = evaluation.getThresholds(evaluation.scores, evaluation.num_gt)
        mota_sum = 0.0
        motp_sum = 0.0
        sample_count = official_eval.num_sample_pts - 1
        for threshold, recall in zip(thresholds, recalls):
            evaluation.reset()
            evaluation.compute3rdPartyMetrics(threshold, recall)
            mota_sum += float(evaluation.MOTA)
            motp_sum += float(evaluation.MOTP)
            if evaluation.MOTA > best_mota:
                best_mota = evaluation.MOTA
                best_threshold = threshold
        evaluation.reset()
        evaluation.compute3rdPartyMetrics(best_threshold)
        idf1 = identity_f1(
            evaluation.gt_trajectories,
            evaluation.ign_trajectories,
            evaluation.n_gt,
            evaluation.tp + evaluation.fp,
        )
        return {
            "MOTA": float(evaluation.MOTA),
            "MOTP": float(evaluation.MOTP),
            "AMOTA": mota_sum / sample_count,
            "AMOTP": motp_sum / sample_count,
            "IDSW": int(evaluation.id_switches),
            "IDF1": float(idf1),
            "FM": int(evaluation.fragments),
            "FP": int(evaluation.fp),
            "FN": int(evaluation.fn),
            "TP": int(evaluation.tp),
            "GT": int(evaluation.n_gt),
            "best_score_threshold": float(best_threshold),
            "summary_dir": str(stage_root / "results" / "KITTI" / name),
        }
    finally:
        os.chdir(previous)


def run_stock_evaluate(root, exported, name="parity_stock"):
    dair_v2x, official_eval, Mail = _prepare_import(root)
    stage_root = _stage(dair_v2x, exported, name)
    previous = Path.cwd()
    os.chdir(dair_v2x)
    captured = {}

    def _save(self, dump, threshold=None, recall=None):
        official_save(self, dump, threshold, recall)
        if threshold is None:
            captured["MOTA"] = float(self.MOTA)
            captured["MOTP"] = float(self.MOTP)
            captured["IDSW"] = int(self.id_switches)
            captured["FM"] = int(self.fragments)
            captured["FP"] = int(self.fp)
            captured["FN"] = int(self.fn)
            captured["TP"] = int(self.tp)
            captured["GT"] = int(self.n_gt)

    def _output(self):
        official_output(self)
        captured["AMOTA"] = float(self.amota)
        captured["AMOTP"] = float(self.amotp)

    official_save = official_eval.trackingEvaluation.saveToStats
    official_output = official_eval.stat.output
    official_eval.trackingEvaluation.saveToStats = _save
    official_eval.stat.output = _output
    try:
        ok = official_eval.evaluate(name, Mail(""), "1", True, False, 0.25)
        if not ok:
            raise RuntimeError("官方评估失败")
        if len(captured) < 10:
            raise RuntimeError("官方原始指标捕获失败")
        captured["summary_path"] = str(stage_root / "results" / "KITTI" / name / "summary_car_average_eval3D.txt")
        return captured
    finally:
        official_eval.trackingEvaluation.saveToStats = official_save
        official_eval.stat.output = official_output
        os.chdir(previous)
