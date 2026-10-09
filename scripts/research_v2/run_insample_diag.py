"""in-sample 诊断 B1 / B2：全量检测器 + fold B 内部留出

正式结果必须等交叉拟合检测器，本脚本产物目录带 insample 前缀
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "outputs/research/roadside_joint_perception_v2"
PY = sys.executable
FOLDS = json.loads((ROOT / "configs/research/v2xseq_train_folds.json").read_text())
TRAIN = FOLDS["inner_train"]["B"]
HOLD = FOLDS["inner_holdout"]["B"]
SEQS = TRAIN + HOLD
SAMPLES = R / "samples/full_insample"
BEV = R / "bev_cache/full"
DET = ROOT / "outputs/centerpoint/detections"
REPLAY = R / "replays/full_train_insample"
LOG = R / "logs/insample_diag.log"


def run(cmd):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write("$ " + " ".join(map(str, cmd)) + "\n")
        stream.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    env_gpu = os.environ.copy()
    env_gpu["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    env_gpu["PYTHONPATH"] = "%s/src:%s/third_party/GRAE-3DMOT:%s/third_party/OpenPCDet" % (ROOT, ROOT, ROOT)
    os.environ.update({k: env_gpu[k] for k in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")})
    missing = [s for s in SEQS if not (SAMPLES / ("%s.pkl" % s)).exists()]
    if missing:
        run([PY, "scripts/research_v2/build_samples.py", "--detections", str(DET), "--replay", str(REPLAY), "--sequences", *missing, "--output", str(SAMPLES)])
    jobs = [
        ("hist_bev_qgt_s0", ["--variant", "hist_bev", "--query", "gt"]),
        ("hist_bev_qgrae_s0", ["--variant", "hist_bev", "--query", "grae"]),
        ("hist_bev_qgrae_pert_s0", ["--variant", "hist_bev", "--query", "grae", "--perturb"]),
        ("cand_qgrae_s0", ["--variant", "cand", "--query", "grae"]),
        ("hist_qgrae_s0", ["--variant", "hist", "--query", "grae"]),
        ("bev_qgrae_s0", ["--variant", "bev", "--query", "grae"]),
        ("hist_bev_nosite_qgrae_s0", ["--variant", "hist_bev_nosite", "--query", "grae"]),
    ]
    summary = []
    for name, extra in jobs:
        ckpt = R / "checkpoints/insample" / name
        if not (ckpt / "last.pth").exists():
            cmd = [PY, "scripts/research_v2/train_tcpn.py", "--samples", str(SAMPLES), "--bev", str(BEV), "--train", *TRAIN, "--holdout", *HOLD, "--eval-query", "grae", "gt", "--seed", "0", "--epochs", "8", "--workers", "2", "--out", str(ckpt), *extra]
            run(cmd)
        hist = json.loads((ckpt / "history.json").read_text())
        last = hist["history"][-1]
        h = last.get("holdout_grae", {})
        row = {
            "name": name,
            "params": hist.get("params"),
            "train_loss": last["train"]["total"],
            "raw_ap": h.get("raw", {}).get("AP"),
            "model_ap": h.get("model", {}).get("AP"),
            "raw_r01": h.get("raw", {}).get("recall@fp0.10"),
            "model_r01": h.get("model", {}).get("recall@fp0.10"),
            "raw_r025": h.get("raw", {}).get("recall@fp0.25"),
            "model_r025": h.get("model", {}).get("recall@fp0.25"),
            "low_raw_ap": h.get("low_raw", {}).get("AP"),
            "low_model_ap": h.get("low_model", {}).get("AP"),
            "quality_auc": h.get("quality_auc"),
            "raw_auc": h.get("raw_auc"),
            "exist_auc": h.get("exist_auc"),
        }
        summary.append(row)
        print(json.dumps(row), flush=True)
    (R / "diagnostics/insample_b1b2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("insample diag done")


if __name__ == "__main__":
    main()
