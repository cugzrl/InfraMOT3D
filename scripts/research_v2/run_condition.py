"""单个实验条件：训练（已存在则跳过）→ 闭环推理 → 检测评估 → 跟踪评估

管线 F 使用检测器 F，训练与留出序列都来自另一折，正式 val 只在 --split val 时使用
基线条件 raw / nms / iso_cls 不训练，直接在相同候选池上重打分后送入原始 GRAE
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = "outputs/research/roadside_joint_perception_v2"
PY = sys.executable
BASELINES = ("raw", "nms", "iso_cls")


def run(cmd, env, log):
    with open(log, "a") as stream:
        stream.write("$ " + " ".join(cmd) + "\n")
        stream.flush()
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, choices=["A", "B"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--query", default="grae")
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--split", default="holdout", choices=["holdout", "val"])
    parser.add_argument("--tag", default="")
    parser.add_argument("--apply-box", action="store_true")
    parser.add_argument("--bev-pattern", default=None)
    args = parser.parse_args()
    if args.bev_pattern:
        args.tag += "_pat%s" % args.bev_pattern
    folds = json.loads((ROOT / "configs/research/v2xseq_train_folds.json").read_text())
    other = "B" if args.pipeline == "A" else "A"
    train = folds["inner_train"][other]
    holdout = folds["inner_holdout"][other]
    val = json.loads((ROOT / "configs/datasets/v2xseq_sequence_split.json").read_text())["val"]
    seqs = holdout if args.split == "holdout" else val
    det = "%s/detections/det%s" % (R, args.pipeline)
    samples = "%s/samples/det%s" % (R, args.pipeline)
    bev = "%s/bev_cache/det%s" % (R, args.pipeline)
    name = args.variant
    if args.variant not in BASELINES:
        name = "%s_q%s%s_s%d%s" % (args.variant, args.query, "_pert" if args.perturb else "", args.seed, args.tag)
    ckpt_dir = "%s/checkpoints/p%s/%s" % (R, args.pipeline, name)
    pred_dir = "%s/predictions/p%s/%s/%s%s" % (R, args.pipeline, args.split, name, "_box" if args.apply_box else "")
    (ROOT / pred_dir).mkdir(parents=True, exist_ok=True)
    log = ROOT / pred_dir / "run.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONPATH="%s/src:%s/third_party/GRAE-3DMOT:%s/third_party/OpenPCDet" % (ROOT, ROOT, ROOT))
    if args.variant not in BASELINES:
        if not (ROOT / ckpt_dir / "last.pth").exists():
            cmd = [PY, "scripts/research_v2/train_tcpn.py", "--samples", samples, "--bev", bev, "--train", *train, "--holdout", *holdout, "--variant", args.variant, "--query", args.query, "--eval-query", "grae", "gt", "--seed", str(args.seed), "--epochs", str(args.epochs), "--out", ckpt_dir]
            if args.perturb:
                cmd.append("--perturb")
            if args.bev_pattern:
                cmd += ["--bev-pattern", args.bev_pattern]
            run(cmd, env, log)
        dump = "%s/holdout_grae_epoch%d.jsonl" % (ckpt_dir, args.epochs)
        cmd = [PY, "scripts/research_v2/infer_tcpn.py", "--model", ckpt_dir + "/last.pth", "--calib-dump", dump, "--detections", det, "--bev", bev, "--sequences", *seqs, "--output", pred_dir]
        if args.apply_box:
            cmd.append("--apply-box")
        run(cmd, env, log)
        rescored, tracks = pred_dir + "/rescored", pred_dir + "/predictions"
    elif args.variant == "raw":
        rescored, tracks = det, pred_dir + "/predictions"
        run([PY, "scripts/research_v2/replay_grae.py", "--detections", det, "--sequences", *seqs, "--output", pred_dir], env, log)
    else:
        rescored, tracks = pred_dir + "/rescored", pred_dir + "/predictions"
        cmd = [PY, "scripts/research_v2/baseline_rescore.py", "--mode", args.variant, "--detections", det, "--sequences", *seqs, "--output", rescored]
        if args.variant == "iso_cls":
            cmd += ["--samples", samples, "--holdout", *holdout]
        run(cmd, env, log)
        run([PY, "scripts/research_v2/replay_grae.py", "--detections", rescored, "--sequences", *seqs, "--output", pred_dir], env, log)
    run([PY, "scripts/research_v2/eval_rescored.py", "--rescored", rescored, "--sequences", *seqs, "--name", name, "--output", pred_dir + "/detection_metrics.json"], env, log)
    split = "train" if args.split == "holdout" else "val"
    run([PY, "scripts/experiments/eval_grae_variant.py", "--predictions", tracks, "--output", pred_dir + "/tracking", "--split", split, "--sequences", *seqs], env, log)
    run([PY, "scripts/experiments/eval_grae_variant.py", "--predictions", tracks, "--output", pred_dir + "/tracking_amota", "--split", split, "--sequences", *seqs, "--amota"], env, log)
    print("done", args.pipeline, args.split, name)


if __name__ == "__main__":
    main()
