"""Paired causal detection and original-GRAE evaluation for V6."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEQUENCES = ("0008", "0035", "0057", "0072", "0084")
THRESHOLD = "0.47854848529411764"


def run(*args):
    command = [sys.executable, *map(str, args)]
    print("running", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output)
    for name, extra in (("on", ()), ("off", ("--disable-feedback",))):
        detection = output / f"detections_{name}"
        run("scripts/research_v6/infer_long_term_query.py",
            "--checkpoint", args.checkpoint, "--sequences", *SEQUENCES,
            "--output", detection, "--device", args.device, *extra)
    run("scripts/research_v5/eval_detector_variants.py",
        "--run", "D0", "outputs/research/fixed_view_memory_p2/pA_D0",
        "--run", "D2", "outputs/research/fixed_view_memory_p2/pA_D2_motion_w020",
        "--run", "V5_on", "outputs/research/attention_query_v2/pA/detections_e08",
        "--run", "V6_off", output / "detections_off",
        "--run", "V6_on", output / "detections_on",
        "--sequences", *SEQUENCES, "--iou", "0.25", "0.5",
        "--score-threshold", THRESHOLD, "--output", output / "detection_metrics.json")
    run("scripts/research_v5/analyze_missed_targets.py",
        "--baseline", "outputs/research/fixed_view_memory_p2/pA_D0",
        "--candidate", output / "detections_on",
        "--candidate-disabled", output / "detections_off",
        "--sequences", *SEQUENCES,
        "--output", output / "weak_target_recovery.json")
    run("scripts/research_v6/analyze_feedback_transitions.py",
        "--baseline", "outputs/research/fixed_view_memory_p2/pA_D0",
        "--feedback-on", output / "detections_on",
        "--feedback-off", output / "detections_off",
        "--sequences", *SEQUENCES,
        "--output", output / "feedback_transitions.json")
    for name in ("on", "off"):
        replay = output / f"grae_{name}"
        run("scripts/research_v2/replay_grae.py",
            "--detections", output / f"detections_{name}",
            "--sequences", *SEQUENCES, "--output", replay)
        run("scripts/experiments/eval_grae_variant.py",
            "--predictions", replay / "predictions", "--output", replay / "evaluation",
            "--sequences", *SEQUENCES, "--score-threshold", THRESHOLD)


if __name__ == "__main__":
    main()
