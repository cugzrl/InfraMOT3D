"""Run the fixed evaluation only after a full V6 training process exits."""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def alive(pid):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists() or stat.read_text().split()[2] == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    while alive(args.training_pid):
        time.sleep(30)
    checkpoint = ROOT / args.checkpoint
    if not checkpoint.exists():
        raise RuntimeError("Training exited without checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["epoch"] != args.expected_epochs:
        raise RuntimeError(f"Training stopped at epoch {saved['epoch']}")
    checkpoints = [("latest", checkpoint)]
    best = checkpoint.parent / "best.pth"
    if best.exists():
        best_saved = torch.load(best, map_location="cpu", weights_only=False)
        if best_saved["epoch"] != saved["epoch"]:
            checkpoints.append(("best", best))
    for name, path in checkpoints:
        subprocess.run([
            sys.executable, "scripts/research_v6/evaluate_long_term_query.py",
            "--checkpoint", path.relative_to(ROOT),
            "--output", Path(args.output) / name,
            "--device", args.device,
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
