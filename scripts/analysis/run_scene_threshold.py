import argparse
import os

# 进程池并行时每个worker只用单线程BLAS，避免线程过量抢占
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from inframot3d.analysis.scene_threshold import run  # noqa: E402
from inframot3d.config import load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/scene_threshold.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
