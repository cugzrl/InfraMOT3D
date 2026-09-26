import argparse

from inframot3d.analysis.tracking_bottleneck import run
from inframot3d.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/tracking_bottleneck.yaml")
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
