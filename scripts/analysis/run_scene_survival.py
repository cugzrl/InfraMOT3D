import argparse

from inframot3d.analysis.scene_survival import run
from inframot3d.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/analysis/scene_survival.yaml")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    run(
        load_config(args.config),
        predict=not args.skip_prediction,
        evaluate=not args.skip_evaluation,
    )


if __name__ == "__main__":
    main()

