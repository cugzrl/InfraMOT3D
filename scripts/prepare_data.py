import argparse

from inframot3d.config import load_config
from inframot3d.data.v2x_seq import convert_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ab3dmot_gt.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = convert_dataset(
        config["project"]["data_root"],
        config["project"]["converted_root"],
    )
    print(
        f"转换完成 序列{manifest['num_sequences']} 帧{manifest['num_frames']} "
        f"目标{manifest['num_objects']}"
    )


if __name__ == "__main__":
    main()
