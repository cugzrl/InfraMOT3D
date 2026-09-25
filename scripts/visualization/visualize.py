import argparse

from inframot3d.config import load_config
from inframot3d.visualization.bev import render_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/ab3dmot/gt.yaml")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = config["project"]["output_root"]
    prediction_path = output_root / "predictions" / f"{args.sequence}.jsonl"
    visual_dir = output_root / "visualizations" / args.sequence
    images, video = render_sequence(
        prediction_path,
        visual_dir,
        config["visualization"],
        args.max_frames,
        args.video,
    )
    print(f"可视化完成 图片{len(images)} 视频{video or '未生成'}")


if __name__ == "__main__":
    main()
