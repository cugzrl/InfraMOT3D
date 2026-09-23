import argparse
import copy
import os

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from inframot3d.config import load_config
from inframot3d.visualization.open3d_sequence import render_open3d_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ab3dmot_gt.yaml")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--view", choices=["classic", "roadside"], default="classic")
    parser.add_argument("--mode", choices=["gt", "track", "both"], default="both")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--no-image-inset", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    output_root = config["project"]["output_root"]
    settings = copy.deepcopy(config["open3d_visualization"])
    if args.no_image_inset:
        settings["views"][args.view]["image_inset"]["enabled"] = False
    if args.output_dir:
        output_dir = config["_root"] / args.output_dir
    else:
        base_dir = output_root / "open3d_visualizations" / args.sequence
        output_dir = base_dir / args.mode if args.view == "classic" else base_dir / args.view / args.mode
    images, video, gif = render_open3d_sequence(
        config["project"]["data_root"],
        config["project"]["converted_root"],
        output_root / "predictions",
        output_dir,
        args.sequence,
        settings,
        view=args.view,
        mode=args.mode,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        make_video=args.video,
        make_gif=args.gif,
    )
    print(f"Open3D可视化完成 图片{len(images)} 视频{video or '未生成'} GIF{gif or '未生成'}")


if __name__ == "__main__":
    main()
