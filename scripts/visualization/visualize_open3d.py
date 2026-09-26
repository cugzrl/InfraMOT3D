import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import yaml

from inframot3d.config import load_config
from inframot3d.visualization.open3d_sequence import render_open3d_sequence


TRACKERS = {
    "ab3dmot": ("configs/trackers/ab3dmot/centerpoint.yaml", "AB3DMOT"),
    "simpletrack": ("configs/trackers/simpletrack/centerpoint.yaml", "SimpleTrack"),
    "immortal": ("configs/trackers/immortal/centerpoint.yaml", "ImmortalTracker"),
    "grae": ("configs/trackers/grae/centerpoint.yaml", "GRAE-3DMOT"),
    "fastpoly": ("configs/trackers/fastpoly/centerpoint.yaml", "Fast-Poly"),
    "3dmotformer": ("configs/trackers/3dmotformer/centerpoint.yaml", "3DMOTFormer"),
}
VISUALIZATION_CONFIG = "configs/visualization/open3d_v2xseq.yaml"
DEFAULT_CONFIG = "configs/trackers/ab3dmot/gt.yaml"


def _frame_limit(num_frames, max_frames):
    if num_frames is not None and max_frames is not None and num_frames != max_frames:
        raise SystemExit("--num-frames 与 --max-frames 不能同时给出不同数值")
    if num_frames is not None:
        return num_frames
    return max_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=sorted(TRACKERS))
    parser.add_argument("--config", default=None)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--view", choices=["classic", "roadside"], default="classic")
    parser.add_argument("--mode", choices=["gt", "track", "both"], default="both")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--no-image-inset", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.tracker and args.config:
        raise SystemExit("请只使用 --tracker 或 --config 之一")
    if args.tracker:
        config_path, method_name = TRACKERS[args.tracker]
        slug = args.tracker
    else:
        config_path = args.config or DEFAULT_CONFIG
        method_name = ""
        slug = Path(config_path).parent.name
    config = load_config(config_path)
    if not method_name:
        method_name = str(config.get("tracker", {}).get("name", ""))
    root = config["_root"]
    settings = yaml.safe_load((root / VISUALIZATION_CONFIG).read_text(encoding="utf-8"))
    settings = copy.deepcopy(settings)
    if args.no_image_inset:
        settings["views"][args.view]["image_inset"]["enabled"] = False
    output_root = config["project"]["output_root"]
    prediction_root = output_root / "predictions"
    prediction_file = prediction_root / f"{args.sequence}.jsonl"
    if not prediction_file.is_file():
        raise SystemExit(f"缺少预测文件 {prediction_file}")
    if args.output_dir:
        output_dir = root / args.output_dir
    else:
        base_dir = output_root / "open3d_visualizations" / slug / args.sequence
        output_dir = base_dir / args.mode if args.view == "classic" else base_dir / args.view / args.mode
    images, video, gif = render_open3d_sequence(
        config["project"]["data_root"],
        config["project"]["converted_root"],
        prediction_root,
        output_dir,
        args.sequence,
        settings,
        view=args.view,
        mode=args.mode,
        start_frame=args.start_frame,
        max_frames=_frame_limit(args.num_frames, args.max_frames),
        make_video=args.video,
        make_gif=args.gif,
        method_name=method_name,
    )
    print(f"Open3D可视化完成 方法{method_name or '未命名'} 预测{prediction_root} 图片{len(images)} 视频{video or '未生成'} GIF{gif or '未生成'}")


if __name__ == "__main__":
    main()
