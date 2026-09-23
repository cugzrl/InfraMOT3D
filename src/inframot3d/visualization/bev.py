from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

from inframot3d.geometry import bev_corners
from inframot3d.io import read_jsonl


def _color(track_id):
    palette = plt.get_cmap("tab20")
    return palette(int(track_id) % 20)


def render_sequence(prediction_path, output_dir, settings, max_frames=None, make_video=False):
    prediction_path = Path(prediction_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []
    for frame_number, frame in enumerate(read_jsonl(prediction_path)):
        if max_frames is not None and frame_number >= max_frames:
            break
        figure, axis = plt.subplots(figsize=(8, 8))
        for value in frame["objects"]:
            corners = bev_corners(value["box"])
            color = _color(value["track_id"])
            axis.add_patch(PolygonPatch(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5))
            axis.text(corners[0, 0], corners[0, 1], str(value["track_id"]), color=color, fontsize=6)
        axis.set_xlim(settings["x_range"])
        axis.set_ylim(settings["y_range"])
        axis.set_aspect("equal")
        axis.grid(True, linewidth=0.3)
        axis.set_xlabel("x/m")
        axis.set_ylabel("y/m")
        axis.set_title(f"{frame['sequence_id']}  {frame['frame_id']}  AB3DMOT")
        image_path = output_dir / f"{frame_number:06d}.png"
        figure.savefig(image_path, dpi=int(settings["dpi"]), bbox_inches="tight")
        plt.close(figure)
        image_paths.append(image_path)
    video_path = None
    if make_video and image_paths:
        first = cv2.imread(str(image_paths[0]))
        height, width = first.shape[:2]
        video_path = output_dir / f"{prediction_path.stem}.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), int(settings["fps"]), (width, height)
        )
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
        writer.release()
    return image_paths, video_path
