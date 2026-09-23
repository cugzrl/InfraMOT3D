import colorsys
import os
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from inframot3d.io import read_json, read_jsonl


BOX_EDGES = np.asarray(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)


def _track_color(track_id):
    hue = (int(track_id) * 0.618033988749895) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.82, 1.0), dtype=float)


def _height_colors(points, z_range):
    values = np.clip((points[:, 2] - z_range[0]) / (z_range[1] - z_range[0]), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.column_stack([red, green, blue])


def _load_point_cloud(path, settings):
    source = o3d.io.read_point_cloud(str(path))
    points = np.asarray(source.points)
    if not len(points):
        raise ValueError(f"点云为空{path}")
    mask = np.isfinite(points).all(axis=1)
    mask &= (points[:, 0] >= settings["x_range"][0]) & (points[:, 0] <= settings["x_range"][1])
    mask &= (points[:, 1] >= settings["y_range"][0]) & (points[:, 1] <= settings["y_range"][1])
    mask &= (points[:, 2] >= settings["z_range"][0]) & (points[:, 2] <= settings["z_range"][1])
    points = points[mask]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(_height_colors(points, settings["z_range"]))
    voxel_size = float(settings.get("voxel_size", 0.0))
    return cloud.voxel_down_sample(voxel_size) if voxel_size > 0.0 else cloud


def _box_corners(box):
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    local = np.asarray(
        [
            [length / 2.0, width / 2.0, -height / 2.0],
            [-length / 2.0, width / 2.0, -height / 2.0],
            [-length / 2.0, -width / 2.0, -height / 2.0],
            [length / 2.0, -width / 2.0, -height / 2.0],
            [length / 2.0, width / 2.0, height / 2.0],
            [-length / 2.0, width / 2.0, height / 2.0],
            [-length / 2.0, -width / 2.0, height / 2.0],
            [length / 2.0, -width / 2.0, height / 2.0],
        ],
        dtype=float,
    )
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    return local @ rotation.T + np.asarray([x, y, z])


def _box_lines(objects, color_fn):
    points = []
    lines = []
    colors = []
    for value in objects:
        offset = len(points)
        points.extend(_box_corners(value["box"]).tolist())
        color = np.asarray(color_fn(value), dtype=float)
        if int(value.get("time_since_update", 0)) > 0:
            color *= 0.45
        lines.extend((BOX_EDGES + offset).tolist())
        colors.extend([color.tolist()] * len(BOX_EDGES))
    line_set = o3d.geometry.LineSet()
    if points:
        line_set.points = o3d.utility.Vector3dVector(np.asarray(points))
        line_set.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return line_set


def _material(shader, point_size=None, line_width=None):
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = shader
    if point_size is not None:
        material.point_size = float(point_size)
    if line_width is not None:
        material.line_width = float(line_width)
    return material


def _add_overlay(image, frame, gt_count, track_count, mode):
    image = Image.fromarray(np.asarray(image)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((16, 16, 530, 83), radius=10, fill=(0, 0, 0, 175))
    title = f"InfraMOT3D | Seq {frame['sequence_id']} | Frame {frame['frame_id']}"
    draw.text((30, 27), title, fill=(255, 255, 255, 255))
    draw.text((30, 52), f"Mode {mode} | GT {gt_count} | Tracks {track_count}", fill=(215, 225, 240, 255))
    draw.line((555, 34, 605, 34), fill=(38, 255, 64, 255), width=4)
    draw.text((615, 25), "GT", fill=(255, 255, 255, 255))
    draw.line((680, 34, 730, 34), fill=(255, 90, 80, 255), width=4)
    draw.text((740, 25), "Track ID color", fill=(255, 255, 255, 255))
    return image


def _write_video(image_paths, path, fps):
    first = cv2.imread(str(image_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"视频写入器创建失败{path}")
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        writer.write(frame)
    writer.release()


def _write_gif(image_paths, path, fps):
    with imageio.get_writer(path, mode="I", duration=1000.0 / float(fps), loop=0) as writer:
        for image_path in image_paths:
            writer.append_data(imageio.imread(image_path))


def render_open3d_sequence(
    data_root,
    converted_root,
    prediction_root,
    output_dir,
    sequence_id,
    settings,
    mode="both",
    start_frame=0,
    max_frames=None,
    make_video=False,
    make_gif=False,
):
    if mode not in {"gt", "track", "both"}:
        raise ValueError(f"不支持的显示模式{mode}")
    data_root = Path(data_root)
    converted_root = Path(converted_root)
    prediction_root = Path(prediction_root)
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(converted_root / "manifest.json")
    entries = {value["sequence_id"]: value for value in manifest["sequences"]}
    if sequence_id not in entries:
        raise KeyError(f"不存在序列{sequence_id}")
    gt_frames = list(read_jsonl(converted_root / entries[sequence_id]["path"]))
    prediction_path = prediction_root / f"{sequence_id}.jsonl"
    track_frames = list(read_jsonl(prediction_path))
    if len(gt_frames) != len(track_frames):
        raise ValueError(f"序列{sequence_id}GT与跟踪帧数不一致")
    stop_frame = len(gt_frames) if max_frames is None else min(len(gt_frames), start_frame + max_frames)
    width, height = int(settings["width"]), int(settings["height"])
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.asarray(settings["background"], dtype=np.float32))
    cloud_material = _material("defaultUnlit", point_size=settings["point_size"])
    gt_material = _material("unlitLine", line_width=settings["gt_line_width"])
    track_material = _material("unlitLine", line_width=settings["track_line_width"])
    image_paths = []
    for index in range(start_frame, stop_frame):
        gt_frame = gt_frames[index]
        track_frame = track_frames[index]
        if gt_frame["frame_id"] != track_frame["frame_id"]:
            raise ValueError(f"序列{sequence_id}第{index}帧编号不一致")
        renderer.scene.clear_geometry()
        cloud_path = data_root / gt_frame["pointcloud_path"]
        cloud = _load_point_cloud(cloud_path, settings)
        renderer.scene.add_geometry("pointcloud", cloud, cloud_material)
        if mode in {"gt", "both"}:
            gt_lines = _box_lines(gt_frame["objects"], lambda _: settings["gt_color"])
            if gt_lines.has_lines():
                renderer.scene.add_geometry("gt_boxes", gt_lines, gt_material)
        if mode in {"track", "both"}:
            track_lines = _box_lines(track_frame["objects"], lambda value: _track_color(value["track_id"]))
            if track_lines.has_lines():
                renderer.scene.add_geometry("track_boxes", track_lines, track_material)
        renderer.setup_camera(
            float(settings["field_of_view"]),
            np.asarray(settings["camera_center"], dtype=float),
            np.asarray(settings["camera_eye"], dtype=float),
            np.asarray(settings["camera_up"], dtype=float),
        )
        image = renderer.render_to_image()
        rendered = _add_overlay(
            image,
            gt_frame,
            len(gt_frame["objects"]),
            len(track_frame["objects"]),
            mode,
        )
        image_path = frame_dir / f"{index:06d}.png"
        rendered.save(image_path)
        image_paths.append(image_path)
        if len(image_paths) == 1 or len(image_paths) % 10 == 0:
            print(f"已渲染{len(image_paths)}帧")
    del renderer
    video_path = output_dir / f"{sequence_id}_{mode}.mp4" if make_video and image_paths else None
    gif_path = output_dir / f"{sequence_id}_{mode}.gif" if make_gif and image_paths else None
    if video_path is not None:
        _write_video(image_paths, video_path, settings["fps"])
    if gif_path is not None:
        _write_gif(image_paths, gif_path, settings["fps"])
    return image_paths, video_path, gif_path
