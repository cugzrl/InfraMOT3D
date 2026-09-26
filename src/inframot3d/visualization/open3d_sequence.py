import colorsys
import os
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import cv2
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

from inframot3d.io import read_json, read_jsonl


BOX_EDGES = np.asarray(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

MACARON_PALETTE = np.asarray(
    [
        [0.98, 0.55, 0.62],
        [0.55, 0.80, 0.98],
        [0.58, 0.90, 0.68],
        [0.99, 0.78, 0.45],
        [0.75, 0.63, 0.95],
        [0.44, 0.86, 0.85],
        [0.98, 0.65, 0.88],
        [0.82, 0.88, 0.55],
        [0.96, 0.67, 0.50],
        [0.62, 0.75, 0.95],
    ],
    dtype=float,
)


def _font(size):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def _resolve_settings(settings, view):
    if view not in settings["views"]:
        raise ValueError(f"不支持的视角{view}")
    resolved = {key: value for key, value in settings.items() if key != "views"}
    resolved.update(settings["views"][view])
    return resolved


def _track_color(track_id, palette="hsv"):
    if palette == "macaron":
        return MACARON_PALETTE[int(track_id) % len(MACARON_PALETTE)].copy()
    hue = (int(track_id) * 0.618033988749895) % 1.0
    return np.asarray(colorsys.hsv_to_rgb(hue, 0.82, 1.0), dtype=float)


def _height_colors(points, z_range):
    values = np.clip((points[:, 2] - z_range[0]) / (z_range[1] - z_range[0]), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.column_stack([red, green, blue])


def _point_colors(points, settings):
    if settings.get("point_color_mode") == "gray":
        color = np.asarray(settings["point_color"], dtype=float)
        return np.repeat(color[None, :], len(points), axis=0)
    return _height_colors(points, settings["z_range"])


def _points_in_box(points, box):
    # 平移到 box 中心后按 -yaw 逆旋转，再判断是否落在长宽高范围内
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    shifted = points - np.asarray([x, y, z], dtype=float)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local_x = cosine * shifted[:, 0] + sine * shifted[:, 1]
    local_y = -sine * shifted[:, 0] + cosine * shifted[:, 1]
    local_z = shifted[:, 2]
    return (
        (np.abs(local_x) <= length / 2.0)
        & (np.abs(local_y) <= width / 2.0)
        & (np.abs(local_z) <= height / 2.0)
    )


def _color_tracked_points(points, colors, objects, palette):
    # 先出现的 track 占住重叠点，避免同一点被后续框反复改色
    claimed = np.zeros(len(points), dtype=bool)
    for value in objects:
        inside = _points_in_box(points, value["box"]) & ~claimed
        if not inside.any():
            continue
        colors[inside] = _track_color(value["track_id"], palette)
        claimed[inside] = True
    return colors


def _load_points(path, settings):
    source = o3d.io.read_point_cloud(str(path))
    points = np.asarray(source.points)
    if not len(points):
        raise ValueError(f"点云为空{path}")
    mask = np.isfinite(points).all(axis=1)
    mask &= (points[:, 0] >= settings["x_range"][0]) & (points[:, 0] <= settings["x_range"][1])
    mask &= (points[:, 1] >= settings["y_range"][0]) & (points[:, 1] <= settings["y_range"][1])
    mask &= (points[:, 2] >= settings["z_range"][0]) & (points[:, 2] <= settings["z_range"][1])
    points = points[mask]
    voxel_size = float(settings.get("voxel_size", 0.0))
    if voxel_size <= 0.0:
        return points
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return np.asarray(cloud.voxel_down_sample(voxel_size).points)


def _cloud_from_points(points, colors):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


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


def _project_corners(corners, intrinsic, extrinsic):
    rotation = np.asarray(extrinsic["rotation"], dtype=np.float64)
    translation = np.asarray(extrinsic["translation"], dtype=np.float64).reshape(1, 3)
    camera_points = np.asarray(corners, dtype=np.float64) @ rotation.T + translation
    matrix = np.asarray(intrinsic["cam_K"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(intrinsic.get("cam_D", []), dtype=np.float64)
    projected, _ = cv2.projectPoints(
        camera_points.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        matrix,
        distortion,
    )
    valid = camera_points[:, 2] > 0.1
    return projected.reshape(-1, 2), valid


def _rgb255(color):
    return tuple(int(value) for value in np.clip(np.asarray(color) * 255.0, 0.0, 255.0))


def _draw_image_boxes(image, objects, intrinsic, extrinsic, color_fn, show_ids, line_width):
    draw = ImageDraw.Draw(image, "RGBA")
    scale_x = image.width / float(intrinsic["width"])
    scale_y = image.height / float(intrinsic["height"])
    label_font = _font(12)
    for value in objects:
        projected, valid = _project_corners(_box_corners(value["box"]), intrinsic, extrinsic)
        projected[:, 0] *= scale_x
        projected[:, 1] *= scale_y
        color = _rgb255(color_fn(value))
        for first, second in BOX_EDGES:
            if valid[first] and valid[second]:
                draw.line(
                    [tuple(projected[first]), tuple(projected[second])],
                    fill=color + (245,),
                    width=line_width,
                )
        if show_ids and valid.any():
            visible = projected[valid]
            anchor_x = float(np.clip(visible[:, 0].min(), 0, image.width - 90))
            anchor_y = float(np.clip(visible[:, 1].min() - 17, 0, image.height - 18))
            label = f"{value['class_name']} {value['track_id']}"
            bounds = draw.textbbox((anchor_x, anchor_y), label, font=label_font)
            draw.rounded_rectangle(
                (bounds[0] - 3, bounds[1] - 2, bounds[2] + 3, bounds[3] + 2),
                radius=3,
                fill=color + (220,),
            )
            draw.text((anchor_x, anchor_y), label, font=label_font, fill=(25, 25, 30, 255))


def _add_classic_overlay(image, frame, gt_count, track_count, mode):
    image = Image.fromarray(np.asarray(image)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    text_font = _font(14)
    draw.rounded_rectangle((16, 16, 530, 83), radius=10, fill=(0, 0, 0, 175))
    title = f"InfraMOT3D | Seq {frame['sequence_id']} | Frame {frame['frame_id']}"
    draw.text((30, 25), title, font=text_font, fill=(255, 255, 255, 255))
    draw.text((30, 51), f"Mode {mode} | GT {gt_count} | Tracks {track_count}", font=text_font, fill=(215, 225, 240, 255))
    draw.line((555, 34, 605, 34), fill=(38, 255, 64, 255), width=4)
    draw.text((615, 25), "GT", font=text_font, fill=(255, 255, 255, 255))
    draw.line((680, 34, 730, 34), fill=(255, 140, 160, 255), width=4)
    draw.text((740, 25), "Track ID color", font=text_font, fill=(255, 255, 255, 255))
    return image


def _add_method_label(canvas, method_name):
    if not method_name:
        return
    draw = ImageDraw.Draw(canvas, "RGBA")
    text_font = _font(22)
    margin = 24
    bounds = draw.textbbox((0, 0), method_name, font=text_font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    left = canvas.width - margin - text_width
    top = margin
    draw.rounded_rectangle(
        (left - 12, top - 8, left + text_width + 12, top + text_height + 10),
        radius=8,
        fill=(255, 255, 255, 220),
    )
    draw.text((left, top), method_name, font=text_font, fill=(32, 36, 42, 255))


def _add_roadside_overlay(
    image,
    frame,
    gt_objects,
    track_objects,
    mode,
    data_root,
    metadata,
    settings,
    method_name="",
):
    canvas = Image.fromarray(np.asarray(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    text_font = _font(15)
    info = f"Seq {frame['sequence_id']}   Frame {frame['frame_id']}   GT {len(gt_objects)}   Tracks {len(track_objects)}"
    bounds = draw.textbbox((24, canvas.height - 42), info, font=text_font)
    draw.rounded_rectangle((16, canvas.height - 52, bounds[2] + 34, canvas.height - 14), radius=8, fill=(255, 255, 255, 220))
    draw.text((24, canvas.height - 42), info, font=text_font, fill=(35, 42, 52, 255))
    _add_method_label(canvas, method_name)
    inset_settings = settings["image_inset"]
    if not inset_settings.get("enabled", False):
        return canvas
    source = Image.open(data_root / frame["image_path"]).convert("RGB")
    inset_width = int(inset_settings["width"])
    inset_height = int(round(inset_width * source.height / source.width))
    inset = source.resize((inset_width, inset_height), Image.Resampling.LANCZOS)
    if inset_settings.get("draw_boxes", True):
        intrinsic = read_json(data_root / metadata["calib_camera_intrinsic_path"])
        extrinsic = read_json(data_root / metadata["calib_virtuallidar_to_camera_path"])
        if mode in {"gt", "both"}:
            _draw_image_boxes(
                inset,
                gt_objects,
                intrinsic,
                extrinsic,
                lambda _: settings["gt_color"],
                False,
                2,
            )
        if mode in {"track", "both"}:
            _draw_image_boxes(
                inset,
                track_objects,
                intrinsic,
                extrinsic,
                lambda value: _track_color(value["track_id"], "macaron"),
                True,
                3,
            )
    margin = int(inset_settings["margin"])
    canvas.paste(inset, (margin, margin))
    draw = ImageDraw.Draw(canvas, "RGBA")
    border = int(inset_settings["border_width"])
    draw.rectangle(
        (margin - border, margin - border, margin + inset_width + border, margin + inset_height + border),
        outline=(255, 255, 255, 245),
        width=border,
    )
    return canvas


def _write_video(image_paths, path, fps):
    first = cv2.imread(str(image_paths[0]))
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), int(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"视频写入器创建失败{path}")
    for image_path in image_paths:
        writer.write(cv2.imread(str(image_path)))
    writer.release()


def _write_gif(image_paths, path, fps):
    frames = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000.0 / float(fps))),
        loop=0,
        optimize=False,
    )
    for frame in frames:
        frame.close()


def render_open3d_sequence(
    data_root,
    converted_root,
    prediction_root,
    output_dir,
    sequence_id,
    settings,
    view="classic",
    mode="both",
    start_frame=0,
    max_frames=None,
    make_video=False,
    make_gif=False,
    method_name="",
):
    if mode not in {"gt", "track", "both"}:
        raise ValueError(f"不支持的显示模式{mode}")
    settings = _resolve_settings(settings, view)
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
    track_frames = list(read_jsonl(prediction_root / f"{sequence_id}.jsonl"))
    metadata_rows = read_json(data_root / "data_info.json")
    metadata_map = {
        (str(value["sequence_id"]), str(value["frame_id"])): value for value in metadata_rows
    }
    if len(gt_frames) != len(track_frames):
        raise ValueError(f"序列{sequence_id}GT与跟踪帧数不一致")
    stop_frame = len(gt_frames) if max_frames is None else min(len(gt_frames), start_frame + max_frames)
    width, height = int(settings["width"]), int(settings["height"])
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background(np.asarray(settings["background"], dtype=np.float32))
    if view == "roadside":
        # 关闭色调映射，否则纯白背景会被压成浅灰
        renderer.scene.view.set_post_processing(False)
    cloud_material = _material("defaultUnlit", point_size=settings["point_size"])
    gt_material = _material("unlitLine", line_width=settings["gt_line_width"])
    track_material = _material("unlitLine", line_width=settings["track_line_width"])
    palette = "macaron" if view == "roadside" else "hsv"
    image_paths = []
    for index in range(start_frame, stop_frame):
        gt_frame = gt_frames[index]
        track_frame = track_frames[index]
        if gt_frame["frame_id"] != track_frame["frame_id"]:
            raise ValueError(f"序列{sequence_id}第{index}帧编号不一致")
        renderer.scene.clear_geometry()
        points = _load_points(data_root / gt_frame["pointcloud_path"], settings)
        colors = _point_colors(points, settings)
        if view == "roadside" and mode in {"track", "both"} and track_frame["objects"]:
            colors = _color_tracked_points(points, colors, track_frame["objects"], palette)
        cloud = _cloud_from_points(points, colors)
        renderer.scene.add_geometry("pointcloud", cloud, cloud_material)
        if mode in {"gt", "both"}:
            gt_lines = _box_lines(gt_frame["objects"], lambda _: settings["gt_color"])
            if gt_lines.has_lines():
                renderer.scene.add_geometry("gt_boxes", gt_lines, gt_material)
        if mode in {"track", "both"}:
            track_lines = _box_lines(
                track_frame["objects"],
                lambda value: _track_color(value["track_id"], palette),
            )
            if track_lines.has_lines():
                renderer.scene.add_geometry("track_boxes", track_lines, track_material)
        renderer.setup_camera(
            float(settings["field_of_view"]),
            np.asarray(settings["camera_center"], dtype=float),
            np.asarray(settings["camera_eye"], dtype=float),
            np.asarray(settings["camera_up"], dtype=float),
        )
        rendered = renderer.render_to_image()
        if view == "roadside":
            metadata = metadata_map[(sequence_id, str(gt_frame["frame_id"]))]
            image = _add_roadside_overlay(
                rendered,
                gt_frame,
                gt_frame["objects"],
                track_frame["objects"],
                mode,
                data_root,
                metadata,
                settings,
                method_name,
            )
        else:
            image = _add_classic_overlay(
                rendered,
                gt_frame,
                len(gt_frame["objects"]),
                len(track_frame["objects"]),
                mode,
            )
        image_path = frame_dir / f"{index:06d}.png"
        image.save(image_path)
        image_paths.append(image_path)
        if len(image_paths) == 1 or len(image_paths) % 10 == 0:
            print(f"已渲染{len(image_paths)}帧")
    del renderer
    video_path = output_dir / f"{sequence_id}_{view}_{mode}.mp4" if make_video and image_paths else None
    gif_path = output_dir / f"{sequence_id}_{view}_{mode}.gif" if make_gif and image_paths else None
    if video_path is not None:
        _write_video(image_paths, video_path, settings["fps"])
    if gif_path is not None:
        _write_gif(image_paths, gif_path, settings["fps"])
    return image_paths, video_path, gif_path
