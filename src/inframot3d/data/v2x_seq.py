from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inframot3d.io import read_json, write_json, write_jsonl


def _convert_object(value):
    dims = value["3d_dimensions"]
    loc = value["3d_location"]
    return {
        "class_name": value["type"],
        "source_track_id": str(value["track_id"]),
        "score": 1.0,
        "box": [
            float(loc["x"]),
            float(loc["y"]),
            float(loc["z"]),
            float(value["rotation"]),
            float(dims["l"]),
            float(dims["w"]),
            float(dims["h"]),
        ],
    }


def convert_dataset(source_root, output_root):
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    frames = read_json(source_root / "data_info.json")
    grouped = defaultdict(list)
    for frame in frames:
        grouped[str(frame["sequence_id"])].append(frame)

    class_counts = Counter()
    track_ids = defaultdict(set)
    sequence_entries = []
    for sequence_id in sorted(grouped):
        source_frames = sorted(
            grouped[sequence_id],
            key=lambda item: (int(item["pointcloud_timestamp"]), int(item["frame_id"])),
        )
        rows = []
        for frame_index, frame in enumerate(source_frames):
            labels = read_json(source_root / frame["label_lidar_std_path"])
            objects = [_convert_object(value) for value in labels]
            for value in objects:
                class_counts[value["class_name"]] += 1
                track_ids[value["class_name"]].add(value["source_track_id"])
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "frame_id": str(frame["frame_id"]),
                    "timestamp": int(frame["pointcloud_timestamp"]),
                    "image_path": frame["image_path"],
                    "pointcloud_path": frame["pointcloud_path"],
                    "objects": objects,
                }
            )
        relative_path = Path("sequences") / f"{sequence_id}.jsonl"
        write_jsonl(output_root / relative_path, rows)
        sequence_entries.append(
            {
                "sequence_id": sequence_id,
                "num_frames": len(rows),
                "path": relative_path.as_posix(),
            }
        )

    manifest = {
        "dataset": "V2X-Seq-SPD",
        "side": "infrastructure",
        "coordinate_system": "virtual_lidar",
        "box_order": ["x", "y", "z", "yaw", "length", "width", "height"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "num_sequences": len(sequence_entries),
        "num_frames": len(frames),
        "num_objects": sum(class_counts.values()),
        "class_counts": dict(sorted(class_counts.items())),
        "unique_tracks_by_class": {key: len(value) for key, value in sorted(track_ids.items())},
        "sequences": sequence_entries,
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest
