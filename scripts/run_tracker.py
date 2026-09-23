import argparse
import time
from pathlib import Path

from inframot3d.config import load_config
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl
from inframot3d.tracking import MultiClassAB3DMOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ab3dmot_gt.yaml")
    parser.add_argument("--sequences", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config)
    converted_root = Path(config["project"]["converted_root"])
    output_root = Path(config["project"]["output_root"])
    prediction_root = output_root / "predictions"
    manifest = read_json(converted_root / "manifest.json")
    allowed = set(args.sequences) if args.sequences else None
    total_frames = 0
    total_objects = 0
    start = time.perf_counter()
    for entry in manifest["sequences"]:
        sequence_id = entry["sequence_id"]
        if allowed is not None and sequence_id not in allowed:
            continue
        tracker = MultiClassAB3DMOT(config["tracker"])
        rows = []
        for frame in read_jsonl(converted_root / entry["path"]):
            objects = tracker.update(frame["objects"])
            rows.append(
                {
                    "sequence_id": frame["sequence_id"],
                    "frame_index": frame["frame_index"],
                    "frame_id": frame["frame_id"],
                    "timestamp": frame["timestamp"],
                    "objects": objects,
                }
            )
            total_frames += 1
            total_objects += len(objects)
        write_jsonl(prediction_root / f"{sequence_id}.jsonl", rows)
        print(f"完成序列{sequence_id} 帧{len(rows)}")
    elapsed = time.perf_counter() - start
    runtime = {
        "tracker": config["tracker"]["name"],
        "input": "ground_truth",
        "num_frames": total_frames,
        "num_output_objects": total_objects,
        "elapsed_seconds": elapsed,
        "fps": total_frames / elapsed if elapsed else 0.0,
    }
    write_json(output_root / "runtime.json", runtime)
    print(f"跟踪完成 帧{total_frames} 用时{elapsed:.2f}秒 FPS{runtime['fps']:.2f}")


if __name__ == "__main__":
    main()
