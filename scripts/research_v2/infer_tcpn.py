"""闭环推理：GRAE 当前在线轨迹 + 当前候选 + 当前 BEV → 候选质量 → 映射到原始分数尺度 → 原始 GRAE

只读取检测、BEV 缓存与在线轨迹，不读取 GT
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "research_v2"))

from inframot3d.config import load_config
from inframot3d.io import read_jsonl, write_json, write_jsonl
from inframot3d.tcpn.bev_cache import BevStore
from inframot3d.tcpn.calibration import ScoreMapper, fit_from_dump
from inframot3d.tcpn.data import CLASS_INDEX, _inside, pack_tracks_online
from inframot3d.tcpn.dataset import collate, frame_sites, make_item, to_device
from inframot3d.tcpn.sampling import LaneSampler
from inframot3d.tcpn.model import TCPN, apply_box
from replay_grae import build_tracker


def current_tracks(tracker, history):
    track = tracker.track
    if track is None or len(track) == 0:
        return []
    out = []
    trans = track.translation.detach().cpu().numpy()
    yaw = track.yaw.detach().cpu().numpy().reshape(-1)
    size = track.size.detach().cpu().numpy()
    score = track.score.detach().cpu().numpy().reshape(-1)
    cls = track.classes.detach().cpu().numpy().reshape(-1)
    ids = track.instance_inds.detach().cpu().numpy().reshape(-1)
    for k in range(len(track)):
        if int(cls[k]) > 3:
            continue
        tid = int(ids[k])
        box = [float(trans[k, 0]), float(trans[k, 1]), float(trans[k, 2]), float(yaw[k]), float(size[k, 0]), float(size[k, 1]), float(size[k, 2])]
        out.append({"track_id": tid, "class_index": int(cls[k]), "box": box, "score": float(score[k]), "history": history[tid][-5:], "hits": len(history[tid])})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/trackers/grae/centerpoint.yaml")
    parser.add_argument("--grae-ckpt", default="outputs/grae_centerpoint/ckpt/checkpoint-best.pth")
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib-dump", required=True)
    parser.add_argument("--detections", required=True)
    parser.add_argument("--bev", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="map", choices=["map", "raw"])
    parser.add_argument("--apply-box", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda")
    ckpt = torch.load(ROOT / args.model, map_location=device, weights_only=False)
    model = TCPN(hidden=ckpt["hidden"], **ckpt["spec"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    mapper = fit_from_dump(ROOT / args.calib_dump)
    config = load_config(args.config)
    tracker = build_tracker(config, ROOT / args.grae_ckpt)
    store = BevStore(ROOT / args.bev) if ckpt["spec"].get("use_bev") else None
    pattern = ckpt["args"].get("bev_pattern") if store is not None else None
    sampler = LaneSampler(ROOT) if pattern in ("lane", "laneline") else None
    out = ROOT / args.output
    start = time.perf_counter()
    frames = 0
    net_time = 0.0
    for sequence_id in args.sequences:
        tracker.reset()
        history = defaultdict(list)
        rows, rescored = [], []
        for det_row in read_jsonl(ROOT / args.detections / ("%s.jsonl" % sequence_id)):
            t_now = int(det_row["timestamp"])
            objects = [dict(item) for item in det_row["objects"]]
            vidx = [i for i, o in enumerate(objects) if o["class_name"] in CLASS_INDEX]
            boxes = np.array([objects[i]["box"] for i in vidx], np.float32).reshape(-1, 7)
            inside = _inside(boxes[:, :2]) if len(boxes) else np.zeros(0, bool)
            local = [vidx[k] for k in np.where(inside)[0]]
            if local:
                cbox = np.array([objects[i]["box"] for i in local], np.float32)
                cscore = np.array([objects[i]["score"] for i in local], np.float32)
                ccls = np.array([CLASS_INDEX[objects[i]["class_name"]] for i in local], np.int64)
                trk, _ = pack_tracks_online(current_tracks(tracker, history), t_now)
                item = make_item(cbox, cscore, ccls, trk)
                if store is not None:
                    item["bev"] = np.asarray(store.frame(sequence_id, det_row["frame_id"]))
                    if pattern:
                        item["cand_sites"] = frame_sites(pattern, cbox, sampler, sequence_id, det_row["frame_id"])
                batch = to_device(collate([item]), device)
                tic = time.perf_counter()
                with torch.no_grad():
                    pred = model(batch)
                    prob = torch.sigmoid(pred["quality"][0, : len(local)]).cpu().numpy()
                    refined = apply_box(batch["cand_box"], pred["box"])[0, : len(local)].cpu().numpy() if args.apply_box else None
                torch.cuda.synchronize()
                net_time += time.perf_counter() - tic
                new = mapper.to_raw_scale(prob) if args.mode == "map" else cscore
                for k, i in enumerate(local):
                    objects[i]["raw_score"] = float(cscore[k])
                    objects[i]["quality"] = float(prob[k])
                    objects[i]["score"] = float(new[k])
                    if refined is not None:
                        objects[i]["raw_box"] = objects[i]["box"]
                        objects[i]["box"] = [float(v) for v in refined[k]]
            tracked = tracker.update(objects, t_now / 1e6, det_row["frame_id"])
            group = ((tracker.last_debug or {}).get("groups") or [{}])[0]
            pairs = [(int(a["input_index"]), int(a["track_id"])) for a in group.get("assignments") or []]
            pairs += [(int(a["input_index"]), int(a["track_id"])) for a in group.get("created") or []]
            for inp, tid in pairs:
                obj = objects[inp]
                if obj["class_name"] in CLASS_INDEX:
                    history[tid].append({"box": obj.get("raw_box", obj["box"]), "score": float(obj.get("raw_score", obj["score"])), "t": t_now})
            rows.append({"sequence_id": det_row["sequence_id"], "frame_index": det_row["frame_index"], "frame_id": det_row["frame_id"], "timestamp": det_row["timestamp"], "objects": tracked})
            rescored.append({"sequence_id": det_row["sequence_id"], "frame_index": det_row["frame_index"], "frame_id": det_row["frame_id"], "timestamp": det_row["timestamp"], "objects": objects})
            frames += 1
        write_jsonl(out / "predictions" / ("%s.jsonl" % sequence_id), rows)
        write_jsonl(out / "rescored" / ("%s.jsonl" % sequence_id), rescored)
    write_json(out / "runtime.json", {"frames": frames, "seconds": time.perf_counter() - start, "network_seconds": net_time, "network_ms_per_frame": 1000 * net_time / max(frames, 1), "model": args.model, "params": int(sum(p.numel() for p in model.parameters())), "max_memory_mb": torch.cuda.max_memory_allocated() / 2**20})
    print("frames", frames, "net ms/frame", round(1000 * net_time / max(frames, 1), 2))


if __name__ == "__main__":
    main()
