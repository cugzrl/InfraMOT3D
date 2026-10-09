"""缓存冻结 CenterPoint 的 BEV 特征，PCA 降到 128 通道并裁剪到评估区域附近

PCA 只用 --fit-sequences 指定的训练侧帧拟合
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.perception.centerpoint_bev import batch_from_points, bev_feature, load_frozen_centerpoint
from inframot3d.tcpn.bev_cache import CROP, PCA_DIM

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"
POINTS = ROOT / "data/centerpoint_v2xseq/points"


def frames_of(sequence_id):
    manifest = read_json(CONVERTED / "manifest.json")
    entry = next(item for item in manifest["sequences"] if item["sequence_id"] == sequence_id)
    return [row["frame_id"] for row in read_jsonl(CONVERTED / entry["path"])]


def crop(feature):
    r0, r1, c0, c1 = CROP
    return feature[0, :, r0:r1, c0:c1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="third_party/OpenPCDet/tools/cfgs/v2x_seq_models/centerpoint.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-sequences", nargs="+", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--fit-frames", type=int, default=240)
    args = parser.parse_args()
    device = torch.device("cuda")
    model, dataset, load_gpu = load_frozen_centerpoint(ROOT / "third_party/OpenPCDet", ROOT / args.cfg, ROOT / args.ckpt, device)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    def feature_of(sequence_id, frame_id):
        points = np.load(POINTS / ("%s_%s.npy" % (sequence_id, frame_id)))
        return crop(bev_feature(model, batch_from_points(dataset, load_gpu, points, device)))

    pca_path = out / "pca.pt"
    if not pca_path.exists():
        rng = random.Random(0)
        pool = [(s, f) for s in args.fit_sequences for f in frames_of(s)]
        chosen = rng.sample(pool, min(args.fit_frames, len(pool)))
        total = None
        moment = None
        count = 0
        for sequence_id, frame_id in chosen:
            feat = feature_of(sequence_id, frame_id).float().reshape(512, -1)
            total = feat.sum(1) if total is None else total + feat.sum(1)
            moment = feat @ feat.T if moment is None else moment + feat @ feat.T
            count += feat.shape[1]
        mean = total / count
        cov = moment / count - mean[:, None] * mean[None, :]
        eigval, eigvec = torch.linalg.eigh(cov.double())
        order = torch.argsort(eigval, descending=True)
        eigval, eigvec = eigval[order], eigvec[:, order]
        comps = eigvec[:, :PCA_DIM].float().T.contiguous()
        retained = float(eigval[:PCA_DIM].sum() / eigval.sum())
        std = eigval[:PCA_DIM].clamp(min=1e-8).sqrt().float()
        torch.save({"mean": mean.cpu(), "components": comps.cpu(), "std": std.cpu(), "retained": retained, "fit_frames": len(chosen), "fit_sequences": args.fit_sequences, "ckpt": args.ckpt}, pca_path)
        print("pca retained", retained)
    pca = torch.load(pca_path)
    mean = pca["mean"].to(device)
    comps = pca["components"].to(device)
    std = pca["std"].to(device)
    r0, r1, c0, c1 = CROP
    for sequence_id in args.sequences:
        path = out / ("%s.npy" % sequence_id)
        if path.exists():
            continue
        frame_ids = frames_of(sequence_id)
        tmp = out / ("%s.tmp.npy" % sequence_id)
        store = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(frame_ids), PCA_DIM, r1 - r0, c1 - c0))
        for index, frame_id in enumerate(frame_ids):
            feat = feature_of(sequence_id, frame_id).float().reshape(512, -1)
            proj = (comps @ (feat - mean[:, None])) / std[:, None]
            store[index] = proj.reshape(PCA_DIM, r1 - r0, c1 - c0).half().cpu().numpy()
        store.flush()
        del store
        tmp.rename(path)
        (out / ("%s.json" % sequence_id)).write_text(json.dumps(frame_ids))
        print("cached", sequence_id, len(frame_ids))


if __name__ == "__main__":
    main()
