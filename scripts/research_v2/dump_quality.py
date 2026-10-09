"""在指定序列上导出开环质量分数，用于和评估集分开的校准"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.tcpn.dataset import FrameDataset, collate, to_device
from inframot3d.tcpn.model import TCPN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--bev", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    ckpt = torch.load(ROOT / args.model, map_location=device, weights_only=False)
    model = TCPN(hidden=ckpt["hidden"], **ckpt["spec"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    dataset = FrameDataset(ROOT / args.samples, ROOT / args.bev, args.sequences, query="grae", need_bev=bool(ckpt["spec"].get("use_bev")))
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, collate_fn=collate)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream, torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            prob = torch.sigmoid(model(batch)["quality"]).cpu().numpy()
            for b, index in enumerate(batch["index"].tolist()):
                sample = dataset.samples[index]
                n = len(sample["cand_box"])
                row = {"sequence_id": sample["sequence_id"], "frame_id": sample["frame_id"], "prob": prob[b, :n].tolist(), "raw": sample["cand_score"].tolist(), "quality": sample["cand_quality"].tolist()}
                stream.write(json.dumps(row) + "\n")
    print("frames", len(dataset), "wrote", out)


if __name__ == "__main__":
    main()
