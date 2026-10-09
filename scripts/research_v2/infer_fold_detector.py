"""用交叉拟合 CenterPoint 导出检测，分数下限 0.01，与原始检测导出一致"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.detection.openpcdet_adapter import load_openpcdet_cfg, prediction_to_object
from inframot3d.io import read_json, read_jsonl, write_json, write_jsonl

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"


class _Logger:
    def info(self, message):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--infos", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.01)
    args = parser.parse_args()
    cfg = load_openpcdet_cfg(ROOT / args.cfg)
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.SCORE_THRESH = args.score_threshold
    cfg.MODEL.POST_PROCESSING.SCORE_THRESH = args.score_threshold
    cfg.DATA_CONFIG.INFO_PATH["test"] = list(args.infos)
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu

    dataset, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, batch_size=4, dist=False, workers=4, logger=_Logger(), training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(ROOT / args.ckpt), logger=_Logger())
    model.cuda().eval()
    predictions = {}
    with torch.no_grad():
        for batch in loader:
            load_data_to_gpu(batch)
            pred_dicts, _ = model(batch)
            for anno in dataset.generate_prediction_dicts(batch, pred_dicts, cfg.CLASS_NAMES):
                predictions[str(anno["frame_id"])] = [prediction_to_object(n, s, b) for n, s, b in zip(anno["name"], anno["score"], anno["boxes_lidar"])]
    manifest = read_json(CONVERTED / "manifest.json")
    out = ROOT / args.output
    written = []
    for entry in manifest["sequences"]:
        rows = []
        for frame in read_jsonl(CONVERTED / entry["path"]):
            key = "%s_%s" % (frame["sequence_id"], frame["frame_id"])
            if key not in predictions:
                rows = None
                break
            rows.append({"sequence_id": frame["sequence_id"], "frame_index": frame["frame_index"], "frame_id": frame["frame_id"], "timestamp": frame["timestamp"], "objects": predictions[key]})
        if rows:
            write_jsonl(out / ("%s.jsonl" % entry["sequence_id"]), rows)
            written.append(entry["sequence_id"])
    write_json(out / "detection_manifest.json", {"cfg": args.cfg, "checkpoint": args.ckpt, "infos": args.infos, "score_threshold": args.score_threshold, "sequences": written})
    print("written", len(written))


if __name__ == "__main__":
    main()
