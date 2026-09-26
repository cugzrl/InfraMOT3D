from pathlib import Path

from inframot3d.evaluation.dair_reference import run_official_metrics
from inframot3d.evaluation.protocols import build_protocol
from inframot3d.io import read_json, write_json


class UnifiedMOTEvaluator:
    def __init__(self, root, protocol_name="v2xseq"):
        self.root = Path(root)
        self.protocol = build_protocol(protocol_name, self.root)

    def evaluate(self, config, prediction_root, output_dir, split="val", sequences=None, score_threshold=None):
        if sequences is None:
            split_ids = read_json(self.root / config["split_file"])[split]
        else:
            split_ids = list(sequences)
        output_dir = Path(output_dir)
        kitti_dir = output_dir / "kitti"
        count = self.protocol.export_split(
            config["project"]["converted_root"],
            prediction_root,
            config["project"]["data_root"],
            split_ids,
            kitti_dir,
        )
        raw = run_official_metrics(
            self.root,
            kitti_dir,
            name=output_dir.name,
            score_threshold=score_threshold,
        )
        metrics = self.protocol.aggregate_metrics(raw)
        metrics["sequences"] = int(count)
        write_json(output_dir / "metrics.json", metrics)
        return metrics
