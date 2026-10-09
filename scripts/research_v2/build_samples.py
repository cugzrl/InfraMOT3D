"""由检测与 GRAE 回放生成逐帧样本，GT 只写入标签字段"""

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "OpenPCDet"))

from inframot3d.io import read_json, read_jsonl
from inframot3d.tcpn.data import build_samples

CONVERTED = ROOT / "data/converted/v2x_seq_infrastructure"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = read_json(CONVERTED / "manifest.json")
    paths = {item["sequence_id"]: item["path"] for item in manifest["sequences"]}
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    for sequence_id in args.sequences:
        gt_rows = list(read_jsonl(CONVERTED / paths[sequence_id]))
        det_rows = list(read_jsonl(ROOT / args.detections / ("%s.jsonl" % sequence_id)))
        states = list(read_jsonl(ROOT / args.replay / "states" / ("%s.jsonl" % sequence_id)))
        samples = build_samples(sequence_id, det_rows, gt_rows, states)
        with (out / ("%s.pkl" % sequence_id)).open("wb") as stream:
            pickle.dump(samples, stream)
        print(sequence_id, len(samples))


if __name__ == "__main__":
    main()
