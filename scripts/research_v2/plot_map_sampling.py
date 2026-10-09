"""检查地图用法 A 的采样点：车道切向网格与车道中心线采样是否落在车道上"""

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.perception.lane_context import _parse_xy
from inframot3d.tcpn.sampling import LaneSampler, candidate_sites

R = ROOT / "outputs/research/roadside_joint_perception_v2"


def main():
    sampler = LaneSampler(ROOT)
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (sequence_id, k) in zip(axes, (("0003", 20), ("0084", 30))):
        samples = pickle.load(open(R / "samples/full" / ("%s.pkl" % sequence_id), "rb"))
        s = samples[k]
        keep = s["cand_score"] >= 0.3
        cbox = s["cand_box"][keep]
        lane = sampler.lane_geometry(sequence_id, s["frame_id"], cbox)
        rot, trans, loc = sampler._pose(sequence_id, s["frame_id"])
        lanes, _, _ = sampler._map(loc)
        for item in lanes.values():
            local = (rot[:2, :2].T @ (item["xy"] - trans[:2]).T).T
            if np.hypot(local[:, 0], local[:, 1]).min() < 110:
                ax.plot(local[:, 0], local[:, 1], c="0.8", lw=0.6)
        grid = candidate_sites("lane", cbox, lane)
        line = candidate_sites("laneline", cbox, lane)
        ax.scatter(grid[..., 0].ravel(), grid[..., 1].ravel(), s=2, c="tab:blue", label="lane-aligned grid")
        ax.scatter(line[..., 0].ravel(), line[..., 1].ravel(), s=2, c="tab:red", label="lane centerline points")
        ax.scatter(cbox[:, 0], cbox[:, 1], s=12, c="k", marker="x", label="candidates (score >= 0.3)")
        ax.set_xlim(0, 108)
        ax.set_ylim(-48, 40)
        ax.set_aspect("equal")
        ax.set_title("%s seq %s frame %s, lane matched %d/%d" % (loc, sequence_id, s["frame_id"], int(lane[2].sum()), len(cbox)))
        ax.legend(loc="lower left", fontsize=8)
    out = R / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "map_sampling_check.png", dpi=130)
    for sequence_id in ("0000", "0003", "0084", "0063"):
        path = R / "samples/full" / ("%s.pkl" % sequence_id)
        if not path.exists():
            continue
        samples = pickle.load(open(path, "rb"))
        tp_valid, all_valid, n_tp, n_all = 0, 0, 0, 0
        for s in samples[::10]:
            lane = sampler.lane_geometry(sequence_id, s["frame_id"], s["cand_box"])
            tp = s["cand_status"] == 0
            tp_valid += int(lane[2][tp].sum())
            n_tp += int(tp.sum())
            all_valid += int(lane[2].sum())
            n_all += len(lane[2])
        stats[sequence_id] = {"tp_lane_matched": tp_valid / max(n_tp, 1), "all_lane_matched": all_valid / max(n_all, 1)}
    (R / "diagnostics/map_sampling_check.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
