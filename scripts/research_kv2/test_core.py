"""检索定义、模式分离和匀速残差的小测试"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research_knowledge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_bank import window_fault  # noqa: E402
from common import sample_window  # noqa: E402
from kv import ctrv_path, cv_path, mode_blend, pointwise_error, residual_forecast  # noqa: E402


def _config():
    return {"retrieval": {"residual_cost_max": 3.5, "residual_endpoint_std_m": 2.5}}


def test_ctrv_matches_straight_cv():
    hist = np.stack([np.linspace(-10.0, 0.0, 11), np.zeros(11)], axis=1)
    assert np.allclose(ctrv_path(hist, 0.1, 20), cv_path(hist, 0.1, 20), atol=1e-4)


def test_mode_blend_does_not_mix_turns():
    left = np.stack([np.linspace(0.2, 10.0, 20), np.linspace(0.2, 4.0, 20)], axis=1)
    right = np.stack([np.linspace(0.2, 10.0, 20), np.linspace(-0.2, -4.0, 20)], axis=1)
    blended = mode_blend([left, right], np.array([0.1, 0.2]), np.array([2, 3]))
    assert np.allclose(blended, left)


def test_residual_falls_back_when_weak_or_split():
    config = _config()
    base = np.zeros((20, 2))
    weak = np.ones((1, 20, 2))
    pred, used = residual_forecast(base, weak, np.array([9.0]), np.array([1]), config)
    assert used is False and np.allclose(pred, base)
    split = np.stack([np.full((20, 2), [0.0, 4.0]), np.full((20, 2), [0.0, -4.0])])
    pred, used = residual_forecast(base, split, np.array([0.2, 0.3]), np.array([1, 1]), config)
    assert used is False and np.allclose(pred, base)
    same = np.stack([np.full((20, 2), [0.0, 1.0]), np.full((20, 2), [0.0, 1.2])])
    pred, used = residual_forecast(base, same, np.array([0.2, 0.3]), np.array([2, 2]), config)
    assert used is True
    assert pred[-1, 1] > 0.5


def test_pointwise_can_stitch_candidates():
    future = np.stack([np.linspace(1.0, 20.0, 20), np.zeros(20)], axis=1)
    early = future.copy()
    early[10:] += 5.0
    late = future.copy()
    late[:10] += 5.0
    error = pointwise_error(np.stack([early, late]), future)
    assert float(error.max()) < 1e-6


def test_sampling_rejects_repeated_and_leaked_times():
    track = {"time": np.array([0.0, 0.1, 0.2, 0.3])}
    assert window_fault(track, 0.2, np.array([0, 1, 1]), np.array([3]), np.array([True])) == "duplicate_history"
    assert window_fault(track, 0.2, np.array([0, 1, 2]), np.array([2]), np.array([True])) == "future_not_after_anchor"
    track = {"time": np.arange(0.0, 4.0, 0.1)}
    anchor = 1.0
    hist_index, hist_ok, fut_index, fut_ok = sample_window(track, anchor, 1.0, 2.0, 0.1, 0.12)
    assert hist_ok.all() and fut_ok.all()
    assert track["time"][hist_index].max() <= anchor + 1e-9
    assert track["time"][fut_index].min() > anchor
    assert window_fault(track, anchor, hist_index, fut_index, fut_ok) == ""


if __name__ == "__main__":
    test_ctrv_matches_straight_cv()
    test_mode_blend_does_not_mix_turns()
    test_residual_falls_back_when_weak_or_split()
    test_pointwise_can_stitch_candidates()
    test_sampling_rejects_repeated_and_leaked_times()
    print("核心测试通过")
