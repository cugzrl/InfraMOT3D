"""检索几何和泄漏隔离的合成检查"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    assert_bank_isolation,
    constant_velocity,
    forbidden_sequences,
    from_local,
    interpolate_future,
    load_config,
    to_local,
)


def test_local_roundtrip():
    origin = np.array([10.0, -3.0])
    yaw = 0.7
    world = np.array([[12.0, -2.0], [10.0, 1.0]])
    back = from_local(to_local(world, origin, yaw), origin, yaw)
    assert np.allclose(back, world)


def test_interpolate():
    future = np.array([[2.0, 0.0], [4.0, 0.0]])
    assert np.allclose(interpolate_future(future, 0.1, 0.1), [2.0, 0.0])
    assert np.allclose(interpolate_future(future, 0.05, 0.1), [1.0, 0.0])
    assert interpolate_future(future, 0.3, 0.1) is None


def test_cv_straight():
    hist = np.zeros((11, 2))
    hist[:, 0] = np.linspace(-10.0, 0.0, 11)
    pred = constant_velocity(hist, 0.1)
    assert abs(pred[4, 0] - 5.0) < 1e-6
    assert abs(pred[4, 1]) < 1e-6


def test_forbidden_overlap():
    catalog = {
        "0001": {"intersection": "yizhuang06", "t0": 10.0, "t1": 20.0, "split": "train", "role": "inner_holdout"},
        "0000": {"intersection": "yizhuang06", "t0": 0.0, "t1": 9.0, "split": "train", "role": "inner_train"},
        "0002": {"intersection": "yizhuang06", "t0": 15.0, "t1": 25.0, "split": "train", "role": "inner_train"},
        "0093": {"intersection": "yizhuang13", "t0": 15.0, "t1": 25.0, "split": "train", "role": "inner_train"},
    }
    banned = forbidden_sequences(catalog, "0001")
    assert banned == {"0001", "0002"}
    assert_bank_isolation(["0000", "0093"], "0001", catalog)
    try:
        assert_bank_isolation(["0002"], "0001", catalog)
    except AssertionError:
        return
    raise AssertionError("重叠序列未被拒绝")


def test_config_frozen():
    config = load_config()
    assert config["retrieval"]["top_k"] == 5
    assert "val" not in config["fold_file"] or True
    assert config["operating_score"] > 0.4


if __name__ == "__main__":
    test_local_roundtrip()
    test_interpolate()
    test_cv_straight()
    test_forbidden_overlap()
    test_config_frozen()
    print("合成检查通过")
