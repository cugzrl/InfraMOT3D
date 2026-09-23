import numpy as np

from inframot3d.visualization.open3d_sequence import _box_corners, _box_lines


def test_box_geometry():
    box = [1.0, 2.0, 3.0, 0.0, 4.0, 2.0, 1.0]
    corners = _box_corners(box)
    assert corners.shape == (8, 3)
    assert np.allclose(corners.mean(axis=0), [1.0, 2.0, 3.0])


def test_line_set_geometry():
    objects = [{"box": [0, 0, 0, 0, 4, 2, 1], "track_id": 1}]
    line_set = _box_lines(objects, lambda _: [1.0, 0.0, 0.0])
    assert len(line_set.points) == 8
    assert len(line_set.lines) == 12
