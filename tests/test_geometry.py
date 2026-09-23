import numpy as np

from inframot3d.geometry import giou_3d, iou_3d


def test_identical_box_iou():
    box = [0.0, 0.0, 0.0, 0.2, 4.0, 2.0, 1.5]
    assert np.isclose(iou_3d(box, box), 1.0)
    assert np.isclose(giou_3d(box, box), 1.0)


def test_disjoint_box_iou():
    first = [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5]
    second = [20.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5]
    assert iou_3d(first, second) == 0.0
    assert giou_3d(first, second) < 0.0
