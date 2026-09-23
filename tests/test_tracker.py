from inframot3d.tracking import MultiClassAB3DMOT


def test_track_id_is_stable():
    config = {
        "score_threshold": 0.0,
        "vehicle": {
            "classes": ["Car"],
            "association": "hungarian",
            "metric": "giou_3d",
            "threshold": -0.2,
            "min_hits": 1,
            "max_age": 2,
        },
    }
    tracker = MultiClassAB3DMOT(config)
    first = tracker.update(
        [{"class_name": "Car", "source_track_id": "a", "score": 1.0, "box": [0, 0, 0, 0, 4, 2, 1.5]}]
    )
    second = tracker.update(
        [{"class_name": "Car", "source_track_id": "a", "score": 1.0, "box": [0.2, 0, 0, 0, 4, 2, 1.5]}]
    )
    assert first[0]["track_id"] == second[0]["track_id"]
