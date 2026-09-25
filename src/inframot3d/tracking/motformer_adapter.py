import math
import sys
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment


def ensure_motformer(root):
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def to_motformer_box(box):
    # InfraMOT3D是x y z yaw length width height
    x, y, z, yaw, length, width, height = [float(value) for value in box]
    assert length > 0.0 and width > 0.0 and height > 0.0
    assert math.isfinite(yaw)
    # 官方内部框是x y z width length height yaw
    converted = [x, y, z, width, length, height, yaw]
    assert len(converted) == 7
    return converted


def from_motformer_box(box):
    x, y, z, width, length, height, yaw = [float(value) for value in box]
    assert width > 0.0 and length > 0.0 and height > 0.0
    assert math.isfinite(yaw) and all(math.isfinite(value) for value in (x, y, z))
    return [x, y, z, yaw, length, width, height]


def configure_runtime(root, class_names, lidar_interval, max_velo):
    ensure_motformer(root)
    import utils.graph_util as graph_util
    from utils.data_util import NuScenesClasses

    graph_util.LIDAR_INTERVAL = float(lidar_interval)
    graph_util.MAX_VELO = [float(value) for value in max_velo]
    assert len(graph_util.MAX_VELO) == len(class_names)
    NuScenesClasses.clear()
    for index, name in enumerate(class_names):
        NuScenesClasses[name] = index


def _corners(box):
    x, y, _, yaw, length, width, _ = box
    local = np.array(
        [
            [length / 2.0, width / 2.0],
            [-length / 2.0, width / 2.0],
            [-length / 2.0, -width / 2.0],
            [length / 2.0, -width / 2.0],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.array([x, y], dtype=np.float64)


def _bev_iou(box_a, box_b):
    poly_a = Polygon(_corners(box_a))
    poly_b = Polygon(_corners(box_b))
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if poly_a.is_empty or poly_b.is_empty or not poly_a.intersects(poly_b):
        return 0.0
    intersection = poly_a.intersection(poly_b).area
    union = poly_a.area + poly_b.area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def match_tracks(detections, ground_truth):
    # 同类别BEV IoU大于0才把GT track_id写进训练监督
    count = len(detections)
    matched = -np.ones(count, dtype=np.int64)
    next_exist = np.zeros(count, dtype=bool)
    next_trans = np.zeros((count, 2), dtype=np.float32)
    if count == 0 or not ground_truth:
        return matched, next_exist, next_trans
    cost = np.full((count, len(ground_truth)), 1e6, dtype=np.float64)
    for row, det in enumerate(detections):
        for column, gt in enumerate(ground_truth):
            if det["class_name"] != gt["class_name"]:
                continue
            iou = _bev_iou(det["box"], gt["box"])
            if iou > 0.0:
                cost[row, column] = -iou
    rows, columns = linear_sum_assignment(cost)
    for row, column in zip(rows, columns):
        if cost[row, column] >= 1e5:
            continue
        matched[row] = int(ground_truth[column]["track_id"])
        if ground_truth[column]["next_box"] is not None:
            next_exist[row] = True
            next_box = ground_truth[column]["next_box"]
            next_trans[row, 0] = float(next_box[0])
            next_trans[row, 1] = float(next_box[1])
    return matched, next_exist, next_trans


def build_model(root, model_cfg, num_classes, device):
    ensure_motformer(root)
    from model.model import STTransformerModel

    model = STTransformerModel(
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        dropout=float(model_cfg["dropout"]),
        encoder_nlayers=int(model_cfg["encoder_nlayers"]),
        decoder_nlayers=int(model_cfg["decoder_nlayers"]),
        norm_first=bool(model_cfg["norm_first"]),
        cross_attn_value_gate=bool(model_cfg["cross_attn_value_gate"]),
        num_classes=int(num_classes),
    )
    return model.to(device)


def load_checkpoint(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return checkpoint


def frame_tensors(objects, class_to_index, device, graph_truncation_dist):
    from torch_geometric.data import Data
    from utils.data_util import torch_one_hot
    import utils.graph_util as graph_util

    boxes, velocities, classes, scores = [], [], [], []
    for item in objects:
        name = item["class_name"]
        if name not in class_to_index:
            continue
        boxes.append(to_motformer_box(item["box"]))
        # detector没有速度头，训练和推理都填0
        velocities.append([0.0, 0.0])
        classes.append(int(class_to_index[name]))
        scores.append(float(item["score"]))
    if not boxes:
        empty = torch.zeros((0, 7), dtype=torch.float32, device=device)
        data = Data(
            x=torch.zeros((0, 10 + len(class_to_index)), dtype=torch.float32, device=device),
            edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
            tracking_id=torch.zeros((0,), dtype=torch.long, device=device),
            det_box=empty,
            det_velo=torch.zeros((0, 2), dtype=torch.float32, device=device),
            det_class=torch.zeros((0,), dtype=torch.long, device=device),
            det_score=torch.zeros((0, 1), dtype=torch.float32, device=device),
        )
        return data
    det_box = torch.tensor(boxes, dtype=torch.float32, device=device)
    det_velo = torch.tensor(velocities, dtype=torch.float32, device=device)
    det_class = torch.tensor(classes, dtype=torch.long, device=device)
    det_score = torch.tensor(scores, dtype=torch.float32, device=device).view(-1, 1)
    one_hot = torch_one_hot(det_class, len(class_to_index))
    features = torch.cat([det_box, det_velo, one_hot, det_score], dim=1)
    adjacency = graph_util.bev_euclidean_distance_adj(det_box, det_class, float(graph_truncation_dist))
    edge_index = graph_util.adj_to_edge_index(adjacency).to(device)
    return Data(
        x=features,
        edge_index=edge_index,
        tracking_id=-torch.ones(len(boxes), dtype=torch.long, device=device),
        det_box=det_box,
        det_velo=det_velo,
        det_class=det_class,
        det_score=det_score,
    )


class MotformerTracker:
    def __init__(self, model, class_names, max_age, active_track_thresh, graph_truncation_dist, score_threshold=0.0):
        from eval.tracker import Tracker

        self.model = model
        self.class_names = list(class_names)
        self.class_to_index = {name: index for index, name in enumerate(self.class_names)}
        self.device = next(model.parameters()).device
        self.tracker = Tracker(
            max_age=int(max_age),
            active_track_thresh=float(active_track_thresh),
            feature_update_weight=1.0,
            hungarian=False,
            graph_truncation_dist=float(graph_truncation_dist),
        )
        self.graph_truncation_dist = float(graph_truncation_dist)
        self.score_threshold = float(score_threshold)

    def reset(self):
        self.tracker.reset()

    def update(self, objects, timestamp=None):
        import utils.graph_util as graph_util

        if timestamp is not None:
            assert math.isfinite(float(timestamp))
        cleaned = [
            {"class_name": item["class_name"], "score": float(item["score"]), "box": item["box"]}
            for item in objects
            if item["class_name"] in self.class_to_index and float(item["score"]) >= self.score_threshold
        ]
        data = frame_tensors(cleaned, self.class_to_index, self.device, self.graph_truncation_dist)
        if data.x.size(0) == 0:
            return []
        with torch.no_grad():
            if not self.tracker.is_initialized:
                self.tracker.init_tracks(data)
            else:
                det_class = data.det_class
                track_class = self.tracker.track_state["classes"]
                det_batch = torch.zeros_like(det_class)
                track_batch = torch.zeros_like(track_class)
                inter_graph = graph_util.build_inter_graph(
                    data.det_box,
                    self.tracker.track_state["boxes"],
                    det_class,
                    track_class,
                    self.tracker.track_state["velo"],
                    self.tracker.track_state["age"],
                    det_batch,
                    track_batch,
                ).to_data_list()[0]
                affinity, track_feat, det_feat, pred_velo = self.model(
                    data.x,
                    self.tracker.track_state["features"],
                    data.edge_index,
                    self.tracker.track_state["edge_index"],
                    inter_graph.edge_index,
                    inter_graph.edge_attr,
                )
                affinity = torch.sigmoid(affinity[-1])
                self.tracker.track_step(affinity, pred_velo, inter_graph, data, track_feat, det_feat)
        outputs = []
        for item in self.tracker.track_info:
            score = float(item["tracking_score"])
            if int(item["active"]) == 0:
                score *= 0.1
            size = [float(value) for value in item["size"]]
            translation = [float(value) for value in item["translation"]]
            yaw = float(item["rotation"].angle if hasattr(item["rotation"], "angle") else _yaw_from_rotation(item["rotation"]))
            box = from_motformer_box(translation + size + [yaw])
            outputs.append(
                {
                    "class_name": str(item["tracking_name"]),
                    "track_id": int(item["tracking_id"]),
                    "score": score,
                    "box": box,
                }
            )
        return outputs


def train_mini_sequence(model, criterion, data_seq, active_thresh, max_age, feature_weight, graph_dist):
    # 按官方在线自回归方式逐帧更新轨迹并计算边分类损失
    import utils.graph_util as graph_util
    from eval.tracker import update_feature, update_field
    from torch_geometric.data import Batch, Data
    from utils import match_util

    losses = []
    tracks = track_boxes = track_velo = edge_index_track = None
    track_class = track_batch = track_gt = track_age = None
    for index, data in enumerate(data_seq):
        if index == 0:
            tracks = data.x
            track_boxes = data.det_box
            track_velo = data.det_velo
            edge_index_track = data.edge_index
            track_class = data.det_class
            track_batch = data.x_batch
            track_gt = data.tracking_id
            track_age = torch.ones_like(track_class, dtype=torch.int)
            continue
        inter_graph = graph_util.build_inter_graph(
            data.det_box,
            track_boxes,
            data.det_class,
            track_class,
            track_velo,
            track_age,
            data.x_batch,
            track_batch,
        )
        affinity, track_feat, det_feat, pred_velo = model(
            data.x,
            tracks,
            data.edge_index,
            edge_index_track,
            inter_graph.edge_index,
            inter_graph.edge_attr,
        )
        target = criterion.generate_target(track_gt, data.tracking_id, inter_graph.edge_index)
        if target.numel() == 0:
            losses.append(pred_velo.sum() * 0.0)
        else:
            losses.append(criterion(affinity, target, pred_velo, data.velo_target, data.next_exist))
        if index == len(data_seq) - 1:
            continue
        affinity_sigmoid = torch.sigmoid(affinity[-1])
        updated = _update_training_tracks(
            affinity_sigmoid,
            inter_graph,
            det_feat,
            track_feat,
            data.det_box,
            track_boxes,
            pred_velo,
            track_velo,
            data.det_class,
            track_class,
            data.tracking_id,
            track_gt,
            track_age,
            data.x_batch,
            track_batch,
            active_thresh,
            max_age,
            feature_weight,
            graph_dist,
            graph_util,
            match_util,
            update_feature,
            update_field,
            Batch,
            Data,
        )
        tracks = updated.x
        track_boxes = updated.boxes
        track_velo = updated.velo
        track_class = updated.classes
        edge_index_track = updated.edge_index
        track_batch = updated.batch
        track_gt = updated.tracking_id
        track_age = updated.ages
    return losses


def _update_training_tracks(
    affinity,
    inter_graph,
    det_feat,
    track_feat,
    det_boxes,
    track_boxes,
    det_velo,
    track_velo,
    det_class,
    track_class,
    det_gt,
    track_gt,
    track_age,
    det_batch,
    track_batch,
    active_thresh,
    max_age,
    feature_weight,
    graph_dist,
    graph_util,
    match_util,
    update_feature,
    update_field,
    batch_cls,
    data_cls,
):
    from torch_geometric.utils import unbatch

    graphs = inter_graph.to_data_list()
    edge_batch = inter_graph.edge_index_batch
    affinity_list = unbatch(affinity, edge_batch)
    pieces = []
    lists = [
        unbatch(det_feat, det_batch),
        unbatch(track_feat, track_batch),
        unbatch(det_boxes, det_batch),
        unbatch(track_boxes, track_batch),
        unbatch(det_velo, det_batch),
        unbatch(track_velo, track_batch),
        unbatch(det_class, det_batch),
        unbatch(track_class, track_batch),
        unbatch(det_gt, det_batch),
        unbatch(track_gt, track_batch),
        unbatch(track_age, track_batch),
    ]
    for graph, score, det_f, trk_f, det_b, trk_b, det_v, trk_v, det_c, trk_c, det_g, trk_g, age in zip(graphs, affinity_list, *lists):
        edge_index = graph.edge_index
        num_tracks = graph.size_s
        num_dets = graph.size_t
        score = score.squeeze(1)
        adjacency = torch.zeros((num_tracks, num_dets), dtype=torch.bool, device=edge_index.device)
        dense = torch.zeros((num_tracks, num_dets), dtype=score.dtype, device=edge_index.device)
        adjacency[edge_index[0], edge_index[1]] = True
        dense[edge_index[0], edge_index[1]] = score
        match, new_det, unmatched = match_util.dets_tracks_matching(
            dense, adjacency, num_dets, num_tracks, active_thresh=active_thresh, hungarian=False
        )
        inactive = [index for index in unmatched if age[index] < max_age]
        new_feat = update_feature(det_f, trk_f, match, new_det, inactive, weight=feature_weight)
        new_boxes = update_field(det_b, trk_b, match, new_det, inactive, update_with_dets=True)
        new_velo = update_field(det_v, trk_v, match, new_det, inactive, update_with_dets=True)
        new_class = update_field(det_c, trk_c, match, new_det, inactive, update_with_dets=True)
        new_gt = update_field(det_g, trk_g, match, new_det, inactive, update_with_dets=True)
        fresh_age = torch.ones(len(match) + len(new_det), dtype=torch.int, device=age.device)
        new_age = torch.cat([fresh_age, age[inactive] + 1], 0)
        new_adj = graph_util.bev_euclidean_distance_adj(new_boxes, new_class, graph_dist)
        pieces.append(
            data_cls(
                x=new_feat,
                edge_index=graph_util.adj_to_edge_index(new_adj),
                boxes=new_boxes,
                velo=new_velo,
                classes=new_class,
                tracking_id=new_gt,
                ages=new_age,
            )
        )
    return batch_cls.from_data_list(pieces)


def _yaw_from_rotation(rotation):
    from pyquaternion import Quaternion

    quaternion = Quaternion(rotation)
    if quaternion.axis[-1] < 0:
        quaternion = -quaternion
    return float(quaternion.radians)
