from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


def identity_f1(trajectories, ignored, gt_count, pred_count):
    # 在官方3D匹配结果上按序列做身份匈牙利
    pairs = Counter()
    for sequence_index, (tracks, skips) in enumerate(zip(trajectories, ignored)):
        for gt_id, assignments in tracks.items():
            skip = skips[gt_id]
            for frame_index, pred_id in enumerate(assignments):
                if skip[frame_index] or int(pred_id) < 0:
                    continue
                pairs[(sequence_index, int(gt_id), int(pred_id))] += 1
    grouped = defaultdict(list)
    for (sequence_index, gt_id, pred_id), count in pairs.items():
        grouped[sequence_index].append((gt_id, pred_id, count))
    idtp = 0
    for values in grouped.values():
        gt_ids = sorted({value[0] for value in values})
        pred_ids = sorted({value[1] for value in values})
        gt_index = {key: index for index, key in enumerate(gt_ids)}
        pred_index = {key: index for index, key in enumerate(pred_ids)}
        matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
        for gt_id, pred_id, count in values:
            matrix[gt_index[gt_id], pred_index[pred_id]] = count
        rows, columns = linear_sum_assignment(-matrix)
        idtp += int(matrix[rows, columns].sum())
    idfp = int(pred_count) - idtp
    idfn = int(gt_count) - idtp
    denom = 2 * idtp + idfp + idfn
    return 2.0 * idtp / denom if denom else 0.0
