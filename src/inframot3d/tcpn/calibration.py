"""训练侧留出序列上的分数校准

F 把原始 CenterPoint 分数映射到候选为 TP 的概率，G 把网络质量分数映射到同一概率
新分数 s' = F^{-1}(G(q))，使 GRAE 原有的出生、两阶段关联与输出阈值保持同一含义
"""

import json

import numpy as np
from sklearn.isotonic import IsotonicRegression

GRID = np.linspace(0.0, 1.0, 2001)


class ScoreMapper:
    def __init__(self):
        self.raw_grid = None
        self.raw_prob = None
        self.q_model = None

    def fit(self, raw, prob, label):
        raw, prob, label = map(np.asarray, (raw, prob, label))
        keep = label >= 0
        raw, prob, label = raw[keep], prob[keep], label[keep]
        f = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(raw, label)
        values = f.predict(GRID)
        values = np.maximum.accumulate(values + GRID * 1e-6)
        self.raw_grid, self.raw_prob = GRID, values
        self.q_model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(prob, label)
        return self

    def calibrated(self, prob):
        return self.q_model.predict(np.asarray(prob, dtype=np.float64))

    def to_raw_scale(self, prob):
        p = self.calibrated(prob)
        return np.interp(p, self.raw_prob, self.raw_grid)

    def state(self):
        return {
            "raw_grid": self.raw_grid.tolist(),
            "raw_prob": self.raw_prob.tolist(),
            "q_x": self.q_model.X_thresholds_.tolist(),
            "q_y": self.q_model.y_thresholds_.tolist(),
        }

    @classmethod
    def from_state(cls, state):
        mapper = cls()
        mapper.raw_grid = np.asarray(state["raw_grid"])
        mapper.raw_prob = np.asarray(state["raw_prob"])
        model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
        x = np.asarray(state["q_x"])
        y = np.asarray(state["q_y"])
        model.fit(x, y)
        mapper.q_model = model
        return mapper


def fit_from_dump(path):
    raw, prob, label = [], [], []
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            raw.extend(row["raw"])
            prob.extend(row["prob"])
            label.extend(row["quality"])
    return ScoreMapper().fit(raw, prob, label)
