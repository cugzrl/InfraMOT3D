from abc import ABC, abstractmethod


class TrajectoryPredictor(ABC):
    @abstractmethod
    def predict(self, track_history):
        raise NotImplementedError
