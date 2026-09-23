from abc import ABC, abstractmethod


class MultimodalFusion(ABC):
    @abstractmethod
    def fuse(self, point_features, image_features, calibration):
        raise NotImplementedError
