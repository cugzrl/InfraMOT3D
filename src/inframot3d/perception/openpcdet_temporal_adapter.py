"""Minimal OpenPCDet bridge for CenterPoint and PAAF-style detectors."""

from torch import nn

from .dual_bev_memory import DualBEVMemory, DualMemoryState


class OpenPCDetTemporalAdapter(nn.Module):
    """Insert memory after `backbone_2d`, before the first detection head.

    The detector retains its native module order and loss/post-processing.
    Sequence state is explicit so a dataloader cannot accidentally share it.
    """

    def __init__(self, detector: nn.Module, memory: DualBEVMemory, feature_key: str = "spatial_features_2d"):
        super().__init__()
        self.detector = detector
        self.memory = memory
        self.feature_key = feature_key

    def forward_step(self, batch_dict, state: DualMemoryState, sequence_id: str,
                     timestamp: float, lidar_hit=None, enabled: bool = True):
        if not hasattr(self.detector, "backbone_2d"):
            raise ValueError("Detector has no BEV backbone insertion point")
        enhanced = False
        diagnostics = {}
        for module in self.detector.module_list:
            batch_dict = module(batch_dict)
            if module is self.detector.backbone_2d:
                current = batch_dict[self.feature_key]
                updated, state, diagnostics = self.memory(
                    current, state, sequence_id, timestamp, lidar_hit, enabled,
                )
                batch_dict[self.feature_key] = updated
                enhanced = True
        if not enhanced:
            raise RuntimeError("BEV backbone absent from module_list")
        return batch_dict, state, diagnostics
