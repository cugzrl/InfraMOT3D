"""冻结 CenterPoint，只读取当前帧 BEV，不改检测头权重"""

import os
from pathlib import Path

import numpy as np
import torch

from inframot3d.detection.openpcdet_adapter import ensure_openpcdet, load_openpcdet_cfg


def load_frozen_centerpoint(openpcdet_root, cfg_file, checkpoint, device):
    ensure_openpcdet(openpcdet_root)
    cfg = load_openpcdet_cfg(cfg_file)
    tools = Path(cfg_file).resolve().parents[2]
    cwd = Path.cwd()
    os.chdir(tools)
    try:
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network, load_data_to_gpu
    finally:
        os.chdir(cwd)
    logger = type("Logger", (), {"info": staticmethod(lambda message: None)})()
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(checkpoint), logger=logger)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, dataset, load_data_to_gpu


def batch_from_points(dataset, load_data_to_gpu, points, device):
    sample = dataset.prepare_data({"frame_id": "query", "points": np.asarray(points[:, :4], dtype=np.float32)})
    batch = dataset.collate_batch([sample])
    load_data_to_gpu(batch)
    return batch


def bev_feature(model, batch):
    """跑到二维骨干为止，检测头不参与，因此不会改写已有检测结果"""
    with torch.no_grad():
        for module in model.module_list:
            if module is model.dense_head:
                break
            batch = module(batch)
    feature = batch["spatial_features_2d"]
    return feature
