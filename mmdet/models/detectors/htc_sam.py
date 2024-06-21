# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .cascade_rcnn import CascadeRCNN
from .sam_cascade_rcnn import SamCascadeRCNN


@MODELS.register_module()
class SamHybridTaskCascade(SamCascadeRCNN):
    """Implementation of `HTC <https://arxiv.org/abs/1901.07518>`_"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @property
    def with_semantic(self) -> bool:
        """bool: whether the detector has a semantic head"""
        return self.roi_head.with_semantic