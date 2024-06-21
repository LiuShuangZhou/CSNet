# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Tuple
import warnings
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.config import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType


@MODELS.register_module()
class FusedHead(BaseModule):
    r"""Multi-level fused semantic segmentation head.

    .. code-block:: none

        in_1 -> 1x1 conv ---
                            |
        in_2 -> 1x1 conv -- |
                           ||
        in_3 -> 1x1 conv - ||
                          |||                  /-> 1x1 conv (mask prediction)
        in_4 -> 1x1 conv -----> 3x3 convs (*4)
                            |                  \-> 1x1 conv (feature)
        in_5 -> 1x1 conv ---
    """  # noqa: W605

    def __init__(
        self,
        num_ins: int,
        fusion_level: int,
        num_convs: int = 4,
        in_channels: int = 256,
        conv_out_channels: int = 256,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        init_cfg: MultiConfig = dict(type='Kaiming')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_ins = num_ins
        self.fusion_level = fusion_level
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_ins):
            self.lateral_convs.append(
                ConvModule(
                    self.in_channels,
                    self.in_channels,
                    1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    inplace=False))

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = self.in_channels if i == 0 else conv_out_channels
            self.convs.append(
                ConvModule(
                    in_channels,
                    conv_out_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function.

        Args:
            feats (tuple[Tensor]): Multi scale feature maps.

        Returns:
            tuple[Tensor]:

                - mask_preds (Tensor): Predicted mask logits.
                - x (Tensor): Fused feature.
        """
        x = self.lateral_convs[self.fusion_level](feats[self.fusion_level])
        fused_size = tuple(x.shape[-2:])
        for i, feat in enumerate(feats):
            if i != self.fusion_level:
                feat = F.interpolate(
                    feat, size=fused_size, mode='bilinear', align_corners=True)
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                x = x + self.lateral_convs[i](feat)

        for i in range(self.num_convs):
            x = self.convs[i](x)
        # print(x.shape)
        return x

