import torch.nn as nn
from .norm import get_norm
from .conv import Conv2d
import math
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.registry import MODELS
import torch
from .pafpn import PAFPN


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


@MODELS.register_module()
class InterpolateFPN(BaseModule):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        patch_size,
        embed_dim,
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        # top_block=LastLevelMaxPool(),
        top_block=None,
        norm="LN",
        square_pad=1024,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(InterpolateFPN, self).__init__()
        num = 4
        out = 64
        self.fusion = PAFPN(in_channels=[out * num, out * num, out * num, out * num],
                             out_channels=out_channels, num_outs=5)
        self.scale_factors = scale_factors
        self.size = [(256, 256), (128, 128), (64, 64), (32, 32)]
        strides = [int(patch_size / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)
        self.fpn = []
        for i in range(num):
            dim = embed_dim
            self.stages = []
            use_bias = norm == ""
            for idx, scale in enumerate(scale_factors):
                layers = [
                        Conv2d(dim, out, kernel_size=1, bias=use_bias, norm=get_norm(norm, out)),
                    ]
                layers = nn.Sequential(*layers)

                stage = int(math.log2(strides[idx]))
                self.add_module(f"simfp_{i}_{stage}", layers)
                self.stages.append(layers)
            self.fpn.append(self.stages)

        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def singe_forward(self, x, stages):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """

        features = x
        results = []
        for i, stage in enumerate(stages):
            features = F.interpolate(
                features,
                self.size[i],
                mode="bilinear",
                align_corners=False,
            )
            results.append(stage(features))
            # results.append(features)
        if self.top_block is not None:
            top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        # return x, y, {f: res for f, res in zip(self._out_features, results)}
        return results

    def forward(self, x):
        fpn = []
        for i in range(4):
            fpn.append(self.singe_forward(x[i], self.fpn[i]))
        results = []
        for i in range(4):
            results.append(torch.cat([fpn[0][i], fpn[1][i], fpn[2][i], fpn[3][i]], dim=1))
        results = self.fusion(results)
        return tuple(results)


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )