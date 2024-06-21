# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList, OptMultiConfig
from ..utils import unpack_gt_instances


class BaseSamMaskHead(BaseModule, metaclass=ABCMeta):
    """Base class for mask heads used in One-Stage Instance Segmentation."""

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

    @abstractmethod
    def loss_by_feat(self, *args, **kwargs):
        """Calculate the loss based on the features extracted by the mask
        head."""
        pass

    @abstractmethod
    def predict_by_feat(self, *args, **kwargs):
        """Transform a batch of output features extracted from the head into
        mask results."""
        pass

    def loss(self,
             x: Union[List[Tensor], Tuple[Tensor]],
             batch_data_samples: SampleList,
             positive_infos: OptInstanceList = None,
             image_embeddings: Tensor = None,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the mask head on
        the features of the upstream network.

        Args:
            x (list[Tensor] | tuple[Tensor]): Features from FPN.
                Each has a shape (B, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            positive_infos (list[:obj:`InstanceData`], optional): Information
                of positive samples. Used when the label assignment is
                done outside the MaskHead, e.g., BboxHead in
                YOLACT or CondInst, etc. When the label assignment is done in
                MaskHead, it would be None, like SOLO or SOLOv2. All values
                in it should have shape (num_positive_samples, *).


        Returns:
            dict: A dictionary of loss components.
        """
        if positive_infos is None:
            outs = self(x)
        else:
            outs = self(x, positive_infos, image_embeddings)

        assert isinstance(outs, tuple), 'Forward results should be a tuple, ' \
                                        'even if only one item is returned'

        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for gt_instances, img_metas in zip(batch_gt_instances,
                                           batch_img_metas):
            img_shape = img_metas['batch_input_shape']
            gt_masks = gt_instances.masks.pad(img_shape)
            gt_instances.masks = gt_masks

        losses = self.loss_by_feat(
            *outs,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            positive_infos=positive_infos,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            **kwargs)
        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False,
                results_list: OptInstanceList = None,
                image_embeddings: Tensor = None,
                **kwargs) -> InstanceList:
        """Test function without test-time augmentation.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
            results_list (list[obj:`InstanceData`], optional): Detection
                results of each image after the post process. Only exist
                if there is a `bbox_head`, like `YOLACT`, `CondInst`, etc.

        Returns:
            list[obj:`InstanceData`]: Instance segmentation
            results of each image after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance,)
                - labels (Tensor): Has a shape (num_instances,).
                - masks (Tensor): Processed mask results, has a
                  shape (num_instances, h, w).
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas \
            = outputs
        for gt_instances, img_metas in zip(batch_gt_instances,
                                           batch_img_metas):
            img_shape = img_metas['batch_input_shape']
            gt_masks = gt_instances.masks.pad(img_shape)
            gt_instances.masks = gt_masks
        if results_list is None:
            outs = self(x)
        else:
            outs = self(x, results_list, image_embeddings)

        results_list = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            rescale=rescale,
            results_list=results_list,
            **kwargs)

        return results_list


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))