# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class SsddDataset(CocoDataset):
    """Dataset for COCO."""
    # SSDD
    METAINFO = {
        'classes':
            ('ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(119, 11, 32)]
    }


@DATASETS.register_module()
class NwpuDataset(CocoDataset):
    """Dataset for COCO."""
    # NWPU
    METAINFO = {
        'classes':
            ('airplane', 'ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
             'basketball_court', 'ground_track_field', 'harbor', 'bridge', 'vehicle'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)]
    }