_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
num_classes = 10
norm_cfg = dict(type='LN', requires_grad=True)
chan = 256
image_size = (1024, 1024)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
sam_model = 'vit_h'
if sam_model == 'vit_h':
    encoder_embed_dim = 1280
    encoder_depth = 32
    encoder_num_heads = 16
    encoder_global_attn_indexes = [7, 15, 23, 31]
    checkpoint1 = '.././checkpoint/image_encoder_vit_h.pth'
    checkpoint2 = '.././checkpoint/mask_decoder_vit_h.pth'
    checkpoint3 = '.././checkpoint/prompt_encoder_vit_h.pth'
elif sam_model == 'vit_l':
    encoder_embed_dim = 1024
    encoder_depth = 24
    encoder_num_heads = 16
    encoder_global_attn_indexes = [5, 11, 17, 23]
    checkpoint1 = '.././checkpoint/image_encoder_vit_l.pth'
    checkpoint2 = '.././checkpoint/mask_decoder_vit_l.pth'
    checkpoint3 = '.././checkpoint/prompt_encoder_vit_l.pth'
elif sam_model == 'vit_b':
    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = [2, 5, 8, 11]
    checkpoint1 = '.././checkpoint/image_encoder_vit_b.pth'
    checkpoint2 = '.././checkpoint/mask_decoder_vit_b.pth'
    checkpoint3 = '.././checkpoint/prompt_encoder_vit_b.pth'

# model settings
model = dict(
    type='SamCascadeRCNN',
    # init_cfg=dict(type='Pretrained', checkpoint=''),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ImageEncoderViT',
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=1024,
        mlp_ratio=4,
        num_heads=encoder_num_heads,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=256,
        checkpoint=checkpoint1),
    neck=dict(
        type='InterpolateFPN',
        patch_size=16,
        embed_dim=encoder_embed_dim,
        out_channels=chan),
    rpn_head=dict(
        type='RPNHead',
        in_channels=chan,
        feat_channels=chan,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeSamAidRoIHead',
        num_stages=3,
        with_context=True,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=chan,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ContextShared2FCBBoxHead',
                in_channels=chan,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                # norm_cfg=norm_cfg,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                ),
            dict(
                type='ContextShared2FCBBoxHead',
                in_channels=chan,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                # norm_cfg=norm_cfg,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            dict(
                type='ContextShared2FCBBoxHead',
                in_channels=chan,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                # norm_cfg=norm_cfg,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            )
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=chan,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SAMFCNMaskHead',
            num_convs=4,
            in_channels=chan*2,
            conv_out_channels=chan*2,
            num_classes=num_classes,
            # class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        sam_mask_head=dict(
            type='GTFCNMaskHead',
            num_convs=4,
            in_channels=chan,
            conv_out_channels=chan,
            num_classes=num_classes,
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        ),
        mask_decoder=dict(
            type='MaskDecoder',
            transformer_dim=256,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            checkpoint=checkpoint2),
        prompt_encoder=dict(
            type='PromptEncoder',
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
            checkpoint=checkpoint3),
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            ),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False,
            )
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='soft_nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5,
        )))

dataset_type = 'NwpuDataset'
data_root = '.././data/NWPU/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.25),
        keep_ratio=False),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=False),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

epoch = 6
max_epochs = 12 * epoch
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8 * epoch, 11 * epoch],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)