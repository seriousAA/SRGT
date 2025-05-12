mmdet_base = "../../thirdparty/OBBDetection/configs/_base_"
_base_ = [
    f"{mmdet_base}/datasets/dota.py",
    f"{mmdet_base}/schedules/schedule_1x.py",
    f"{mmdet_base}/default_runtime.py",
    "global_setting_vars.py"
]

model = dict(
    type='OBBDinoDETR',
    init_cfg=dict(
        type='Pretrained', 
        checkpoint='checkpoints/dino_detr_sup_12e_coco_ckpt.pth'),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    bbox_head=dict(
        type='SemiDinoDETROBBHead',
        num_classes="${num_classes}",
        num_query=900,
        num_feature_levels=5,
        num_backbone_outs=3,
        backbone_channels=[512, 1024, 2048],
        dn_number=100,
        random_refpoints_xy=False,
        bbox_embed_diff_each_layer=False,
        in_channels=2048,
        transformer=dict(type='DinoTransformer', num_queries=900, num_feature_levels=5),
        positional_encoding=dict(
            type='SinePositionalEncodingHW', 
            temperatureH=20, temperatureW=20,
            num_feats=128,
            normalize=True),
        loss_cls1=dict(
            type='TaskAlignedFocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            loss_weight=1.0),
        loss_cls2=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='RotatedIoULoss', loss_weight=2.0),
        loss_enc_iou=dict(type='GIoULoss', loss_weight=2.0)),)

# training and testing settings
train_cfg=dict(
    assigner1=dict(
        type='O2MAssigner'),
    assigner2=dict(
        type='RotatedHungarianAssigner',
        cls_cost=dict(type='FocalLossCost', weight=2.0),
        reg_cost=dict(type='BBoxL1Cost', weight=5.0),
        iou_cost=dict(type='RotatedIoUCost', iou_mode='iou', weight=2.0),
        iou_cost_hbb=dict(type='IoUCost', iou_mode='giou', weight=2.0)))

test_cfg=dict(max_per_img=400)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='LoadOBBAnnotations', with_bbox=True, with_label=True, obb_as_mask=True),
    dict(type='LoadDOTASpecialInfo'),
    dict(
            type="RandResize",
            img_scale=[(1333, 1024), (1333, 1200)],
            multiscale_mode="range",
            keep_ratio=True,
    ),
    dict(type='hvRandomFlipOBB', h_flip_ratio=0.5, v_flip_ratio=0.5),
    dict(
        type="OneOf",
        transforms=[
            dict(type=k)
            for k in [
                "Identity",
                "AutoContrast",
                "RandEqualize",
                "RandSolarize",
                "RandColor",
                "RandContrast",
                "RandBrightness",
                "RandSharpness",
                "RandPosterize",
            ]
        ],
    ),
    dict(type="RandRotate", rotate_after_flip=True, keep_iof_thr=0.2,
         angles=(-90, 90), vert_rate=0.5, vert_cls=['roundabout', 'storage-tank']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # dict(type='DOTASpecialIgnore', ignore_size=2),
    dict(type='FliterEmpty'),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type='OBBDefaultFormatBundle'),
    dict(
        type="OBBCollect",
        keys=["img", "gt_bboxes", "gt_obboxes", "gt_labels"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "img_norm_cfg",
            "pad_shape",
            "scale_factor",
            "img_info"
        ),
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipRotateAug',
        img_scale=[(1024, 1024)],
        h_flip=False,
        v_flip=False,
        rotate=False,
        transforms=[
            dict(type='RandResize', keep_ratio=True),
            dict(type='hvRandomFlipOBB'),
            dict(type='RandRotate', rotate_after_flip=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='OBBCollect', keys=['img']),
        ])
]

data_root=None
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="DOTADataset",
        task='Task1',
        ann_file=None,
        img_prefix=None,
        pipeline=train_pipeline
    ),
    val=dict(
        type='DOTADataset',
        task='Task1',
        ann_file=None,
        img_prefix=None,
        fp_ratio='all',
        filter_empty_gt=False,
        pipeline=test_pipeline
    ),
    test=dict(
        type='DOTADataset',
        task='Task1',
        ann_file=None,
        img_prefix=None,
        pipeline=test_pipeline
    )
)

# optimizer
optimizer = dict(
    _delete_=True, 
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))