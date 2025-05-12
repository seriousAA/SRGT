_base_ = "../_base_/dino_detr_semi_head_r50_dota20.py"


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

fold = 1
percent = 1

data_root='data/DOTA/DOTA-v2.0_all/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="DOTADataset",
        task='Task1',
        ann_file=data_root+"trainval/semi_supervised/${fold}_${percent}/labeled/",
        img_prefix=data_root+"trainval/images/",
        pipeline=train_pipeline
    ),
    val=dict(
        type='DOTADataset',
        task='Task1',
        samples_per_gpu="${val_samples_per_gpu}",
        ann_file=data_root + "val/annfiles/",
        img_prefix=data_root + "val/images/",
        fp_ratio='all',
        filter_empty_gt=False,
        pipeline=test_pipeline
    ),
    test=dict(
        type='DOTADataset',
        task='Task1',
        samples_per_gpu="${val_samples_per_gpu}",
        ann_file=data_root + "test_dev/annfiles/",
        img_prefix=data_root + "test_dev/images/",
        pipeline=test_pipeline
    )
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(step=[160000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, interval=10000, save_best='mAP', metric='mAP', skip_at_resume_point=True)

# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

auto_resume=True
