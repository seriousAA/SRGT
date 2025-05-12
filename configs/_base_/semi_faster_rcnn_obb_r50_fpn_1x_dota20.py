mmdet_base = "../../thirdparty/OBBDetection/configs/"
_base_ = [
    f"{mmdet_base}/faster_rcnn_obb/faster_rcnn_obb_r50_fpn_1x_dota10.py",
    "global_setting_vars.py"
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes="${num_classes}"
        ),
    ),
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadImageFromFile", file_client_args=dict(backend="${backend}")),
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
    dict(type="ExtraAttrs", tag="sup"),
    dict(type='OBBDefaultFormatBundle'),
    dict(
        type="OBBCollect",
        keys=["img", "gt_bboxes", "gt_obboxes", "gt_labels"],
        meta_keys=(
            "filename", "ori_shape", "img_shape", "img_norm_cfg",
            "pad_shape", "scale_factor", "tag"
        ),
    ),
]

strong_pipeline = [
    dict(
        type="Sequential",
        transforms=[
            dict(
                type="RandResize",
                img_scale=[(1333, 1024)]
            ),
            dict(type='hvRandomFlipOBB', h_flip_ratio=0.5, v_flip_ratio=0.5),
            dict(
                type="ShuffledSequential",
                transforms=[
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
                    dict(
                        type="OneOf",
                        transforms=[
                            dict(type="RandTranslate", x=(-0.1, 0.1)),
                            dict(type="RandTranslate", y=(-0.1, 0.1)),
                            dict(type="RandRotate", rotate_after_flip=True, keep_iof_thr=0.2,
                                angles=(-90, 90), vert_rate=1.0, vert_cls=['roundabout', 'storage-tank']),
                            dict(type="RandShear", x=(-10, 10)),
                            dict(type="RandShear", y=(-10, 10)),
                        ],
                    ),
                ],
            ),
            dict(
                type="RandErase",
                n_iterations=(1, 5),
                size=[0, 0.1],
                squared=False,
            ),
        ],
        record=True,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type="ExtraAttrs", tag="unsup_student"),
    dict(type="OBBDefaultFormatBundle"),
    dict(
        type="OBBCollect",
        keys=["img", "gt_bboxes", "gt_obboxes", "gt_labels"],
        meta_keys=(
            "filename", "ori_shape", "img_shape", "img_norm_cfg",
            "pad_shape", "scale_factor", "tag", "transform_matrix",
            "h_flip", "v_flip"
        ),
    ),
]
weak_pipeline = [
    dict(type="Sequential",
        transforms=[
        dict(
                type="RandResize",
                img_scale=[(1333, 1024), (1333, 1200)],
                multiscale_mode="range",
                keep_ratio=True,
        ),
        dict(type='hvRandomFlipOBB', h_flip_ratio=0.5, v_flip_ratio=0.5)],
        record=True,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='Mask2OBB', obb_type='obb'),
    dict(type="ExtraAttrs", tag="unsup_teacher"),
    dict(type='OBBDefaultFormatBundle'),
    dict(
        type="OBBCollect",
        keys=["img", "gt_bboxes", "gt_obboxes", "gt_labels"],
        meta_keys=(
            "filename", "ori_shape", "img_shape", "img_norm_cfg",
            "pad_shape", "scale_factor", "tag", "transform_matrix",
            "h_flip", "v_flip"
        ),
    ),
]
unsup_pipeline = [
    dict(type="LoadImageFromFile", file_client_args=dict(backend="${backend}")),
    dict(type='LoadOBBAnnotations', with_bbox=True, with_label=True, obb_as_mask=True),
    # generate fake labels for data format compatibility
    # dict(type="PseudoSamples", with_bbox=True),
    dict(
        type="MultiBranch", unsup_student=strong_pipeline, unsup_teacher=weak_pipeline
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

data_root = None
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type="SemiDataset",
        sup=dict(
            type="DOTADataset",
            task='Task1',
            ann_file=None,
            img_prefix=None,
            pipeline=train_pipeline,
        ),
        unsup=dict(
            type="DOTADataset",
            task='Task1',
            ann_file=None,
            img_prefix=None,
            fp_ratio='all',
            filter_empty_gt=False,
            pipeline=unsup_pipeline,
        ),
    ),
    val=dict(
        type='DOTADataset',
        task='Task1',
        ann_file=None,
        img_prefix=None,
        fp_ratio='all',
        filter_empty_gt=False,
        pipeline=test_pipeline),
    test=dict(
        type='DOTADataset',
        task='Task1',
        ann_file=None,
        img_prefix=None,
        pipeline=test_pipeline),
    sampler=dict(
        train=dict(
            type="MultiSourceSampler",
            sample_ratio=[1, 1]
        )
    ),
)

semi_wrapper = dict(
    model="${model}",
    train_cfg=dict(
        unsup_weight=0.5,
    ),
    test_cfg=dict(inference_on="teacher")
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type="GetCurrentIter"),
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type="SemiTextLoggerHook", by_epoch=False)
    ],
)
