_base_ = [
    '../_base_/datasets/dota.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='OrientedRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            target_means=[.0, .0, .0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='OBBStandardRoIHead',
        bbox_roi_extractor=dict(
            type='OBBSingleRoIExtractor',
            roi_layer=dict(type='RoIAlignRotated', out_size=7, sample_num=2),
            out_channels=256,
            extend_factor=(1.4, 1.2),
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='OBBShared2FCBBoxHead',
            start_bbox_type='obb',
            end_bbox_type='obb',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=16,
            bbox_coder=dict(
                type='OBB2OBBDeltaXYWHTCoder',
                target_means=[0., 0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2, 0.1]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                           loss_weight=1.0))))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            match_low_quality=True,
            gpu_assign_thr=200,
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
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=False,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='OBBOverlaps')),
        sampler=dict(
            type='OBBRandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.8,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='obb_nms', iou_thr=0.1), max_per_img=2000))

data_root='data/DOTA/DOTA-v1.5/ss/'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type="DOTADataset",
        task='Task1',
        ann_file=data_root+"/trainval/annfiles/",
        # ann_file=data_root+"/trainval/semi-supervised/2_1/labeled/",
        # ann_file=data_root+"/trainval/semi_supervised/${fold}_${percent}/labeled/",
        # img_prefix=data_root+"/trainval/images/",
        img_prefix=data_root+"/trainval/images/",
    ),
    val=dict(
        type='DOTADataset',
        task='Task1',
        # ann_file=data_root + "/val/annfiles/",
        ann_file=data_root + "/val/annfiles/",
        # ann_file=data_root + "/../DOTA-v2.0/trainval/semi_supervised/2_10/labeled/",
        # img_prefix=data_root + "/val/images/",
        img_prefix=data_root + "/val/images/",
        # img_prefix=data_root + "/../DOTA-v2.0/trainval/images/",
    ),
    test=dict(
        type='DOTADataset',
        task='Task1',
        # ann_file=data_root + "/test/annfiles/",
        ann_file=data_root + "/test/annfiles/",
        # ann_file=data_root + "/../DOTA-v1.5/ss/test/annfiles/",
        # ann_file=data_root + "/../DOTA-v2.0/trainval/annfiles/",
        # img_prefix=data_root + "/test/images/",
        img_prefix=data_root + "/test/images/",
        # img_prefix=data_root + "/../DOTA-v1.5/ss/test/images/",
        # img_prefix=data_root + "/../DOTA-v2.0/trainval/images/",
    )
)

custom_hooks = [
    dict(type="ProfileRecorder", log_dir="./profile_logs", log_freq=10)
]

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000, 160000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=10, create_symlink=False)
evaluation = dict(_delete_=True, interval=10000, save_best='mAP', metric='mAP')

# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

auto_resume=False
find_unused_parameters=True