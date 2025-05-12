_base_ = "../_base_/semi_faster_rcnn_obb_r50_fpn_1x_dota20.py"

data_root='data/DOTA/DOTA-v2.0_all/'

model = dict(
    roi_head=dict(
        type='OBBPseudoRoIHeadWithVar',
        bbox_head=dict(
            type='OBBConvFCBBoxHeadWithVar',
            num_classes="${num_classes}",
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0
            ),
            loss_bbox=dict(
                type='NLLoss', 
                beta=1.0, 
                nll_weight=0.05, 
                loss_weight=1.0
            )
        )
    )
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="DOTADataset",
            ann_file=data_root+"trainval/semi_supervised/${fold}_${percent}/labeled/",
            img_prefix=data_root+"trainval/images/",
        ),
        unsup=dict(
            type="DOTADataset",
            ann_file=data_root+"trainval/semi_supervised/${fold}_${percent}/unlabeled/",
            img_prefix=data_root+"trainval/images/",
        ),
    ),
    val=dict(
        type='DOTADataset',
        samples_per_gpu="${val_samples_per_gpu}",
        ann_file=data_root + "val/annfiles/",
        img_prefix=data_root + "val/images/",
    ),
    test=dict(
        type='DOTADataset',
        samples_per_gpu="${val_samples_per_gpu}",
        ann_file=data_root + "test_dev/annfiles/",
        img_prefix=data_root + "test_dev/images/",
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[2, 1],
        )
    ),
)

semi_wrapper = dict(
    type="UnbiasedTeacherV2",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=True,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.5,
        cls_reg_pseudo_threshold=0.5,
        min_pseduo_box_size=0,
        unsup_weight=0.5,
        use_hbb_rpn=True,
    # =======UBTeacherV2 Configs=======
        unsup_rcnn_reg=dict(
            type='TSBetterPseudoLoss',
            ts_better=0.1, 
            t_cert=0.5,
            loss_weight=1.0,
        )
    ),
    test_cfg=dict(inference_on="teacher")
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, type="SubModulesEvalHook", interval=10000, save_best='mAP', metric='mAP',
                  skip_at_resume_point=True)

auto_resume=True
find_unused_parameters=True