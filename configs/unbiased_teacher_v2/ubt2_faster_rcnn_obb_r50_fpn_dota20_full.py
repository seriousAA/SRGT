_base_ = "ubt2_faster_rcnn_obb_r50_fpn_dota20_partial.py"

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes="${num_classes}"
        ),
    ),
)

data_root='data/dota_2.0_gsd/'

data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="DOTADataset",
            ann_file=data_root+"trainval_split/annfiles/",
            img_prefix=data_root+"trainval_split/images/",
        ),
        unsup=dict(
            type="DOTADataset",
            ann_file=data_root+"test_challenge_split/annfiles/",
            img_prefix=data_root+"test_challenge_split/images/",
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
            sample_ratio=[3, 2],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        use_teacher_proposal=True,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.5,
        cls_reg_pseudo_threshold=0.5,
        min_pseduo_box_size=0,
        use_hbb_rpn=True,
    # =======UBTeacherV2 Configs=======
        unsup_rcnn_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        unsup_rcnn_reg=dict(
            type='TSBetterPseudoLoss',
            ts_better=0.1, 
            t_cert=0.5,
            loss_weight=1.0,
        ),
        unsup_weight=0.5,
    )
)

optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(type="IterBasedRunner", max_iters=180000 * 4)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, type="SubModulesEvalHook", interval=10000, save_best='mAP', metric='mAP',
                  skip_at_resume_point=True)

auto_resume=True