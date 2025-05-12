_base_ = "mixT_faster_rcnn_obb_r50_fpn_dota20_partial.py"

num_classes=17
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes="${num_classes}"
        ),
    ),
)

data_root='data/dota_2.0_gsd/'
classes="DOTA_GSD_2025_03"
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="DOTADataset",
            classes="${classes}",
            ann_file=data_root+"trainval_split/annfiles/",
            img_prefix=data_root+"trainval_split/images/",
        ),
        unsup=dict(
            type="DOTADataset",
            classes="${classes}",
            ann_file=data_root+"test_challenge_split/annfiles/",
            img_prefix=data_root+"test_challenge_split/images/",
        ),
    ),
    val=dict(
        type='DOTADataset',
        classes="${classes}",
        samples_per_gpu="${val_samples_per_gpu}",
        ann_file=data_root + "val/annfiles/",
        img_prefix=data_root + "val/images/",
    ),
    test=dict(
        type='DOTADataset',
        classes="${classes}",
        save_ori=True,
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
        use_teacher_proposal=False,
        unsup_rcnn_cls = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        unsup_weight=0.5,
        no_unlabeled_gt=True
    )
)

optimizer = dict(type="SGD", lr=0.005, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(type="IterBasedRunner", max_iters=180000 * 4)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, type="MSISubModulesEvalHook", interval=10000, evaluated_modules=['teacher'],
                  test_scales=['fr', 'hr'], save_best='mAP', metric='mAP', skip_at_resume_point=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="SemiTextLoggerHook", by_epoch=False, ignore_keys=["gsd", "pseudo"]),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="SRGT",
                # entity='gtawessd-university-of-chinese-academy-of-sciences',  # Your wandb team/username
                # id='t4qggern',  # The run ID to resume (must match the remote run)
                # resume='must',  # Required to resume logging to the same run
                name="${run_name}",
                config=dict(
                    work_dirs="${work_dir}",
                    total_step="${runner.max_iters}",
                ),
            ),
            by_epoch=False,
        )
    ],
)

auto_resume=True