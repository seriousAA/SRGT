_base_ = "../_base_/srgt_faster_rcnn_obb_r50_fpn_1x_dota20.py"

num_classes=17
model = dict(
    type='MSInputFasterRCNNOBB',
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes="${num_classes}"
        ),
    ),
)
# test_cfg=dict(rcnn=dict(score_thr=1e-3))

data_root='data/dota_2.0_gsd/'
classes="DOTA_GSD_2025_03"
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="DOTADataset",
            classes="${classes}",
            # gsd_csv=data_root+"trainval_split/gsd_meta.csv",
            ann_file=data_root+"trainval_split/semi_supervised/${fold}_${percent}/labeled/",
            img_prefix=data_root+"trainval_split/images/",
        ),
        unsup=dict(
            type="DOTADataset",
            classes="${classes}",
            # gsd_csv=data_root+"trainval_split/gsd_meta.csv",
            ann_file=data_root+"trainval_split/semi_supervised/${fold}_${percent}/unlabeled/",
            img_prefix=data_root+"trainval_split/images/",
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
            sample_ratio=[2, 1],
        )
    ),
)

semi_wrapper = dict(
    type="MixSRGT",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.5,
        cls_pseudo_threshold=0.9,
        mine_pseudo_threshold=0.7,
        diff_score_threshold=0.1,
        reg_pseudo_threshold=0.02,
        jitter_times=10,
        jitter_scale=0.06,
        min_pseduo_box_size=0,
        unsup_weight=0.5,
        unsup_rcnn_cls = dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        use_hbb_rpn=True,
    ),
    test_cfg=dict(inference_on="student")
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=20)
evaluation = dict(_delete_=True, type="MSISubModulesEvalHook", interval=10000, evaluated_modules=['teacher'],
                  test_scales=['fr', 'hr'], save_best='mAP', metric='mAP', skip_at_resume_point=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="SemiTextLoggerHook", by_epoch=False, ignore_keys=["gsd", "pseudo"]),
        # dict(
        #     type="WandbLoggerHook",
        #     init_kwargs=dict(
        #         project="SRGT",
        #         # entity='gtawessd-university-of-chinese-academy-of-sciences',  # Your wandb team/username
        #         # id='t4qggern',  # The run ID to resume (must match the remote run)
        #         # resume='must',  # Required to resume logging to the same run
        #         name="${run_name}",
        #         config=dict(
        #             work_dirs="${work_dir}",
        #             total_step="${runner.max_iters}",
        #         ),
        #     ),
        #     by_epoch=False,
        # )
    ],
)

auto_resume=True