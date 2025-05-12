_base_ = "../_base_/semi_faster_rcnn_obb_r50_fpn_1x_dota20.py"

data_root='data/DOTA/DOTA-v2.0_all/'

model = dict(
    type='MSInputFasterRCNNOBB',
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        norm_eval=True,
    ),
)
# test_cfg=dict(rcnn=dict(score_thr=1e-3))

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
    type="MixTeacher",
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
        use_hbb_rpn=True,
    ),
    test_cfg=dict(inference_on="student")
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="WeightSummary"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type="GetCurrentIter"),
]

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, type="MSISubModulesEvalHook", interval=10000, evaluated_modules=['teacher'],
                  test_scales=['fr', 'hr'], save_best='mAP', metric='mAP', skip_at_resume_point=True)

auto_resume=True