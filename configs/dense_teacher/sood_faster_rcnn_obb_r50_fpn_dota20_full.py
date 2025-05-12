_base_ = "sood_faster_rcnn_obb_r50_fpn_dota20_partial.py"

data_root='data/DOTA/DOTA-v2.0_all/'

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="DOTADataset",
            ann_file=data_root+"trainval/annfiles/",
            img_prefix=data_root+"trainval/images/",
        ),
        unsup=dict(
            type="DOTADataset",
            ann_file=data_root+"test_challenge/annfiles/",
            img_prefix=data_root+"test_challenge/images/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[2, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        use_teacher_proposal=True,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_threshold=0.5,
        min_pseduo_box_size=0,
        unsup_weight=0.2,
        use_hbb_rpn=True,
        semi_loss=dict(
            type="RotatedTSOTDTLoss", 
            cls_channels="${num_classes}", 
            dynamic_raw_type="50ang",
            aux_loss="ot_loss_norm", 
            aux_loss_cfg=dict(clamp_ot=True, loss_weight=0.1),
        )
    ),
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, type="SubModulesEvalHook", interval=10000, save_best='mAP', metric='mAP',
                  skip_at_resume_point=True)

auto_resume=True