_base_ = "fcos_obb_r50_fpn_dota20_partial.py"

num_classes=17
model = dict(
    bbox_head=dict(
        num_classes="${num_classes}"
    ),
)

data_root='data/dota_2.0_gsd/'
classes="DOTA_GSD_2025_03"
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=3,
    train=dict(
        type="DOTADataset",
        classes="${classes}",
        task='Task1',
        ann_file=data_root+"trainval_split/annfiles/",
        img_prefix=data_root+"trainval_split/images/",
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
)

optimizer = dict(_delete_=True, type="SGD", lr=0.0025 * 2, momentum=0.9, weight_decay=0.0001)
lr_config = dict(step=[120000 * 2, 160000 * 2])
runner = dict(type="IterBasedRunner", max_iters=180000 * 2)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, interval=10000, save_best='mAP', metric='mAP', skip_at_resume_point=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

auto_resume=True
