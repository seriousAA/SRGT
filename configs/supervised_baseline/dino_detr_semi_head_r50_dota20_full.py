_base_ = "dino_detr_semi_head_r50_dota20_partial.py"

data_root='data/DOTA/DOTA-v2.0_all/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="DOTADataset",
        task='Task1',
        ann_file=data_root+"trainval/annfiles/",
        img_prefix=data_root+"trainval/images/",
    ),
)

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(step=[160000])
runner = dict(type="IterBasedRunner", max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=20000, max_keep_ckpts=10)
evaluation = dict(_delete_=True, interval=20000, save_best='mAP', metric='mAP', skip_at_resume_point=True)

# fp16 = dict(loss_scale="dynamic")

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
    ],
)

auto_resume=True
