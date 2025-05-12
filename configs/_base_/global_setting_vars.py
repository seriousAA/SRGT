num_classes=18

fold = 1
percent = 1

val_samples_per_gpu=2

# fp16 = dict(loss_scale="dynamic")
auto_resume=False
find_unused_parameters=False

backend="disk"
resume_optimizer=True