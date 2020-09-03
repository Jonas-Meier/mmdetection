_base_ = 'ssd300_doordetect.py'
input_size = 512
model = dict(
    pretrained=None,  # './models/imagenet/vgg16-397923af.pth',  # alternative is 'vgg19-dcbb9e9d.pth' in same directory
    backbone=dict(input_size=input_size),
    bbox_head=dict(
        in_channels=(512, 1024, 512, 256, 256, 256, 256),
        anchor_generator=dict(
            input_size=input_size,
            strides=[8, 16, 32, 64, 128, 256, 512],
            basesize_ratio_range=(0.15, 0.9),
            ratios=([2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]))))
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [  # only image scale changes compared to base config 'ssd300_doordetect.py'
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [  # only img_scale changes compared to base config 'ssd300_doordetect.py'
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# Some checkpoint and logging settings as well as working directory path
checkpoint_config = dict(  # Config to set the checkpoint hook, see: 'mmcv/runner/hooks/checkpoint.py'
    interval=1)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.
log_level = 'INFO'  # The level of logging.
work_dir = './workdirs/doordetect_v2_2'  # Directory to save the model checkpoints and logs for the current experiments.
load_from = './models/coco/ssd512_coco_20200308-038c5591.pth'
