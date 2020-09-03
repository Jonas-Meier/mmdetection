_base_ = [
    '../_base_/models/ssd300.py',
    '../_base_/datasets/doordetect_detection.py',
    '../_base_/default_runtime.py'
]

# we have to rewrite the bbox_head, especially num_classes as it differs from the settings of
# '../_base_/models/ssd300.py' which are based on the COCO dataset (with 80 classes)
# parameters for anchor_generator are taken from 'ssd300_voc0712.py'
model = dict(
    pretrained='./models/imagenet/vgg16-397923af.pth',  # alternative is 'vgg19-dcbb9e9d.pth' in same directory
    bbox_head=dict(
        num_classes=4, anchor_generator=dict(basesize_ratio_range=(0.2,
                                                                   0.9))))

# dataset settings
dataset_type = 'DoorDetectDataset'
data_root = 'data/door_detect/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [  # same as 'ssd300_voc0712.py' and 'ssd300_coco.py'
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
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),  # TODO: why False?
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [  # same as 'ssd300_voc0712.py' and 'ssd300_coco.py'
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),  # TODO: why False?
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# Here, 'ssd300_voc0712.py' and 'ssd300_coco.py' differ much, especially inside 'train' inside 'data' key
# -> 'ssd300_coco.py' sets _delete_=True inside train dict because in 'coco_detection.py' they do not yet use
# -> RepeatDataset, therefore they have to completely overwrite the train dict. Whereas 'ssd300_voc0712.py' already uses
# -> a RepeatDataset inside it's base config 'voc0712.py' and therefore only has to make some small modifications
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type='RepeatDataset', times=2, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer
# TODO: is this lr for bs=64(8GPUs x 8img_per_GPU)?
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 20])  # TODO: Adjust to new 'total_epochs', maybe [10,13]?
checkpoint_config = dict(interval=1)
# runtime settings
total_epochs = 24  # TODO: set to 16
