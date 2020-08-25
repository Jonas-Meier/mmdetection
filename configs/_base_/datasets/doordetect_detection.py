dataset_type = 'DoorDetectDataset'  # string has to be conform with python class names!
data_root = 'data/door_detect/'  # shall end with a '/'
# img_norm seems to be default for most datasets
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),  # TODO: does this automatically load the images randomly?
    # LoadImageFromFile creates a CustomDataset, train calls mmdet/datasets/builder.py:build_dataloader() which per
    # default shuffles the dataset each epoch
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 768), keep_ratio=True),  # most images are around 1024x768px
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,  # Batch size of a single GPU (bs=1)
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        type='RepeatDataset',
        times=12,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/annotations_train.json',
            img_prefix=data_root + 'train/pos',
            pipeline=train_pipeline
        )),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/annotations_val.json',
        img_prefix=data_root + 'val/pos',
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/annotations_val.json',
        img_prefix=data_root + 'val/pos',
        pipeline=test_pipeline
    ))
evaluation = dict(interval=1, metric='bbox')  # evaluation (with validation set) interval in epochs,
# see mmdet/core/evaluation/eval_hooks.py
