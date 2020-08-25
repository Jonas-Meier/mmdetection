_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/doordetect_detection.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained=None,  # ImageNet pretrained backbone to be loaded. Not necessary since we use pretrained coco model
        # (e.g. 'torchvision://resnet50' or 'torchvision://resnet101')
    # backbone=dict(),
    # neck=dict(),
    # rpn_head=dict(),
    roi_head=dict(  # we only need to modify the roi_head in this case (for now)
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,  # we only have four classes
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)))  # TODO: maybe use SmoothL1Loss with beta=1.0 ?
)
# optimizer. lr=0.02 is for bs=16! More types at 'mmdet/core/optimizer/default_constructor.py'
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)  # 0.00125=0.02/16: we have bs=1, not 16!
optimizer_config = dict(grad_clip=None)  # Most methods do not use gradient clipping
# learning policy
lr_config = dict(  # Learning rate scheduler config used to register LrUpdater hook
    policy='step',  # The policy of scheduler, more supported, see: 'mmcv/runner/hooks/lr_updater.py'
    warmup='linear',  # The warmup policy, also support `exp` and `constant`
    warmup_iters=500,  # The number of iterations for warmup
    warmup_ratio=0.001,  # The ratio of the starting learning rate used for warmup
    step=[8, 11])  # Steps to decay the learning rate
total_epochs = 12  # Total epochs to train the model TODO: 1 epoch = 1 whole dataset? epochs == times dataset repeated?
checkpoint_config = dict(  # Config to set the checkpoint hook, see: 'mmcv/runner/hooks/checkpoint.py'
    interval=1)  # The save interval is 1
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook')
    ])  # The logger used to record the training process.
dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
log_level = 'INFO'  # The level of logging.

# load models as a pre-trained model from a given path. This will not resume training.
# Here, a COCO pretrained model.
# load_from = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa
load_from = './models/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # saved locally
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved.
# workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow
# named 'train' is executed once. The workflow trains the model by 12 epochs according to the total_epochs.
work_dir = './workdirs/doordetect_v1'  # Directory to save the model checkpoints and logs for the current experiments.
