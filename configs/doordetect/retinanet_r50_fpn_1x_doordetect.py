_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/doordetect_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained='./models/imagenet/resnet50-19c8e357.pth',  # equal to 'torchvision://resnet50' other: None
    bbox_head=dict(
        num_classes=4
    )
)

# optimizer
# TODO: adjust lr (0.01, as in coco, corresponding to what batch size? 2x8=16.
#  -> Then why 0.01 for bs=16 instead of 0.02?)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

work_dir = './workdirs/doordetect_v4_1'  # Directory to save the model checkpoints and logs for the current experiments.
load_from = None  # './models/coco/......'
