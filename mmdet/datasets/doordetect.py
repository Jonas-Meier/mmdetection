
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DoorDetectDataset(CocoDataset):
    # we inherit from CocoDataset since we use COCO-json annotation format. We only need to change the CLASSES value,
    # since coco.py should already have the necessary tools for training, reading annotations and images, as well as
    # evaluating the detection task

    CLASSES = ('door', 'handle', 'cabinet door', 'refrigerator door')  # we have only these four classes in this dataset
