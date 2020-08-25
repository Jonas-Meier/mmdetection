
from .builder import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class DoorDetectDataset(CocoDataset):

    CLASSES = ('door', 'handle', 'cabinet door', 'refrigerator door')
