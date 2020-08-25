import argparse
import glob
import os

import mmcv
import numpy as np

# Note: code is based on 'mmdetection/tools/convert_datasets/cityscapes.py' but is modified for ObjectDetection task
# (cityscape is for Semantic Segmentation and Instance Segmentation tasks).
# cityscapes.py converts annotations into COCO format (.json files with certain keys)
# pascal_voc.py converts .xml files (in Pascal-VOC format) into mmdetection's 'middle format' (as pickle (.pkl) file)
# More information at: https://mmdetection.readthedocs.io/en/latest/tutorials/new_dataset.html
# Another difference is the organization of images and annotations for train and validation sets:
# For cityscape it looks like (leftImg8bit = images, gtFine = annotations):
# |-- leftImg8bit
#   |-- train
#   |-- val
# |-- gtFine
#   |-- train
#   |-- val
# For doordetect-dataset it looks like (pos = images):
# |-- train
#   |-- pos
#   |-- annotations
# |-- validation
#   |-- pos
#   |-- annotations

ID_TO_CLASS = {0: 'door', 1: 'handle', 2: 'cabinet door', 3: 'refrigerator door'}
CLASS_TO_ID = {'door': 0, 'handle': 1, 'cabinet door': 2, 'refrigerator door': 3}


def collect_files(img_dir, anno_dir):
    """
    We do not check whether the corresponding annotation to an image even exist!
    :param img_dir: directory where all images reside
    :param anno_dir: directory where the corresponding annotations reside
    :return: returns a list containing tuples (pairs) of image path and annotation path
    """
    files = []
    # note: os.listdir already gives the relative file names without the whole path before it (exactly as glob.glob would do)
    for img_name in os.listdir(img_dir):    # we could also use glob.glob, but we know we will take every picture in the given directory!
        suffix = "." + img_name.split(".")[-1]  # image suffixes are often different (.jpg, .JPG, .jpeg, .png, etc.)
        anno_name = img_name.replace(suffix, ".txt")  # adjust the name of the corresponding annotation file (not checked!)
        files.append((
            os.path.join(img_dir, img_name),
            os.path.join(anno_dir, anno_name)
        ))
    return files


def collect_annotations(files):
    print('Loading annotation images')
    images = mmcv.track_progress(load_img_info, files)  # executes load_img_info for each file in files and collects results
    return images

# a mix of
# cityscapes.py (but no handling for masks as this dataset is for detection only) and
# pascal_voc.py (but conversion to .json instead of pickle (.pkl) and no usage of np.array for bboxes and labels!)
def load_img_info(file):
    """
    Loads image information(height, width, name) and annotations (bbox, class_name) from a SINGLE
    file (=image file + corresponding annotation file)
    :param file:
    :return:
    """
    img_file, anno_file = file
    img = mmcv.imread(img_file, 'unchanged')
    annotations_list = mmcv.list_from_file(anno_file)  # each element in the list is a line of the text file
    # collect necessary information of the image
    img_name = os.path.basename(img_file)
    img_height = img.shape[0]
    img_width = img.shape[1]
    # collect information about all annotations
    annos = []
    for annotation_line in annotations_list:  # annos inside the list are each like <class-id> <x> <y> <width> <height>
        id, rel_center_x, rel_center_y, rel_width, rel_height = annotation_line.split(" ")
        assert 0 <= int(id) <= 3, "Found invalid object id: {}".format(id)  # we have four classes
        # print(id, rel_center_x, rel_center_y, rel_width, rel_height)
        # we have yolo format of bounding boxes, convert them to coco format
        x, y, width, height = yolo2coco_bbox(float(rel_center_x), float(rel_center_y), float(rel_width), float(rel_height), img_width, img_height)
        bbox = [
            int(x),
            int(y),
            int(width),
            int(height)
        ]
        anno = dict(
            bbox=bbox,
            category_id=int(id)
        )
        annos.append(anno)
    img_info = dict(
        file_name=img_name,
        height=img_height,
        width=img_width,
        annos=annos
    )
    return img_info


#  TODO: change name to cvt_coco_annotations_to_json (pascal voc converter converts xml to pickle '.pkl')
def cvt_annotations(image_infos, out_json_name):
    """
    COCO annotations for object detection are assumed to be in the following format:
    'images': [
        {
            'file_name': 'xxx',     # string
            'height': xxx,          # int
            'width': xxx,           # int
            'id': xxx               # int
        },
        ...
    ],

    'annotations': [
        {
            'segmentation': [[xxx.xx, ..., xxx.xx]] # List(List(floats)) or empty list []
            'area': xxx.xx                          # float (area of segmentation, bbox otherwise)
            'iscrowd': 0                            # 0: single object (default), 1: group of objects
            'image_id': xxx,                        # int
            'bbox': [x,y,width,height],             # List(floats)
            'category_id': xxx,                     # int
            'id': xxx                               # int
        },
        ...
    ]

    'categories': [
        {'id': 0, 'name': 'xxx'}, ...
    ]
    :param image_infos:
    :param out_json_name:
    :return:
    """
    out_json = dict()
    img_id = 0
    ann_id = 0
    out_json['images'] = []
    out_json['annotations'] = []
    out_json['categories'] = []
    # fill 'images' and 'annotations' key fo json file
    for image_info in image_infos:
        image_info['id'] = img_id
        annos = image_info.pop('annos')  # annos was only temporarily saved in image infos!
        out_json['images'].append(image_info)
        for anno in annos:
            anno['segmentation'] = []  # no segmentation annotations, we set it to be empty
            anno['iscrowd'] = 0  # only single objects since we have detection task, we should set it nonetheless
            anno['area'] = anno['bbox'][2] * anno['bbox'][3]  # we set area to be the area of the bbox
            anno['image_id'] = img_id
            anno['id'] = ann_id
            out_json['annotations'].append(anno)
            ann_id += 1
        img_id += 1
    # fill 'categories' key for json file
    for label, name in ID_TO_CLASS.items():
        cat = dict(id=label, name=name)
        out_json['categories'].append(cat)

    if len(out_json['annotations']) == 0:
        out_json.pop('annotations')

    mmcv.dump(out_json, out_json_name)
    return out_json


def yolo2coco_bbox(rel_center_x, rel_center_y, rel_width, rel_height, img_width, img_height):
    """
    Converts a yolo bounding box to a bounding box in coco format
    :param rel_center_x: x-coordinate of the center of the bounding box, relative to image width
    :param rel_center_y: y-coordinate  of the center of the bounding box, relative to image height
    :param rel_width: width of the bounding box, relative to image width
    :param rel_height: height of the bounding box, relative to image height
    :param img_width: width of the image
    :param img_height: height of the image
    :return: corresponding bounding box in coco format:
         x,y: top left corner of the bounding box
         width, height: absolute width and height of the bounding box
    """
    assert rel_center_x <= 1 and rel_center_y <= 1 and rel_width <= 1 and rel_height <= 1, \
        "Invalid yolo bounding box: ({},{},{},{})".format(rel_center_x, rel_center_y, rel_width, rel_height)
    width = rel_width * img_width
    height = rel_height * img_height
    x = rel_center_x * img_width - width / 2
    y = rel_center_y * img_height - height / 2
    return x, y, width, height


def main():
    """
    Collects files: collect image directories and corresponding annotations
    Collects annotations: (reads and reformats annotation files)
    Converts annotations to json and saves them
    """
    doordetect_path = "/home/jonas/data/door_detect" # TODO: maybe change this to an argument in terminal?
    out_dir = "/home/jonas/data/door_detect/annotations" # or coco_json_annotations instead of annotations?
    mmcv.mkdir_or_exist(out_dir)

    img_dir_name = "pos"
    anno_dir_name = "annotations"
    set_name = dict(
        train='annotations_train.json',
        validation='annotations_validation.json'
    )

    for split, json_name in set_name.items():
        print(f'Converting {split} into {json_name}')
        with mmcv.Timer(print_tmpl='It took {}s to convert DoorDetect-Dataset annotation'):
            files = collect_files(
                os.path.join(doordetect_path, split, img_dir_name),
                os.path.join(doordetect_path, split, anno_dir_name)
            )
            image_infos = collect_annotations(files)
            cvt_annotations(image_infos, os.path.join(out_dir, json_name))


if __name__ == '__main__':
    main()
