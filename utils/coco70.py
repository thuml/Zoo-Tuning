import os
from typing import Optional
from .imagelist import ImageList


class COCO70(ImageList):
    """COCO-70 dataset is a large-scale classification dataset (1000 images per class) created from
    `COCO <https://cocodataset.org/>`_ Dataset.
    It is used to explore the effect of fine-tuning with a large amount of data.
    """

    image_list = {
        "train": "train.txt",
        "test": "test.txt",
    }
    CLASSES =['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
              'boat', 'traffic_light', 'fire_hydrant', 'stop_sign', 'bench', 'bird', 'cat', 'dog',
              'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
              'handbag', 'tie', 'suitcase', 'skis', 'kite', 'baseball_bat', 'skateboard', 'surfboard',
              'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
              'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
              'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'teddy_bear']

    def __init__(self, root: str, split: str, **kwargs):

        if split == 'train':
            list_name = 'train'
            assert list_name in self.image_list
            data_list_file = os.path.join(root, self.image_list[list_name])
        else:
            data_list_file = os.path.join(root, self.image_list['test'])

        super(COCO70, self).__init__(root, COCO70.CLASSES, data_list_file=data_list_file, **kwargs)
