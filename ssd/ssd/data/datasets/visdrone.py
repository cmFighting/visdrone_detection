import os
import torch.utils.data
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

from ssd.structures.container import Container


class VisDroneDataset(torch.utils.data.Dataset):
    # 问题是datasets是在哪里进行设置的，应该会有一个参数，这就很麻烦
    # 也就是11个类别再加上background
    # 这边基本完事了，把文件加载进来就可以，修改一下类名， 然后看下主启动函数那边
    class_names = ('ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van',
                   'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others')

    def __init__(self, data_dir, split, transform=None, target_transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
                我一开始是把occlusion为1的地方对应到voc中的diffcult，但是训练过程会出问题，然后我就默认diffcult为0了，应该是可以跑起来。还有一个问题，有个进程训练过程中就占用两张显卡，而且都占一半，这就很尴尬。
        """
        self.data_dir = data_dir
        # self.data_dir = "/home/chenmingsong/coding_data/coco/VOC_ROOT"
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        image_sets_file = os.path.join(self.data_dir, "ImageSets", "Main", "%s.txt" % self.split)
        # 涉及到的路径信息
        print('训练涉及到的路径信息')
        print('image_sets_file:{}'.format(image_sets_file))
        self.ids = VisDroneDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        return image, targets, index

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % image_id)
        # print('annotation_file1:{}'.format(annotation_file))
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            # is_difficult_str = obj.find('difficult').text
            is_difficult_str = '0'
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.data_dir, "Annotations", "%s.xml" % img_id)
        # print('annotation_file2:{}'.format(annotation_file))
        anno = ET.parse(annotation_file).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "JPEGImages", "%s.jpg" % image_id)
        # print('image_file:{}'.format(image_file))
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image
