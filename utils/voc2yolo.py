import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import sys

# sets=[('2018', 'train'), ('2018', 'val')]

# classes = ["a", "b", "c", "d"]

# soft link your VOC2018 under here
# root_dir = sys.argv[1]
classes = ['pedestrian', 'people', 'bicycle', 'car', 'van',
           'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']


# 直接使用
def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def single_voc2yolo(voc_folder, voc_name, yolo_path):
    voc_path = os.path.join(voc_folder, voc_name)
    yolo_name = voc_name.split(".")[0] + ".txt"
    yolo_path = os.path.join(yolo_path, yolo_name)
    with open(voc_path) as voc_f, open(yolo_path, "w") as yolo_f:
        tree = ET.parse(voc_f)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            yolo_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def transform_all(voc_folder, yolo_path):
    voc_names = os.listdir(voc_folder)
    for voc_name in voc_names:
        single_voc2yolo(voc_folder, voc_name, yolo_path)
        print("{} done!".format(voc_name))


if __name__ == '__main__':
    voc_folder = "E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/Annotations/"
    # voc_name = "0000001_02999_d_0000005.xml"
    yolo_path = "E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/annotations_yolo/"
    # single_voc2yolo(voc_folder, voc_name, yolo_path)
    transform_all(voc_folder, yolo_path)

# def convert_annotation(year, image_id):
#     in_file = open(os.path.join(root_dir, 'VOC%s/Annotations/%s.xml'%(year, image_id)))
#     out_file = open(os.path.join(root_dir, 'VOC%s/labels/%s.txt'%(year, image_id)), 'w')
#     tree=ET.parse(in_file)
#     root = tree.getroot()
#     size = root.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)
#
#     for obj in root.iter('object'):
#         difficult = obj.find('difficult').text
#         cls = obj.find('name').text
#         if cls not in classes or int(difficult)==1:
#             continue
#         cls_id = classes.index(cls)
#         xmlbox = obj.find('bndbox')
#         b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
#         bb = convert((w,h), b)
#         out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
#
# wd = getcwd()
#
# for year, image_set in sets:
#     labels_target = os.path.join(root_dir, 'VOC%s/labels/'%(year))
#     print('labels dir to save: {}'.format(labels_target))
#     if not os.path.exists(labels_target):
#         os.makedirs(labels_target)
#     image_ids = open(os.path.join(root_dir, 'VOC{}/ImageSets/Main/{}.txt'.format(year, image_set))).read().strip().split()
#     list_file = open(os.path.join(root_dir, '%s_%s.txt'%(year, image_set)), 'w')
#     for image_id in image_ids:
#         img_f = os.path.join(root_dir, 'VOC%s/JPEGImages/%s.jpg\n'%(year, image_id))
#         list_file.write(os.path.abspath(img_f))
#         convert_annotation(year, image_id)
#     list_file.close()

# print('done.')
