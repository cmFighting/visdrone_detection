'''
用于将比赛提供的文件转化为voc形式的xml文件
主要涉及到的内容有坐标和类别，其他的内容相对不是很重要
源文件：<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
1. ？？？ score表示对象实例预测边界值的置信度，1表示在评估中考虑边界框，0表示在评估中忽略边界框, 这个score的存在作用待定score为0的区域类别也为0，佛了，暂时不使用
2. category表示对象的类别 忽略的区域 （0）、行人 （1）、人 （2）、自行车 （3）、汽车 （4）、面包车 （5）、卡车 （6）、三轮车 （7）、遮阳篷三轮车 （8）、公共汽车 （9）、电机 （10）、其他人 （11)
忽略的区域 （0）、行人 （1）、人 （2）、自行车 （3）、汽车 （4）、面包车 （5）、卡车 （6）、三轮车 （7）、遮阳篷三轮车 （8）、公共汽车 （9）、电机 （10）、其他人 （11）
ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11)
3. truncation表示截断，无截断为0，部分截断为1
4. occlusion表示遮挡，0表示无遮挡，1表示部分遮挡，2表示全部遮挡
目标文件：name、pose、truncated、difficult、左上角坐标和右下角坐标。文件名还有文件路径
1. name 就是具体的离别，对应category
2. truncated表示截断，默认是0，这个是txt文件中的truncation是一一对应的
3. ？？？ difficult表示是否很难识别堆成，默认为0，然后遮挡不遮挡来进行转化
4. pose表示视角，默认为0，这里直接采用默认值即可
tips yolo的标注文件形式为 类别 中心点x，中心点y，宽w和高h
总之，转化过来之后，原文的score和occlusion没有用到，对于转化过后的xml文件来说，坐标点要进行变换，然后截断采用txt中的截断，其余均采用默认值
参考代码1，直接采用字符串进行替换，https://blog.csdn.net/angelbeats11/article/details/88427359
参考代码2，txt转化为xml，然后xml再转化为yolo https://blog.csdn.net/qq_29762941/article/details/80797790
参考代码3：将xml转化为txt https://blog.csdn.net/WK785456510/article/details/81565637
*** 可以使用！ 参考代码4 将txt转化为xml，是特殊的txt  https://blog.csdn.net/jocelyn870/article/details/81210375?tdsourcetag=s_pcqq_aiomsg
*** 可以参考 是个大佬 https://blog.csdn.net/weixin_38106878/article/details/90142445
'''
import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from xml.dom.minidom import *
classes = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def write_xml(img_name, width, height, object_dicts, save_path, folder='DET2019'):
    '''
    object_dict = {'name': classes[int(object_category)],
                            'truncated': int(truncation),
                            'difficult': int(occlusion),
                            'xmin':int(bbox_left),
                            'ymin':int(bbox_top),
                            'xmax':int(bbox_left) + int(bbox_width),
                            'ymax':int(bbox_top) + int(bbox_height)
                            }
    '''
    doc = Document
    root = ET.Element('Annotation')
    ET.SubElement(root, 'folder').text = folder
    ET.SubElement(root, 'filename').text = img_name
    size_node = ET.SubElement(root, 'size')
    ET.SubElement(size_node, 'width').text = str(width)
    ET.SubElement(size_node, 'height').text = str(height)
    ET.SubElement(size_node, 'depth').text = '3'
    for object_dict in object_dicts:
        object_node = ET.SubElement(root, 'object')
        ET.SubElement(object_node, 'name').text = object_dict['name']
        ET.SubElement(object_node, 'pose').text = 'Unspecified'
        ET.SubElement(object_node, 'truncated').text = str(object_dict['truncated'])
        ET.SubElement(object_node, 'difficult').text = str(object_dict['difficult'])
        bndbox_node = ET.SubElement(object_node, 'bndbox')
        ET.SubElement(bndbox_node, 'xmin').text = str(object_dict['xmin'])
        ET.SubElement(bndbox_node, 'ymin').text = str(object_dict['ymin'])
        ET.SubElement(bndbox_node, 'xmax').text = str(object_dict['xmax'])
        ET.SubElement(bndbox_node, 'ymax').text = str(object_dict['ymax'])

    pretty_xml(root, '\t', '\n')
    tree = ET.ElementTree(root)
    tree.write(save_path,encoding='utf-8')


def go():
    # TODO 修改修改这里的路径
    # train
    # img_folder_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-train\\images\\'
    # txt_annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-train\\annotations_txt\\'
    # annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-train\\annotations_xml\\'

    # val
    # img_folder_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-val\\images\\'
    # txt_annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-val\\annotations_txt\\'
    # annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-val\\annotations_xml\\'

    # test
    img_folder_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-test_dev\\images\\'
    txt_annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-test_dev\\annotations_txt\\'
    annotations_path = 'F:\\datas\\VOC\\DET\\VisDrone2019-DET-test_dev\\annotations_xml\\'

    txt_names = os.listdir(txt_annotations_path)
    for txt_name in txt_names:  # 批量读.txt文件
        # 读取图片基本信息
        img_name = txt_name.strip('.txt') + '.jpg'
        img_path = img_folder_path + img_name
        img = np.array(Image.open(img_path))
        size_height, size_width = img.shape[0], img.shape[1]  # img.shape[0]是图片的高度720

        with open(txt_annotations_path + txt_name, 'r') as f:
            # img_id = os.path.splitext(label)[0]
            # contents = f.readlines()splitlines()
            contents = f.read().splitlines()

            # 读取目标
            object_dicts = []
            for content_str in contents:
                content = content_str.split(',')
                # img.shape[1]是图片的宽度720
                # content = content.strip('\n').split()
                # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                # print(content)
                bbox_left = content[0]
                bbox_top = content[1]
                bbox_width = content[2]
                bbox_height = content[3]
                score = content[4]
                object_category = content[5]
                truncation = content[6]
                occlusion = content[7]
                object_dict = {'name': classes[int(object_category)],
                            'truncated': int(truncation),
                            'difficult': int(occlusion),
                            'xmin':int(bbox_left),
                            'ymin':int(bbox_top),
                            'xmax':int(bbox_left) + int(bbox_width),
                            'ymax':int(bbox_top) + int(bbox_height)
                            }
                '''
                'xmin': x + 1 - w / 2,
                'ymin': y + 1 - h / 2,
                'xmax': x + 1 + w / 2,
                'ymax': y + 1 + h / 2
                '''
                # 转移为字典数组的形式
                object_dicts.append(object_dict)

            write_xml(img_name, size_width, size_height, object_dicts, annotations_path + txt_name.strip('.txt') + '.xml',)


if __name__ == '__main__':
    # go(
    a = np.array([[],[]])
    # si = torch.randn(4, 5)
    a = torch.tensor(a)
    # 说明a是一个空值
    a.max(1)