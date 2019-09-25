#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/3 18:35
# @Author  : LH
# @Site    : 
# @File    : gen_xml.py
# @Software: PyCharm
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
def gen_xml(save_path,class_path,out_boxes, out_scores, out_classes,score=0):
    classes_path = class_path
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    ENCODE_METHOD = 'utf8'
    # 设置文件头
    top = Element('annotation')
    # top.set('verified', 'no')

    folder = SubElement(top, 'folder')
    folder.text = 'folder'

    filename = SubElement(top, 'filename')
    filename.text = 'text'

    localImgPath = SubElement(top, 'path')
    localImgPath.text = 'path'

    source = SubElement(top, 'source')
    database = SubElement(source, 'database')
    database.text = 'database'

    size_part = SubElement(top, 'size')
    width = SubElement(size_part, 'width')
    height = SubElement(size_part, 'height')
    depth = SubElement(size_part, 'depth')
    width.text = str(832)
    height.text = str(832)
    depth.text = str(3)

    segmented = SubElement(top, 'segmented')
    segmented.text = '0'
    out_file = codecs.open(save_path, 'w', encoding=ENCODE_METHOD)
    # 写入object内容

    for i in range(len(out_scores)):
        if out_scores[i]>score:  # 设置显示阈值
            head, left, bottom, right = out_boxes[i]

            node_object = SubElement(top, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = class_names[out_classes[i]]               # 写入类别名称

            node_pose = SubElement(node_object, 'pose')
            node_pose.text = 'Unspecified'

            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = '0'

            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'

            # 写入box信息
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(int(left))
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(int(head))
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(int(right))
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(int(bottom))
    rough_string = ElementTree.tostring(top, 'utf8')
    root = etree.fromstring(rough_string)
    tree = etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())

    out_file.write(tree.decode('utf8'))
    out_file.close()

