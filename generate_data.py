# -*- coding: utf-8 -*-
import random
import xml.etree.ElementTree as ET
import os
from collections import Counter
# from kmeans import YOLO_Kmeans
import math
import cv2
import numpy as np
f = open('object_classes_all.txt','r')
cla = f.readlines()
pre_classes =[c.strip().split(' ')[0] for c in cla]
classes_name = []
print(pre_classes)

train_percent = 0.85  # 20% 用来验证 80% 用来训练
txtsavepath = 'train_data'
datapath = r'H:\pyProject\detection\rbox\kaggle-airbus-ship-detection-challenge-master\dataset\data'
filedir = os.listdir(datapath)
picDir="H:\\pyProject\\detection\\rbox\\kaggle-airbus-ship-detection-challenge-master\\dataset\\data\\images\\"
ftest = open('train_data\\test_v2.txt', 'w')
ftrain = open('train_data\\train_v2.txt', 'w')
not_enough = []
#数据问题标记，0为正常，1为不正常
a = 0
train_ = 0
test_ = 0
for filedir in filedir:
    total_files = os.listdir(os.path.join(datapath, filedir))
    random.shuffle(total_files)
    total_xml = []
    total_image = []
    #找到所有xml，image
    for file in total_files:
        filetype = os.path.splitext(file)[1]
        if filetype == '.xml':
            total_xml.append(file)
            per = ET.parse(os.path.join(datapath, filedir,file))
            root = per.getroot()
            for Object in root.findall('object'):
                name = Object.find('name').text
                if name in pre_classes:
                    classes_name.append(name)
                else:
                    print(name,'in',file,'is not in defult classes!!!')
                    a +=1
                # if name == 'd':
                #     Object.find('name').text = 'D'
                #     per.write(os.path.join(datapath, filedir,file))
        else:
            total_image.append(file)
    #确认图片与xml对应
    for image in total_image:
        if image.split('.')[0] +'.xml' not in total_xml:
            print(image,'has no xml file!!!')
            a += 1

    for xml in total_xml:
        if xml.split('.')[0] +'.jpg' not in total_image:
            print(xml,'has no jpg file!!!')
            a+=1

    random.shuffle(total_xml)
    num = len(total_xml)
    list = range(num)
    trainNum = int(num * train_percent)
    train = random.sample(list, trainNum)
    for i in list:
        name = os.path.join(datapath, filedir, total_xml[i])+ '\n'
        if i in train:
            ftrain.write(name)
            train_ +=1
        else:
            ftest.write(name)
            test_ +=1
    print(filedir,int(len(total_files)/2),train_,test_)
ftrain.close()
ftest.close()
clas = Counter(classes_name)
clakey = [c for c in clas.keys()]
clav = [c for c in clas.values()]
classes_1 = []
not_enough_class = []
classes = []
for i in range(len(clakey)):
    if clav[i] <100:
        not_enough_class.append(clakey[i])
    else:
        classes_1.append(clakey[i])
for c in pre_classes:
    if c in classes_1:
        classes.append(c)
print(clas)
print('not_enough_file:',not_enough)
print('not_enough_class:',not_enough_class)
print('classes:',classes,len(classes))
print('total train:',train_)
print('total test:',test_)
if a == 0:
    print('data checked!')
else:
    print('data error!')

#输出模型需要的类别目录
co=open('train_data/train_classes.txt','w')
for c in classes:
        co.write(c)
        co.write('\n')
co.close()

sets=[ 'train_v2','test_v2','val']
#提取训练和测试数据
def convert_annotation(image_path, list_file,image_set):
    img_h=768.
    img_w=768.

    picOut="H:\\pyProject\\detection\\rbox\\kaggle-airbus-ship-detection-challenge-master\\dataset\\%s\\"%(image_set)
    in_file = open(image_path,encoding='UTF-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    filename = root.find('filename').text
    picture = cv2.imread(picDir + filename)
    height, width = picture.shape[:2]
    changeA = width / img_h
    rePic = cv2.resize(picture, (int(width/changeA), int(height/changeA)))
    reheight, rewidth = rePic.shape[:2]
    pic=np.zeros((int(img_h),int(img_w),3),dtype=np.uint8)
    startY=img_h/2-reheight/2
    pic[int(startY):int(startY)+reheight,:,:]=rePic
    imageSave =picOut+filename
    cv2.imwrite(imageSave, pic)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

        cx = (float(xmlbox.find('x').text))/changeA
        cy = (float(xmlbox.find('y').text))/changeA
        rw = (float(xmlbox.find('w').text))/changeA
        rh = (float(xmlbox.find('h').text))/changeA
        angle = float(xmlbox.find('angle').text)
        #角度转弧度
        angle=angle/180.0*math.pi
        if rh<rw:
            angle=angle+math.pi/2
            temp=rw
            rw=rh
            rh=temp
        if angle>0:
            while(angle>math.pi):
                angle-=math.pi
        else :
            while(angle<=0):
                angle+=math.pi

        # bow_x = b[0] + b[2] / 2 * math.cos(float(b[4]))
        # bow_y = b[1] - b[2] / 2 * math.sin(float(b[4]))
        #
        # tail_x = b[0] - b[2] / 2 * math.cos(float(b[4]))
        # tail_y = b[1] + b[2] / 2 * math.sin(float(b[4]))
        #
        # # print(bow_x,bow_y,tail_x,tail_y)
        #
        # x1 = round(bow_x + b[3] / 2 * math.sin(float(b[4])))
        # y1 = round(bow_y + b[3] / 2 * math.cos(float(b[4])))
        #
        # x2 = round(tail_x + b[3] / 2 * math.sin(float(b[4])))
        # y2 = round(tail_y + b[3] / 2 * math.cos(float(b[4])))
        #
        # x3 = round(tail_x - b[3] / 2 * math.sin(float(b[4])))
        # y3 = round(tail_y - b[3] / 2 * math.cos(float(b[4])))
        #
        # x4 = round(bow_x - b[3] / 2 * math.sin(float(b[4])))
        # y4 = round(bow_y - b[3] / 2 * math.cos(float(b[4])))
        #
        # # print(bow_x,bow_y,tail_x,tail_y)
        # print(x1, y1, x2, y2, x3, y3, x4, y4)
        # list_file.write(" " + "%d %d %d %d %d %d %d %d"%(x1,y1,x2,y2,x3,y3,x4,y4) + ' ' + str(cls_id))

        list_file.write(filename+",%f,%f,%f,%f,%f\n" % (cx,cy,rw,rh,angle))



for image_set in sets:
    image_ids = open('train_data\\%s.txt'%(image_set),encoding='UTF-8').read().strip().split()
    list_file = open('data_%s.txt'%(image_set), 'w',encoding='UTF-8')
    list_file.write('ImageID,x,y,height,width,rotate\n')
    for image_id in image_ids:
        image_path = image_id.split('.')[0] +'.jpg'
        convert_annotation(image_id, list_file,image_set)
        # try:
        #     convert_annotation(image_xml, list_file)
        # except:
        #     print(image_xml,'is broken')
    list_file.close()

cluster_number = 9
filename = "data_train.txt"
# kmeans = YOLO_Kmeans(cluster_number, filename)
# kmeans.txt2clusters()
#
