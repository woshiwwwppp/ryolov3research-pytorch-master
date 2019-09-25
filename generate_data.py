# -*- coding: utf-8 -*-
import random
import xml.etree.ElementTree as ET
import os
from collections import Counter
from kmeans import YOLO_Kmeans


f = open('object_classes_all.txt','r')
cla = f.readlines()
pre_classes =[c.strip().split(' ')[0] for c in cla]
classes_name = []
print(pre_classes)
trainval_percent = 0.15  # 20% 用来验证 80% 用来训练
test_val_percent = 0.2  # 30%用来测试 70%用来验证
txtsavepath = 'train_data'
datapath = "/media/wp/windows/data/insulator"
filedir = os.listdir(datapath)
ftrainval = open('train_data/trainval.txt', 'w')
ftest = open('train_data/test.txt', 'w')
ftrain = open('train_data/train.txt', 'w')
fval = open('train_data/val.txt', 'w')
not_enough = []
total_train_1 = 0
total_train_val = 0
total_test_1 = 0
total_test_val = 0
a = 0
for filedir in filedir:
    total_files = os.listdir(os.path.join(datapath, filedir))
    random.shuffle(total_files)
    total_xml = []
    total_image = []
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
    for image in total_image:
        if image.split('.')[0] +'.xml' not in total_xml:
            print(image,'has no xml file!!!')
            a += 1

    for xml in total_xml:
        if xml.split('.')[0] +'.jpg' not in total_image:
            print(xml,'has no jpg file!!!')
            a+=1
    train_1 = 0
    train_val = 0
    test_1 = 0
    test_val = 0

    random.shuffle(total_xml)
    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * test_val_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    for i in list:
        name = os.path.join(datapath, filedir, total_xml[i])+ '\n'
        if i in trainval:
            ftrainval.write(name)
            train_val +=1
            total_train_val +=1
            if i in train:
                ftest.write(name)
                test_1 +=1
                total_test_1 +=1
            else:
                fval.write(name)
                test_val +=1
                total_test_val +=1
        else:
            ftrain.write(name)
            train_1 +=1
            total_train_1 +=1
    print(filedir,int(len(total_files)/2),train_1,train_val,test_1,test_val)
ftrainval.close()
ftrain.close()
fval.close()
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
print('total train:',total_train_1)
print('total val:',total_train_val)
print('total test:',total_test_1)
print('total test_val:',total_test_val)
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

sets=[ 'train','val','test']
#提取训练和测试数据
def convert_annotation(image_path, list_file):
    in_file = open(image_path,encoding='UTF-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue

        cls_id = classes.index(cls)
        xmlbox = obj.find('robndbox')
        b = (int(xmlbox.find('cx').text), int(xmlbox.find('cy').text), int(xmlbox.find('w').text), int(xmlbox.find('h').text),float(xmlbox.find('angle').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

for image_set in sets:
    image_ids = open('train_data/%s.txt'%(image_set),encoding='UTF-8').read().strip().split()
   # image_ids = open('train_data\\%s.txt' % (image_set), encoding='UTF-8').read().split()
    list_file = open('data_%s.txt'%(image_set), 'w',encoding='UTF-8')
    for image_id in image_ids:
        image_path = image_id.split('.')[0] +'.jpg'
        list_file.write(image_path)
        convert_annotation(image_id, list_file)
        # try:
        #     convert_annotation(image_xml, list_file)
        # except:
        #     print(image_xml,'is broken')
        list_file.write('\n')
    list_file.close()

cluster_number = 9
filename = "data_train.txt"
# kmeans = YOLO_Kmeans(cluster_number, filename)
# kmeans.txt2clusters()

