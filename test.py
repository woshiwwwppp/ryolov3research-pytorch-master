import torch
import torch.nn as nn
import torch.nn.functional as F
from yolo import base_model
import time
import cv2
import numpy as np
from yolo.utility import convert_yolo_outputs,convert_ground_truth,resize,get_input_data
import matplotlib.pyplot as plt
import config
from yolo.utility import angle2point
import math
torch.cuda.empty_cache()
start1 = time.time()
yolo = base_model.yolo_body(1)
yolo.load_weight('/media/wp/windows/pyProject/detection/ryolov3research-pytorch-master/aa.pt')
yolo.eval()
yolo.cuda()
print('load time:',time.time()-start1)
start2 = time.time()
image = cv2.imread('/media/wp/windows/data/insulator/images_test/003436194_K1586173_10000025_1_23.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
resize_image,ratio = resize(image,(640,640))
image_data = get_input_data(resize_image)
image_data = np.expand_dims(image_data,0)
X =torch.from_numpy(image_data)
#输出转换
with open('object_classes_all.txt','r') as f:
    classes = f.readlines()
classes = [c.strip() for c in classes]
anchors = config.dataSet['anchors']
anchors = np.array(anchors)
anchors = anchors.reshape(-1, 3)
X = X.cuda()
for aaa in range(20):
    start2 = time.time()
    with torch.no_grad():
        out_puts = yolo(X)
    out_boxes,out_scores,out_classes = convert_yolo_outputs(out_puts,(640,640),ratio, anchors,
                                                            classes,confidence = config.dataSet['confidence'],NMS = config.dataSet['NMS'],CUDA= True)
    print('processing time:', time.time() - start2)


for i,box in enumerate(out_boxes):
    label = out_classes[0][i]
    box[4]=math.pi-box[4]
    boxPoint=angle2point(box)
    boxPoint = boxPoint.astype(np.int16).copy()
    cv2.line(image, (boxPoint[0][0],boxPoint[0][1]), (boxPoint[1][0],boxPoint[1][1]), (0, 255, 0),5)
    cv2.line(image, (boxPoint[1][0],boxPoint[1][1]), (boxPoint[2][0],boxPoint[2][1]), (0, 255, 0),5)
    cv2.line(image, (boxPoint[2][0],boxPoint[2][1]), (boxPoint[3][0],boxPoint[3][1]), (0, 255, 0),5)
    cv2.line(image, (boxPoint[3][0],boxPoint[3][1]), (boxPoint[0][0],boxPoint[0][1]), (0, 255, 0),5)
    # cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,0,0),3)
    # cv2.putText(image, label, (box[0],box[1]+20), cv2.FONT_HERSHEY_PLAIN, 2, [225,0,0], 2)

cv2.namedWindow("pic",cv2.WINDOW_NORMAL)
cv2.imshow("pic",image)
cv2.waitKey(0)
# plt.imsave('test.jpg',image)
# print('Output Processing:',time.time()-start3)
# print('total time:',time.time()-start1)
# with SummaryWriter() as w:
#     w.add_graph(yolo,X,verbose=True)
# from data_aug.data_aug import *
# from data_aug.bbox_util import *
# import cv2
# #
# #
# with open('test1.txt') as f:
#     lines_train = f.readlines()
# for annotation_line in lines_train:
#     annotation_line = annotation_line.strip()
#     bboxes = np.array([np.array(box.split(',')) for box in annotation_line.split()[1:]],dtype=np.float32)
#     np.random.shuffle(bboxes)
#     img = cv2.imread(annotation_line.split()[0])[:,:,::-1] #convert to RGB
#     img,ratio = resize(img,(416,416))
#     bboxes[...,:4] *=ratio
#
#     transforms = Sequence([RandomHorizontalFlip(0.5), RandomRotate(10,remove=0.8),
#                            RandomScale(0.2,diff=True,remove=0.8),RandomShear(0.1),
#                            RandomTranslate(0.1,diff=True,remove=0.8),RandomHSV(20,40,40)])
#
#
#     temp_boxes = np.array([[100,100,200,200,1]],dtype=np.float32)
#     if len(bboxes) != 0:
#         img, bboxes = transforms(img, bboxes)
#         plt.imshow(draw_rect(img, bboxes))
#         plt.show()
#     else:
#         img, temp_boxes = transforms(img, temp_boxes)
#         plt.imshow(img)
#         plt.show()
#     print(bboxes)
# b = []
# b.append(trans_bboxes)
# anchor = [[21,21], [36,36], [54,54], [75,72], [99,95], [121,134], [179,179], [268,283], [475,480]]
# y = convert_ground_truth(b,(416,416),anchor,30)
