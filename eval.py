import numpy as np
from yolo.utility import data_generator,eval,evalAll
from yolo.base_model import yolo_body,yolo_loss
import config
from tensorboardX import SummaryWriter
from tensorboardX import SummaryWriter

def creat_yolo_model(num_classes,weight_path=None):
    print('load model...')
    yolo = yolo_body(num_classes)
    if weight_path:
        yolo.load_weight(weight_path)
    print('load {} successed!'.format(weight_path))
    return yolo

if __name__ == '__main__':
    CUDA = True
    anchors = config.dataSet['anchors']
    anchors = np.array(anchors)
    anchors = anchors.reshape(-1, 3)
    input_shape = (config.dataSet['img_shape'][0],config.dataSet['img_shape'][1])
    classes = open('train_data/train_classes.txt').readlines()
    classes = [c.strip() for c in classes]
    num_classes = len(classes)
    annotations = 'data_train.txt'
    val = 'data_val.txt'
    weight_path = '/media/wp/windows/pyProject/detection/ryolov3research-pytorch-master/aa.pt'
    batch_size = 8
    yolo = creat_yolo_model(num_classes,weight_path = weight_path)
    feeeze_body = False
    evalAll(yolo, val, 0, 0, 0, input_shape, batch_size, anchors, classes,loss_function=None, CUDA=CUDA)

