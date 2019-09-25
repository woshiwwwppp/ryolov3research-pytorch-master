import torch
import torch.nn.functional as F
import numpy as np
from yolo.utility import data_generator,eval
from yolo.base_model import yolo_body,yolo_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import config
from tensorboardX import SummaryWriter
#

def early_stop(map):
    map = np.array(map)
    index = map[-1] <= map[-8:-1]*1.01
    if index.sum() >=len(map)-1:
        return True
    else:
        return False

def train(yolo_model,feeeze_body,epoche,batch_size,annotations,val,
          input_shape, anchors, classes, CUDA,loss_function ):
    annotation_lines = open(annotations,'r').readlines()
    if '\n' in annotation_lines:
        annotation_lines.remove('\n')
    steps = len(annotation_lines)//batch_size
    num_classes = len(classes)
    # if feeeze_body:
    #     optimizer = torch.optim.Adam([{'params': yolo_model.yolo_block1.conv7.parameters()},
    #                                   {'params': yolo_model.yolo_block2.conv7.parameters()},
    #                                   {'params': yolo_model.yolo_block3.conv7.parameters()}], lr=1e-5)
    # else:
    optimizer = torch.optim.Adam(yolo_model.parameters(),lr=1e-4)
    if CUDA:
        yolo_model.cuda()
    scheduler = ReduceLROnPlateau(optimizer, mode= 'min',verbose = True, )
    map_list = []
    writer = SummaryWriter()
    for i in range(epoche):
        print('train yolov3 on epoch {} with bath size {}'.format(i+1,batch_size))
        np.random.shuffle(annotation_lines)
        losses = []
        for step in range(steps):
            print('step: {}/{}'.format(step+1,steps))
            X, y_true,conf_false_mask,ratio= data_generator(annotation_lines, input_shape, anchors, num_classes, batch_size, step)
            if CUDA:
                X = X.cuda()
            optimizer.zero_grad()
            Y = yolo_model(X)
            loss = yolo_loss(Y, y_true,conf_false_mask, num_classes, anchors, input_shape,ratio, CUDA, loss_function = loss_function, print_loss=True)
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.cpu().item())
            writer.add_scalar("train_loss", losses[-1], step+i*steps)
        if i>3:
            map,mar,val_loss = eval(yolo_model,val,0.5,config.dataSet['NMS'],config.dataSet['confidence'],input_shape,batch_size, anchors,classes,loss_function=None,CUDA=CUDA)
            writer.add_scalar("map", map, i)
            writer.add_scalar("mar", mar, i)
            writer.add_scalar("val_loss", val_loss, i)
            scheduler.step(val_loss)
            map_list.append(map)
        # plt.plot(losses)
        # plt.show()
        if not ((i+1)%10):
            torch.save(yolo.state_dict(),'epoche%d.pt'%(i) )

        # if len(map_list) >= 8 and early_stop(map_list):
        #     break
    # plt.plot(map_list)
    # plt.show()

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
    weight_path = None
    batch_size = 5
    epoche = 30
    dummy_input = torch.rand(batch_size, 3, input_shape[0], input_shape[1])
    yolo = creat_yolo_model(num_classes,weight_path = weight_path)
    # with SummaryWriter(comment='yolo')as w:
    #      w.add_graph(yolo, (dummy_input,))
    feeeze_body = False
    train(yolo, feeeze_body, epoche, batch_size, annotations, val, input_shape, anchors,
          classes,CUDA, loss_function = 'None')
    torch.save(yolo.state_dict(),'aa.pt')
