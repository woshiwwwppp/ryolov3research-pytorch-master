import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import torch
import cv2
from yolo.base_model import yolo_loss
from data_aug.data_aug import *
from data_aug.bbox_util import *
import config
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import math
from tensorboardX import SummaryWriter
import copy
import matplotlib.pyplot as plt
import config
from yolo.nms.r_nms import r_nms
def angle2point(b):
    # b = (cx, cy, rw, rh,angle)
    bow_x = b[0] + b[2] / 2 * math.cos(float(b[4]))
    bow_y = b[1] - b[2] / 2 * math.sin(float(b[4]))
    tail_x = b[0] - b[2] / 2 * math.cos(float(b[4]))
    tail_y = b[1] + b[2] / 2 * math.sin(float(b[4]))
    x1 = int(round(bow_x + b[3] / 2 * math.sin(float(b[4]))))
    y1 = int(round(bow_y + b[3] / 2 * math.cos(float(b[4]))))
    x2 = int(round(tail_x + b[3] / 2 * math.sin(float(b[4]))))
    y2 = int(round(tail_y + b[3] / 2 * math.cos(float(b[4]))))
    x3 = int(round(tail_x - b[3] / 2 * math.sin(float(b[4]))))
    y3 = int(round(tail_y - b[3] / 2 * math.cos(float(b[4]))))
    x4 = int(round(bow_x - b[3] / 2 * math.sin(float(b[4]))))
    y4 = int(round(bow_y - b[3] / 2 * math.cos(float(b[4]))))
    return np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]],dtype='float32')

def rbox_iou(a,b):
    b = angle2point(b)
    a = angle2point(a)

    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    poly2 = Polygon(b).convex_hull
    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2

    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou
def convert_ground_truth(gt_boxes, input_shape, anchors, num_classes,batch_size,ratio):
    '''convert ground truth boxes into yolo_outputs frame as following functions:
        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        bw = pw*exp(tw)
        bh = ph*exp(th)

        Parameters
        ----------
        input_shape: model input shape, such as (416,416)
        gt_boxes: list of ground truth boxes
            [[x_min, y_min, x_max, y_max, class_id],[x_min, y_min, x_max, y_max, class_id],...]
        anchors: anchros array, shape=(9, 2)
        num_classes: .number of classes, integer
        Returns
        -------
        y_true: list of array, shape like yolo_outputs

        '''
    grid = config.dataSet['grid']
    num_layers = config.dataSet['num_layers']
    anchor_mask = config.dataSet['anchors_mask']
    m = len(gt_boxes)     # batch_size
    # initialize y_true with zeros
    y_true = [np.zeros((m, len(anchor_mask[i]), num_classes + 6, input_shape[0] // grid[i], input_shape[0] // grid[i]),
                       dtype='float32') for i in range(num_layers)]
    conf_false_mask = [y[..., 5, :, :].copy() for y in y_true]
    for i in range(m):#
        true_boxes = gt_boxes[i]#boxes in one batch
        if len(true_boxes) !=0:
            boxes_xy = true_boxes[...,0:2]
            boxes_wh = true_boxes[...,2:4]
            boxes_angle=true_boxes[...,4]

            anchors_list=anchors.copy()
            anchors_list[:,0:2]=anchors_list[:,0:2]*ratio#anchor and gt were resized
            iou=np.zeros((len(true_boxes),len(anchors)))
            for ii,true_boxe in enumerate(true_boxes):
                for jj,anchor in enumerate(anchors_list):
                    iou[ii,jj]=rbox_iou((0,0,anchor[0],anchor[1],anchor[2]),(0,0,true_boxe[2],true_boxe[3],true_boxe[4]))


            best_anchor = np.argmax(iou, axis=-1)
            ignore_anchor= np.where(iou>0.4)
            # print('a')
            # convert ground truth boxes
            for t, n in enumerate(best_anchor):#n: id of best anchor in mask,t :id of gtbox
                for l in range(num_layers):
                    if n in anchor_mask[l]:
                        xy = np.floor(boxes_xy[t] / grid[l]).astype('int32')
                        y_xy = boxes_xy[t] / grid[l] - xy
                        y_wh = np.log(boxes_wh[t]/anchors_list[n,0:2])
                        angle_offest=np.tan(boxes_angle[t]-anchors_list[n,2])
                        anchro_id = anchor_mask[l].index(n)
                        class_id = true_boxes[t, 5]

                        y_true[l][i, anchro_id, 0:2, xy[0], xy[1]] = y_xy#l:num layer,i:anchor_num,label,x,y
                        y_true[l][i, anchro_id, 2:4, xy[0], xy[1]] = y_wh
                        y_true[l][i, anchro_id, 4, xy[0], xy[1]] = angle_offest
                        y_true[l][i, anchro_id, 5, xy[0], xy[1]] = 1
                        y_true[l][i, anchro_id, int(class_id + 6), xy[0], xy[1]] = 1

            for iii in range(ignore_anchor[0].shape[0]):  # 0,gt 1,mask
                for l in range(num_layers):
                    if ignore_anchor[1][iii] in anchor_mask[l]:
                        xy = np.floor(boxes_xy[ignore_anchor[0][iii]] / grid[l]).astype('int32')
                        anchro_id = anchor_mask[l].index(ignore_anchor[1][iii])
                        conf_false_mask[l][i, anchro_id,xy[0], xy[1]] = 1

        else: continue

    return y_true,conf_false_mask

def resize(image,input_shape):
    '''
    resize image with unchanged aspect ratio using padding
    return: resized image,ratio
    '''
    img_w, img_h = image.shape[1], image.shape[0]
    w, h = input_shape
    ratio = min(w / img_w, h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    empty = np.zeros((w, h, 3), dtype='uint8')
    empty[0:new_h, 0:new_w, :] = resized_image
    return empty,ratio


def get_input_data(image):
    image_data = np.array(image,dtype='float32')
    image_data /= 255
    image_data = np.array([image_data[..., i] for i in range(3)])
    return image_data


def IOU(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    s1 = abs(bottom1 - top1) * abs(right1 - left1)
    s2 = abs(bottom2 - top2) * abs(right2 - left2)
    cross = max((min(bottom1, bottom2) - max(top1, top2)), 0) * max((min(right1, right2) - max(left1, left2)), 0)
    return cross / (s1 + s2 - cross) if (s1 + s2 - cross)!=0 else 0

def convert_yolo_outputs(out_puts, input_shape, ratio, anchors, classes, confidence = 0.05, NMS = 0.5, CUDA= True):
    '''convert yolo out puts into object boxes with following functions:
           bx = sigmoid(tx) + cx
           by = sigmoid(ty) + cy
           bw = pw*exp(tw)
           bh = ph*exp(th)

           Parameters
           ----------
           out_puts : yolo out puts
           input_shape: model input shape, such as (416,416)
           gt_boxes: list of ground truth boxes
               [[x_min, y_min, x_max, y_max, class_id],[x_min, y_min, x_max, y_max, class_id],...]
           anchors: anchros array, shape=(9, 2)
           classes: list of classes
           confidence : confidence threshold
           NMS ： NMS threshold
           Returns
           -------
           object boxes: list of boxes

           '''
    anchor_mask = config.dataSet['anchors_mask']
    num_layers = len(anchor_mask)
    num_classes = len(classes)
    input_shape = np.array(input_shape)
    out_box = []
    out_scor = []
    out_class = []
    for k in range(out_puts[0].shape[0]):
        outRes=torch.tensor([[0.,0.,0.,0.,0.,0.]]).cuda()
        outId=torch.tensor([0]).cuda()
        for i in range(num_layers):
            scal = out_puts[i].cpu().data.shape[-1]
            pred = out_puts[i].cpu().data.reshape(-1,len(anchor_mask[i]),num_classes+6,scal,scal)[k,...].unsqueeze(0)
            anchor = [anchors[anchor_mask[i][n]] for n in range(len(anchor_mask[i]))]
            anchor = torch.FloatTensor(anchor)
            grid = np.meshgrid(range(scal),range(scal))
            grid = torch.FloatTensor(grid[::-1]).unsqueeze(0).repeat(len(anchor_mask[i]),1,1,1).unsqueeze(0)
            # 用GPU完成张量运算
            if CUDA:
                pred = pred.cuda()
                anchor = anchor.cuda()
                grid = grid.cuda()
            # 计算预测框包含目标的置信度score
            confidence_prob =  torch.sigmoid(pred[..., 5, :, :]).unsqueeze(2)
            object_mask =  (confidence_prob > confidence)
            pred[..., 6:, :, :] = torch.sigmoid(pred[..., 6:, :, :]) * confidence_prob#scores
            # 计算预测框中心点坐标x,y
            pred[..., 0:2, :, :] = (torch.sigmoid(pred[..., 0:2, :, :]) + grid) / scal * input_shape[0] / ratio#xy
            # 计算预测框的长宽h,w
            x = anchor[:, 0].view(-1, 1).repeat(1, scal * scal).reshape(len(anchor_mask[i]), scal, scal).unsqueeze(1)
            y = anchor[:, 1].view(-1, 1).repeat(1, scal * scal).reshape(len(anchor_mask[i]), scal, scal).unsqueeze(1)
            angle_offset = anchor[:, 2].view(-1, 1).repeat(1, scal * scal).reshape(len(anchor_mask[i]), scal, scal)
            anchor_xy = torch.cat((x, y), 1)
            pred[..., 2:4, :, :] = (torch.exp(pred[..., 2:4, :, :]).squeeze(0) * anchor_xy).unsqueeze(0)#wh
            pred[..., 4, :, :]=torch.atan(pred[...,4, :, :])+angle_offset.unsqueeze(0)#angle
            pred.permute(0,1,3,4,2)
            object_mask=object_mask.repeat(1,1,pred.size()[2],1,1)
            pred[..., 5, :, :]=torch.max(pred[..., 6:,:,:])#max score
            res=pred[object_mask].view(-1,pred.size()[2])
            resBox=res[:,0:6]
            cla_id = torch.argmax(res[:,6:],dim=1)
            outRes = torch.cat([outRes, resBox], 0)
            outId=torch.cat((outId,cla_id),0)
        inds = r_nms(outRes,NMS )
        inds_=inds.cpu().data.numpy()
        outRes_=outRes.cpu().data.numpy()
        outId_=outId.cpu().data.numpy()
        for id in inds_[1:]:
            out_box.append(outRes_[id][0:5])
            out_scor.append(outRes_[id][5])
            out_class.append(classes[outId_[id]])

    return out_box,out_scor,out_class

def data_generator(annotation_lines,input_shape,anchors, num_classes,batch_size,step,rand = True):
    image = []
    gt_boxes = []
    Ratio=0.
    # imgName=[]
    for annotation in annotation_lines[step*batch_size:(step+1)*batch_size]:
        annotation = annotation.strip()
        img = cv2.imread(annotation.split()[0])[:,:,::-1]
        # imgName.append(annotation.split()[0])
        boxes = np.array([np.array(box.split(',')) for box in annotation.split()[1:]],dtype=np.float32)
        img,ratio = resize(img,input_shape)
        Ratio=ratio
        boxes[...,:4] *=ratio
        if rand:
            temp_boxes = np.array([[100,100,200,200,0]],dtype = np.float32)
            transforms = Sequence([RandomHorizontalFlip(0.5),
                                   RandomTranslate(0.1, diff=True, remove=0.8),RandomHSV(20, 40, 40)])
            if len(boxes) !=0:
                img, boxes = transforms(img, boxes)
            else:
                img, temp_boxes = transforms(img, temp_boxes)

        image_data = get_input_data(img)
        image.append(image_data)
        gt_boxes.append(boxes)
    image = np.array(image)
    y_true,conf_false_mask = convert_ground_truth(gt_boxes,input_shape,anchors,num_classes,batch_size,Ratio)
    X = torch.from_numpy(image)


    return X,y_true,conf_false_mask,ratio

def evalAll(model,val,MatchIOUNum,NMSNum,confidenceNum,input_shape,batch_size, anchors,classes,loss_function,CUDA):
    writer = SummaryWriter()
    model.eval()
    val_lines = open(val,'r').readlines()
    if '\n' in val_lines:
        val_lines.remove('\n')
    np.random.shuffle(val_lines)
    steps = len(val_lines)//batch_size
    num_classes = len(classes)
    precision = {}
    recall = {}
    if CUDA:
        model.cuda()
    for i in classes:
        precision[i] = []
        recall[i] = []
    loss = 0
    MatchIOUNum=4#0.3+0.1*
    NMSNum=3#0.3+0.1*
    confidenceNum=10#0+0.08*
    All = [[[copy.deepcopy(precision) for i in range(NMSNum*confidenceNum)],[copy.deepcopy(recall) for j in range(NMSNum*confidenceNum)]] for kk in range(MatchIOUNum)]
    for step in range(steps):
        torch.cuda.empty_cache()
        sys.stdout.write('\r')
        sys.stdout.write("evaluating validation data...%d//%d" % (int(step + 1), int(steps)))
        sys.stdout.flush()
        X, y_true,conf_false_mask,ratio = data_generator(val_lines, input_shape, anchors, num_classes, batch_size, step, rand = False)
        if CUDA:
            X = X.cuda()
        with torch.no_grad():
            out_puts = model(X)
        loss += yolo_loss(out_puts,y_true,conf_false_mask,num_classes,anchors,input_shape,ratio,CUDA,
                          loss_function = 'None',print_loss = False)

        for match_iou in range(0, MatchIOUNum):
            for NMS in range(0, NMSNum):
                for confidence in range(0, confidenceNum):
                    out_box, out_score, out_class = convert_yolo_outputs(out_puts, input_shape, ratio, anchors,
                                                                              classes, (confidence+1)*0.08, (NMS+3)*0.1, CUDA=True)
                    for k, v in enumerate(val_lines[step * batch_size:(step + 1) * batch_size]):
                        gt_boxes = []
                        gt_classes = []
                        for gt in v.strip().split(' ')[1:]:
                            gt_boxes.append(list(map(float, gt.split(',')[:-1])))
                            gt_classes.append(classes[int(gt.split(',')[-1].strip())])
                        out_classes = out_class[k]
                        out_boxes = out_box[k]
                        # 计算ap
                        for i in range(len(out_classes)):
                            for j in range(len(gt_classes)):
                                if rbox_iou(gt_boxes[j], out_boxes[i]) > (match_iou+1)*0.1 and gt_classes[j] == out_classes[i]:
                                    # precision[out_classes[i]].append(1)
                                    All[match_iou][0][NMS*confidenceNum+confidence][out_classes[i]].append(1)
                                    break
                            else:
                                All[match_iou][0][NMS * confidenceNum + confidence][out_classes[i]].append(0)
                            # 计算ar
                        for i in range(len(gt_classes)):
                            for j in range(len(out_classes)):
                                if rbox_iou(gt_boxes[i], out_boxes[j]) > 0.1*(match_iou+3) and out_classes[j] == gt_classes[i]:
                                    # recall[gt_classes[i]].append(1)
                                    All[match_iou][1][NMS * confidenceNum + confidence][gt_classes[i]].append(1)

                                    break
                            else:
                                # recall[gt_classes[i]].append(0)
                                All[match_iou][1][NMS * confidenceNum + confidence][gt_classes[i]].append(0)


    print('\n')
    torch.cuda.empty_cache()
    model.train()
    plt.figure()
    for match_iou in range(0, MatchIOUNum):
        plt.subplot(2, 2, match_iou+1)
        plt.title("match_iou%.3f"%(0.1*(match_iou+3)))
        plt.xlim(xmax=1, xmin=0)
        plt.ylim(ymax=1, ymin=0)
        plt.xlabel("mar")
        plt.ylabel("map")
        for NMS in range(0, NMSNum):
            for confidence in range(0, confidenceNum):
                precision=All[match_iou][0][NMS * confidenceNum + confidence]
                recall=All[match_iou][1][NMS * confidenceNum + confidence]
                ap = []
                ar = []
                for k in precision.keys():
                    p = sum(precision[k]) / len(precision[k]) if len(precision[k]) != 0 else 0
                    r = sum(recall[k]) / len(recall[k]) if len(recall[k]) != 0 else 0
                    ap.append(p)
                    ar.append(r)

                plt.plot(float(sum(ar)/len(ar)), float(sum(ap)/len(ap)), 'ro')

                # writer.add_scalar("match_iou%.3f"%(0.1*(match_iou+3)), float(sum(ap)/len(ap)), float(sum(ar)/len(ar)))
                print('match_iou:%.3f,NMS:%.3f,confidence:%.3f,mAP :'%(0.1*(match_iou+3),(NMS+3)*0.1,(confidence+1)*0.08), '%.3f' % float(sum(ap)/len(ap)), 'mAR:', '%.3f' % float(sum(ar)/len(ar)))
        print('\n')
    plt.show()



def eval(model,val,matchIou,NMS,confidence,input_shape,batch_size, anchors,classes,loss_function,CUDA):
    model.eval()
    val_lines = open(val,'r').readlines()
    if '\n' in val_lines:
        val_lines.remove('\n')
    np.random.shuffle(val_lines)
    steps = len(val_lines)//batch_size
    num_classes = len(classes)
    precision = {}
    recall = {}
    if CUDA:
        model.cuda()
    for i in classes:
        precision[i] = []
        recall[i] = []
    loss = 0
    for step in range(steps):
        torch.cuda.empty_cache()
        sys.stdout.write('\r')
        sys.stdout.write("evaluating validation data...%d//%d" % (int(step + 1), int(steps)))
        sys.stdout.flush()
        X, y_true,conf_false_mask,ratio = data_generator(val_lines, input_shape, anchors, num_classes, batch_size, step, rand = False)
        if CUDA:
            X = X.cuda()
        with torch.no_grad():
            out_puts = model(X)
        loss += yolo_loss(out_puts,y_true,conf_false_mask,num_classes,anchors,input_shape,ratio,CUDA,
                          loss_function = 'None',print_loss = False)
        out_box, out_score, out_class = convert_yolo_outputs(out_puts, input_shape, ratio, anchors,
                                                                  classes, confidence, NMS, CUDA=True)
        for k, v in enumerate(val_lines[step * batch_size:(step + 1) * batch_size]):
            gt_boxes = []
            gt_classes = []
            for gt in v.strip().split(' ')[1:]:
                gt_boxes.append(list(map(float, gt.split(',')[:-1])))
                gt_classes.append(classes[int(gt.split(',')[-1].strip())])
            out_classes = out_class[k]
            out_boxes = out_box[k]
            # 计算ap
            for i in range(len(out_classes)):
                for j in range(len(gt_classes)):
                    if rbox_iou(gt_boxes[j], out_boxes[i]) > matchIou and gt_classes[j] == out_classes[i]:
                        precision[out_classes[i]].append(1)
                        break
                else:
                    precision[out_classes[i]].append(0)
                # 计算ar
            for i in range(len(gt_classes)):
                for j in range(len(out_classes)):
                    if rbox_iou(gt_boxes[i], out_boxes[j]) > matchIou and out_classes[j] == gt_classes[i]:
                        recall[gt_classes[i]].append(1)
                        break
                else:
                    recall[gt_classes[i]].append(0)
    print('\n')
    torch.cuda.empty_cache()
    model.train()
    ap = []
    ar = []
    for k in precision.keys():
        p = sum(precision[k]) / len(precision[k]) if len(precision[k]) != 0 else 0
        r = sum(recall[k]) / len(recall[k]) if len(recall[k]) != 0 else 0
        ap.append(p)
        ar.append(r)
        print(k, 'AP:', '%.3f' % (p), 'AR:', '%.3f' % (r))
    print('mAP :', '%.3f' % float(sum(ap)/len(ap)), 'mAR:', '%.3f' % float(sum(ar)/len(ar)))
    return sum(ap)/len(ap),sum(ar)/len(ar),loss/steps

if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0, 0.99],
                      [60, 60, 100, 100, 0, 0.88],
                      [50, 50, 100, 100, -45., 0.66],
                      [200, 200, 100, 100, 0., 0.77]], dtype=np.float32)
    dets_th = torch.from_numpy(boxes).cuda()
    iou_thr = 0.1
    inds = r_nms(dets_th, iou_thr)
    print(inds)






