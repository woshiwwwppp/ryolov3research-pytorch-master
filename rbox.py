import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import math

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

aa=(0,0,10,10,0.7854)
bb=(-6,6,10,10,0.7854)
iou=rbox_iou(aa,bb)

print(iou)