import numpy as np
import torch

import r_nms

if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0,0.99],
                      [60, 60, 100, 100, 0,0.88],
                      [50, 50, 100, 100, -45.,0.66],
                      [200, 200, 100, 100, 0.,0.77]],dtype=np.float32)
    dets_th=torch.from_numpy(boxes).cuda()
    iou_thr = 0.1
    inds = r_nms.r_nms(dets_th, iou_thr)
    print(inds)

