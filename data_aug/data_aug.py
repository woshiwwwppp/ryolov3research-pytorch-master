import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
# from data_aug.bbox_util import *
import math
lib_path = os.path.join(os.path.realpath("."), "data_aug")
sys.path.append(lib_path)


class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, bboxes):
            img_center = np.array(img.shape[:2])[::-1]/2
            img_center = np.hstack((img_center, img_center))
            if random.random() < self.p:
                img = img[:, ::-1, :]
                bboxes[:, [0]] += 2*(img_center[[0]] - bboxes[:, [0]])
                bboxes[:, [4]]=math.pi-bboxes[:, [4]]
            elif random.random() < self.p*2:
                img = img[::-1, :, :]
                bboxes[:, [1]] += 2 * (img_center[[1]] - bboxes[:, [1]])
                bboxes[:, [4]] = math.pi - bboxes[:, [4]]

            return img, bboxes

class RandomTranslate(object):
    """Randomly Translates the image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn 
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, translate = 0.2, diff = True , remove = 0.5):
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1


        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)
            
            
        self.diff = diff
        self.remove = remove

    def __call__(self, img, bboxes):        
        #Chose a random digit to scale by 
        img_shape = img.shape
        canvas = np.zeros(img_shape).astype(np.uint8)
        corner_x = int(random.uniform(-70,70))
        corner_y = int(random.uniform(-70,70))
        bboxes_change = bboxes[:, :2] + [corner_x, corner_y]
        if (bboxes_change>0).all() and (bboxes_change<img_shape[0]).all() :
            orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),min(img_shape[1], corner_x + img.shape[1])]
            mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
                   max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
            canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
            img = canvas
            bboxes[:, :2] += [corner_x, corner_y]
            return img, bboxes
        return img, bboxes


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
    #
    # # compute the new bounding dimensions of the image
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    #
    # # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (w, h))

    #    image = cv2.resize(image, (w,h))
    return image


def rotate_box(corners, angle, cx, cy, h, w):

    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])
    #
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # # adjust the rotation matrix to take into account translation
    # M[0, 2] += (nW / 2) - cx
    # M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    # calculated = calculated.reshape(-1, 8)

    return calculated

class RandomRotate(object):
    """Randomly rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle=10, remove=0.5):
        self.angle = angle
        self.remove = remove

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):
        angle = random.uniform(*self.angle)
        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2
        bboxes_change = bboxes.copy()

        if len(bboxes) > 0:
            bboxes_change[:,4]-=angle/180*math.pi
            bboxes_change[:,0:2] = rotate_box(bboxes[:,0:2], angle, cx, cy, h, w)
        if (bboxes_change[:,0:2]>0).all() and (bboxes_change[:,0:2]<img.shape[0]).all():
            img = rotate_im(img, angle)
            bboxes = bboxes_change
            for box in bboxes:
                if box[4]>math.pi:
                    box[4]-=math.pi
                if box[4]<0:
                    box[4]+=math.pi

        return img, bboxes

class RandomHSV(object):
    """HSV Transform to vary hue saturation and brightness
    
    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255. 
    Chose the amount you want to change thhe above quantities accordingly. 
    
    
    
    
    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-hue, hue) and added to the 
        hue of the image. If tuple, the int is sampled from the range 
        specified by the tuple.   
        
    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-saturation, saturation) 
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.   
        
    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-brightness, brightness) 
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.   
    
    Returns
    -------
    
    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """
    
    def __init__(self, hue = None, saturation = None, brightness = None,p=0.5):
        if hue:
            self.hue = hue 
        else:
            self.hue = 0
            
        if saturation:
            self.saturation = saturation 
        else:
            self.saturation = 0
            
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0
        self.p=p
            

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
            
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        
        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)
    
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            hue = random.randint(*self.hue)
            saturation = random.randint(*self.saturation)
            brightness = random.randint(*self.brightness)

            img = img.astype(int)

            a = np.array([hue, saturation, brightness]).astype(int)
            img += np.reshape(a, (1,1,3))

            img = np.clip(img, 0, 255)
            img[:,:,0] = np.clip(img[:,:,0],0, 179)

            img = img.astype(np.uint8)


        
        return img, bboxes
    
class Sequence(object):

    """Initialise Sequence object
    
    Apply a Sequence of transformations to the images/boxes.
    
    Parameters
    ----------
    augemnetations : list 
        List containing Transformation Objects in Sequence they are to be 
        applied
    
    probs : int or list 
        If **int**, the probability with which each of the transformation will 
        be applied. If **list**, the length must be equal to *augmentations*. 
        Each element of this list is the probability with which each 
        corresponding transformation is applied
    
    Returns
    -------
    
    Sequence
        Sequence Object 
        
    """
    def __init__(self, augmentations, probs = 1):

        
        self.augmentations = augmentations
        self.probs = probs
        
    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes

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


if __name__ == '__main__':
    boxes = np.array([[2675, 1706, 435, 1333, 1.06467271816]], dtype=np.float32)
    img=cv2.imread('/media/wp/windows/data/insulator/images/001501928_K1590435_10000020_1_06.jpg')
    transforms = Sequence([RandomHorizontalFlip(0.3),
                           RandomTranslate(0.1, diff=True, remove=0.8), RandomRotate(20, remove=0.8),RandomHSV(20, 40, 40)])
    img, boxess = transforms(img, boxes)
    for i, box in enumerate(boxess):
        box[4] = math.pi - box[4]
        boxPoint = angle2point(box)
        boxPoint = boxPoint.astype(np.int16).copy()
        cv2.line(img, (boxPoint[0][0], boxPoint[0][1]), (boxPoint[1][0], boxPoint[1][1]), (0, 255, 0), 5)
        cv2.line(img, (boxPoint[1][0], boxPoint[1][1]), (boxPoint[2][0], boxPoint[2][1]), (0, 255, 0), 5)
        cv2.line(img, (boxPoint[2][0], boxPoint[2][1]), (boxPoint[3][0], boxPoint[3][1]), (0, 255, 0), 5)
        cv2.line(img, (boxPoint[3][0], boxPoint[3][1]), (boxPoint[0][0], boxPoint[0][1]), (0, 255, 0), 5)
    cv2.namedWindow("pic",cv2.WINDOW_NORMAL)
    cv2.imshow('pic',img)

    cv2.waitKey(0)
