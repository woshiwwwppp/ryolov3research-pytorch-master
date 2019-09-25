from PIL import Image, ImageFont, ImageDraw
import numpy as np
import colorsys



#定义画框函数
def draw_image_boxes(image,classes_path,out_boxes,out_classes,out_scores=[]):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    f = open('object_classes_all.txt', 'r')
    cla = f.readlines()
    pre_classes = [c.strip().split(' ')[0] for c in cla]
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    #定义ground_truth的画框函数（无scores数据）
    if not np.any(out_scores):
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]

            label = class_names[int(c)]

            label_size = draw.textsize(label, font)

            left, top, right, bottom = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print('gt:',label, (left, top), (right, bottom))

            if bottom + label_size[1] <= 832:
                text_origin = np.array([left, bottom])
            else:
                text_origin = np.array([left, bottom - label_size[1]])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=(0,0,0))
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=(0,0,0))
            draw.text(text_origin, label, fill=(255, 255, 255), font=font)
    #定义预测框的画框函数（有scores数据）
    else:
        hsv_tuples = [(x / len(pre_classes), 1., 1.)
                      for x in range(len(pre_classes))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[int(c)]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.5f}'.format(predicted_class, score)

            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print('predicted:',label, (left, top), (right, bottom))
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    del draw



