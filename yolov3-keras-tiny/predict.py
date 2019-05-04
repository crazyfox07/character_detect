# @Time    : 2019/5/3 16:29
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : predict.py
import numpy as np
from PIL import Image, ImageDraw
from keras import Input
import os
from model import tiny_yolo_body
from yolov3_config import img_w, img_h, img_c, num_classes, num_anchors_per_layer, num_layers, grid_shape, threshold, \
    anchors

def calculate_iou(box1, box2):
    xmin = np.maximum(box1[0], box2[0])
    ymin = np.maximum(box1[1], box2[1])
    xmax = np.minimum(box1[2], box2[2])
    ymax = np.minimum(box1[3], box2[3])

    interact_area = (xmax - xmin) * (ymax - ymin)
    union_area =(box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - interact_area
    iou = interact_area / union_area
    return iou


img_input = Input(shape=(img_h, img_w, img_c))
model_body = tiny_yolo_body(img_input, num_anchors_per_layer, num_classes)
model_body.load_weights(r'D:\logs\yolov3\trained_weights_stage_1.h5', by_name=True)

def predict_box(img_path=r'D:\data\pascal-voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\2007_000027.jpg'):
    im = Image.open(img_path)
    im = im.resize((img_w, img_h), Image.ANTIALIAS)
    im_arr = np.asarray(im)
    im_arr = im_arr / 255
    im_arr = np.expand_dims(im_arr, axis=0)
    predict = model_body.predict(im_arr)
    draw_obj = ImageDraw.Draw(im)
    for layer in range(num_layers):
        predict_reshape = np.reshape(predict[layer],
                                     newshape=[-1, grid_shape[layer][0], grid_shape[layer][1], num_anchors_per_layer,
                                               5 + num_classes])
        box_xy = 1 / (1 + np.exp(predict_reshape[..., 0:2]))
        box_wh = predict_reshape[..., 2:4]
        box_confidence = 1 / (1 + np.exp(-predict_reshape[..., 4:5]))
        boxes_tuple = np.where(box_confidence[..., 0] > 0.6)
        boxes_index = np.stack(boxes_tuple, axis=0).transpose([1, 0])

        for box_index in boxes_index:
            b, grid_h, grid_w, anchor_index = box_index[0], box_index[1], box_index[2], int(box_index[3])
            # print([b, grid_h, grid_w, anchor_index])
            anchor_x = int((grid_w + 0.5) * img_w / grid_shape[layer][1])
            anchor_y = int((grid_h + 0.5) * img_h / grid_shape[layer][0])
            anchor_w, anchor_h = anchors[layer][anchor_index]
            anchor_x0 = anchor_x - anchor_w / 2
            anchor_y0 = anchor_y - anchor_h / 2
            anchor_x1 = anchor_x + anchor_w / 2
            anchor_y1 = anchor_y + anchor_h / 2

            box = predict_reshape[b, grid_h, grid_w, anchor_index]
            box_x = (1 / (1 + np.exp(-box[0])) + grid_w) * img_w / grid_shape[layer][1]
            box_y = (1 / (1 + np.exp(-box[1])) + grid_h) * img_h / grid_shape[layer][0]

            box_w = anchor_w * np.exp(box[2])
            box_h = anchor_h * np.exp(box[3])

            box_x0 = int(box_x - box_w/2)
            box_y0 = int(box_y - box_h / 2)
            box_x1 = int(box_x + box_w / 2)
            box_y1 = int(box_y + box_h / 2)
            box_confidence = 1 / (1 + np.exp(-box[4]))
            box_class = 1 / (1 + np.exp(-box[5:]))

            iou = calculate_iou([box_x0, box_y0, box_x1, box_y1], [anchor_x0, anchor_y0, anchor_x1, anchor_y1])
            print([anchor_x, anchor_y, anchor_w, anchor_h])
            print([box_x, box_y, box_w, box_h, iou])  # , box_confidence, box_class

            # print([box_x0, box_y0, box_x1, box_y1])
            if iou > 0.4:
                draw_obj.rectangle(xy=[box_x0, box_y0, box_x1, box_y1], outline='#ff0000', width=2)

        print('*' * 100)
    img_name = os.path.split(img_path)[-1]
    im.save('D:\logs\yolov3\img-dir\{}'.format(img_name))


if __name__ == '__main__':
    img_d = r'D:\data\pascal-voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
    for item in os.listdir(img_d)[:10]:
        predict_box(os.path.join(img_d, item))
