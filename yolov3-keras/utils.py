# @Time    : 2019/4/5 15:50
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : utils.py
from random import shuffle
import time
import os
import numpy as np
from PIL import Image
from yolov3_config import img_dir_train, img_h, img_w, img_c, imgname_label_txt, boxes_max_num, grid_shape, anchors, \
    num_classes, classes, num_anchors_per_layer
import tensorflow as tf
from keras import backend as K
def handle_img(img_path):
    im = Image.open(img_path)
    im = im.resize((img_w, img_h), Image.ANTIALIAS)
    im = np.asarray(im)
    im = im / 255
    return im


def handle_y_true(img_w_src, img_h_src, boxes):
    result = list()
    for box in boxes:
        class_name, x0, y0, x1, y1 = box.split(',')
        scale_x, scale_y = img_w/float(img_w_src), img_h/float(img_h_src)
        class_index = classes[class_name]
        x0, x1 = scale_x * float(x0), scale_x * float(x1)
        y0, y1 = scale_y * float(y0), scale_y * float(y1)
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        x, y = (x0 + x1) // 2, (y0 + y1) // 2
        w, h = x1 - x0, y1 - y0
        anchors_arr = np.array(anchors)
        box_arr = np.array([w, h])
        box_maxs = box_arr // 2
        box_mins = -box_maxs
        anchor_maxs = anchors_arr // 2
        anchor_mins = -anchor_maxs
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxs = np.minimum(box_maxs, anchor_maxs)
        intersect_w_h = intersect_maxs - intersect_mins
        intersect_areas = intersect_w_h[..., 0] * intersect_w_h[..., 1]
        union_areas = w * h + anchors_arr[..., 0] * anchors_arr[..., 1] - intersect_areas
        iou_arr = intersect_areas / union_areas
        iou_max_index = iou_arr.argmax()
        result.append((class_index, iou_max_index, x, y, w, h))
    return result


def data_generate(batch_size=64):
    with open(imgname_label_txt) as fr:
        lines = fr.readlines()
        lines_len = len(lines)
    while True:
        count = 0
        x_img = np.zeros(shape=(batch_size, img_h, img_w, img_c))
        y_true = [np.zeros(shape=(batch_size, shape[0], shape[1], num_anchors_per_layer, 5 + num_classes)) for shape in
             grid_shape]
        begin = time.time()
        for i in range(batch_size):
            line = lines[count]
            items = line.split(' ')
            img_name, img_w_src, img_h_src = items[0].split(',')
            img_path = os.path.join(img_dir_train, img_name)
            x_img[i] = handle_img(img_path)
            y_true_boxes = handle_y_true(img_w_src, img_h_src, items[1:])
            for y_true_box in y_true_boxes:
                class_index, iou_max_index, x, y, w, h = y_true_box
                # print(class_index, iou_max_index, x, y, w, h)
                layer = iou_max_index // 3
                anchor_index = iou_max_index % 3
                grid_shape_h, grid_shape_w = grid_shape[layer]
                grid_h = int(y / img_h * grid_shape_h)
                grid_w = int(x / img_w * grid_shape_w)

                tx = x / img_w * grid_shape_w - grid_w
                ty = y / img_h * grid_shape_h - grid_h
                tw = np.log(w / anchors[layer][anchor_index][0])
                th = np.log(w / anchors[layer][anchor_index][1])

                y_true[layer][i, grid_h, grid_w, anchor_index, :4] = np.array([tx, ty, tw, th])
                y_true[layer][i, grid_h, grid_w, anchor_index, 4] = 1
                y_true[layer][i, grid_h, grid_w, anchor_index, 5 + class_index] = 1
            count = count + 1
            if count == lines_len:
                shuffle(lines)
                count = 0
        # print('time use: {}'.format(time.time() - begin))
        # with tf.Session() as sess:
        #     boxes = tf.boolean_mask(y_true[0][...,:5], y_true[0][..., 4])
        #     a = sess.run(boxes)
        #     m = K.shape(boxes)
        #     mf = K.cast(m, K.dtype(boxes))
        #     print(a)
        # y_true_reshape = [np.reshape(item, newshape=[batch_size, item.shape[1], item.shape[2], -1]) for item in y_true]
        yield [x_img, *y_true], np.zeros(batch_size)




if __name__ == '__main__':
    img_path = r'D:\data\pascal-voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg'
    # handle_img(img_path)

    data_gen = data_generate(batch_size=2)
    step =1
    for item in data_gen:
        print(item)
        step += 1
        if step == 3:
            break
