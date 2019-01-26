# -*- coding:utf-8 -*-
"""
File Name: model
Version:
Description:
Author: liuxuewen
Date: 2018/8/8 16:51
"""
import os
from functools import reduce, wraps

from PIL import Image
from keras import backend as K, Model
from keras import Input
from keras.activations import sigmoid
from keras.layers import Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add, UpSampling2D, Concatenate, \
    MaxPooling2D, Flatten, Dense
from keras.regularizers import l2
import tensorflow as tf
import numpy as np

from config import HEIGHT, WIDTH, input_shape, num_classes, grid_cell_shape, grid_shape, data_train


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            DarknetConv2D_BN_Leaky(num_filters // 2, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
        DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
        DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body2():
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(shape=(HEIGHT, WIDTH, 3))
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        # MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        # DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    # y = compose(
    #     DarknetConv2D_BN_Leaky(512, (3, 3)),
    #     DarknetConv2D(num_classes + 5, (1, 1)),
    #     UpSampling2D(2))(x2)
    y = Conv2D(filters=num_classes + 5, kernel_size=(1,1))(x1)

    # y = Dense(grid_shape[0] * grid_shape[1] * (5 + num_classes), activation='sigmoid')(y)
    # y = Reshape(target_shape=(grid_shape[0], grid_shape[1], (5 + num_classes)))(y)

    return Model(inputs=inputs, outputs=y)

def yolo_body():
    """Create YOLO_V3 model CNN body in Keras."""
    inputs = Input(shape=(HEIGHT, WIDTH, 3))
    x1 = compose(
        DarknetConv2D_BN_Leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        DarknetConv2D_BN_Leaky(256, (3, 3)))(inputs)

    y1 = Conv2D(filters=2, kernel_size=(1, 1), activation='sigmoid', name='output1')(x1)
    y2 = Conv2D(filters=2, kernel_size=(1, 1), activation='sigmoid', name='output2')(x1)
    y3 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', name='output3')(x1)
    y4 = Conv2D(filters=num_classes, kernel_size=(1, 1), activation='sigmoid', name='output4')(x1)

    # y = Dense(grid_shape[0] * grid_shape[1] * (5 + num_classes), activation='sigmoid')(y)
    # y = Reshape(target_shape=(grid_shape[0], grid_shape[1], (5 + num_classes)))(y)

    return Model(inputs=[inputs], outputs=[y1, y2, y3, y4])

inputs = Input(shape=(HEIGHT, WIDTH, 3))


def resize_img(img_path):
    img = Image.open(img_path)
    iw, ih = img.size  # (344, 344)
    h, w = input_shape  # (416, 416)

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image) / 255.
    return image_data

def get_random_data(data, max_boxes=20):
    '''random preprocessing for real-time data augmentation'''
    data_split = data.split()  # 纬投悍秋幽_1533628573.jpg 215,215,240,239,0 86,45,136,93,0 172,87,193,108,0 172,130,194,152,0 258,46,303,88,0
    img_path = data_split[0]
    img = Image.open(img_path)
    iw, ih = img.size  # (344, 344)
    h, w = input_shape  # (416, 416)
    box = np.array([list(map(lambda x: int(x), item.split(','))) for item in [box for box in data_split[1:]]])

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    image = img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image) / 255.

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box_data[:len(box)] = box

    return image_data, box_data


def preprocess_true_boxes(true_boxes):
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy
    true_boxes[..., 2:4] = boxes_wh

    b_s = true_boxes.shape[0]

    y_true = [np.zeros(shape=(b_s, grid_shape[0], grid_shape[1], 2), dtype='float32'),
              np.zeros(shape=(b_s, grid_shape[0], grid_shape[1], 2), dtype='float32'),
              np.zeros(shape=(b_s, grid_shape[0], grid_shape[1], 1), dtype='float32'),
              np.zeros(shape=(b_s, grid_shape[0], grid_shape[1], num_classes), dtype='float32')]
    for i in range(b_s):
        for j in range(true_boxes.shape[1]):
            # 判断box的w（宽度）是否大于0
            if true_boxes[i, j, 2] > 0:
                grid_y = int(true_boxes[i, j, 1] // grid_cell_shape[0])
                grid_x = int(true_boxes[i, j, 0] // grid_cell_shape[1])
                c = int(true_boxes[i, j, 4])
                y_true[1][i, grid_y, grid_x, 0:2] = true_boxes[i, j, 2:4] / (input_shape[::-1])
                y_true[0][i, grid_y, grid_x, 0] = true_boxes[i, j, 0] / grid_cell_shape[1] - grid_x
                y_true[0][i, grid_y, grid_x, 1] = true_boxes[i, j, 1] / grid_cell_shape[0] - grid_y
                y_true[2][i, grid_y, grid_x, 0] = 1  # 置信度，判断是否有box
                y_true[3][i, grid_y, grid_x, c] = 1
            else:
                continue
    # y_true = np.reshape(y_true, newshape=(b_s, grid_shape[0] * grid_shape[1] * (5 + num_classes)))
    return y_true


def data_generator(data, batch_size=32):
    '''data generator for fit_generator'''

    n = len(data)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(data)
            image, box = get_random_data(data[i])
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data)
        yield image_data, y_true


if __name__ == '__main__':
    for i in data_generator(data_train):
        print(i[0].shape)
