# @Time    : 2019/4/5 14:41
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : model.py
from functools import reduce, wraps
from keras import backend as K, Model
from keras import Input
from keras.layers import Lambda, Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add, UpSampling2D, Concatenate, \
    MaxPooling2D
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
from yolov3_config import grid_shape, num_layers, batch_size, threshold, num_anchors_per_layer
from yolov3_config import img_h, img_w, img_c, anchors, classes, num_classes


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


def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
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
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        DarknetConv2D_BN_Leaky(1024, (3, 3)),
        DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
        DarknetConv2D_BN_Leaky(512, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        DarknetConv2D_BN_Leaky(256, (3, 3)),
        DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def xywh_to_raw(xywh, grid_shape_h, grid_shape_w, layer):
    grid_y = tf.tile(input=tf.reshape(tf.range(grid_shape_h), shape=[-1, 1, 1, 1]),
                     multiples=[1, grid_shape_w, 1, 1])
    grid_x = tf.tile(input=tf.reshape(tf.range(grid_shape_w), shape=[1, -1, 1, 1]),
                     multiples=[grid_shape_h, 1, 1, 1])
    grid = tf.concat([grid_x, grid_y], axis=-1)
    grid = tf.cast(grid, dtype=xywh.dtype)
    box_x = (xywh[..., 0:1] + grid[..., 0:1]) * img_w / grid_shape_w
    box_y = (xywh[..., 1:2] + grid[..., 1:2]) * img_h / grid_shape_h
    box_w = tf.exp(xywh[..., 2:3]) * np.array(anchors[layer])[:, 0:1]
    box_h = tf.exp(xywh[..., 3:4]) * np.array(anchors[layer])[:, 1:2]
    box_raw = tf.concat([box_x, box_y, box_w, box_h], axis=-1)
    return box_raw


def get_ignore_mask(y_true_xywh, y_pred_xywh, object_mask, grid_shape_h, grid_shape_w, layer):
    result = list()
    y_pred_xywh_raw = xywh_to_raw(y_pred_xywh, grid_shape_h, grid_shape_w, layer)
    y_true_xywh_raw = xywh_to_raw(y_true_xywh, grid_shape_h, grid_shape_w, layer) * object_mask
    for b in range(batch_size):
        true_boxes = tf.boolean_mask(y_true_xywh_raw[b], object_mask[b, ..., 0] > 0)
        b_y_true_x = true_boxes[..., 0]
        b_y_true_y = true_boxes[..., 1]
        b_y_true_w = true_boxes[..., 2]
        b_y_true_h = true_boxes[..., 3]

        b_y_pred_x = y_pred_xywh_raw[b, ..., 0:1]
        b_y_pred_y = y_pred_xywh_raw[b, ..., 1:2]
        b_y_pred_w = y_pred_xywh_raw[b, ..., 2:3]
        b_y_pred_h = y_pred_xywh_raw[b, ..., 3:4]
        xmin = tf.maximum(b_y_true_x - b_y_true_w / 2, b_y_pred_x - b_y_pred_w / 2)
        ymin = tf.maximum(b_y_true_y - b_y_true_h / 2, b_y_pred_y - b_y_pred_h / 2)
        xmax = tf.minimum(b_y_true_x + b_y_true_w / 2, b_y_pred_x + b_y_pred_w / 2)
        ymax = tf.minimum(b_y_true_y + b_y_true_h / 2, b_y_pred_y + b_y_pred_h / 2)
        interact_area = (xmax - xmin) * (ymax - ymin)
        union_area = b_y_pred_w * b_y_pred_h + b_y_true_w * b_y_true_h - interact_area
        iou = interact_area / union_area
        iou_max = tf.reduce_max(iou, axis=-1)
        iou_max = tf.expand_dims(iou_max, axis=-1)  # shape=(grid_shape_h, grid_shape_w, 3, 1)
        iou_mask = tf.where(iou_max > threshold, tf.ones_like(iou_max), tf.zeros_like(iou_max))
        result.append(iou_mask)
    result_stack = tf.stack(result, axis=0)  # shape=(batch_size, grid_shape_h, grid_shape_w, 3, 1)
    return result_stack


def create_model(num_anchors, num_classes):
    img_input = Input(shape=(img_h, img_w, img_c))
    model_body = tiny_yolo_body(img_input, num_anchors, num_classes)
    # model_body = yolo_body(img_input, num_anchors, num_classes)
    y_true = [Input(shape=(shape[0], shape[1], num_anchors, 5 + num_classes)) for shape in grid_shape]

    loss_layer = Lambda(function=yolo_loss, output_shape=(1,), name='yolo-loss')([*model_body.output, *y_true])
    new_model = Model(inputs=[model_body.input, *y_true], outputs=[loss_layer])
    return new_model


def yolo_loss(args):
    y_pred = args[: num_layers]
    y_true = args[num_layers:]
    loss = 0
    for layer in range(num_layers):
        grid_shape_h, grid_shape_w = grid_shape[layer]
        y_true_xy = y_true[layer][..., 0:2]
        y_true_wh = y_true[layer][..., 2:4]
        object_mask = y_true[layer][..., 4:5]

        y_pred_reshape = tf.reshape(y_pred[layer],
                                    shape=(-1, grid_shape_h, grid_shape_w, num_anchors_per_layer, 5 + num_classes))
        y_pred_xy = tf.sigmoid(y_pred_reshape[..., 0:2])
        y_pred_wh = y_pred_reshape[..., 2:4]
        y_pred_xywh = tf.concat([y_pred_xy, y_pred_wh], axis=-1)

        y_pred_confidence = tf.sigmoid(y_pred_reshape[..., 4:5])
        y_pred_class = tf.sigmoid(y_pred_reshape[..., 5:])

        # 加入ignore_mask主要是为了平衡正负样本的比例，一张图中object的数量相对背景的数量太少了
        ignore_mask = get_ignore_mask(y_true[layer][..., 0: 4], y_pred_xywh, object_mask, grid_shape_h, grid_shape_w,
                                      layer)
        # box_loss_scale = 2 - tf.exp(y_true_wh[..., 0:1]) * tf.exp(y_true_wh[..., 1:2])
        # xy_loss = object_mask * K.binary_crossentropy(y_true_xy, y_pred_xy, from_logits=True)
        xy_loss = object_mask * K.square(y_true_xy - y_pred_xy)
        wh_loss = object_mask * K.square(y_true_wh - y_pred_wh)
        # confidence_loss = object_mask * K.binary_crossentropy(object_mask, y_pred_confidence, from_logits=True) + \
        #                   (1 - object_mask) * K.binary_crossentropy(object_mask, y_pred_confidence, from_logits=True) *ignore_mask
        confidence_loss = object_mask * K.square(object_mask - y_pred_confidence) + (1 - object_mask) * K.square(
            object_mask - y_pred_confidence) * ignore_mask
        # class_loss = object_mask * K.binary_crossentropy(y_true[layer][..., 5:],y_pred_class, from_logits=True)
        class_loss = object_mask * K.square(y_true[layer][..., 5:] - y_pred_class)

        xy_loss = tf.reduce_sum(xy_loss) / batch_size
        wh_loss = tf.reduce_sum(wh_loss) / batch_size
        confidence_loss = tf.reduce_sum(confidence_loss) / batch_size
        class_loss = tf.reduce_sum(class_loss) / batch_size

        loss = loss + xy_loss + wh_loss + confidence_loss
    return loss


if __name__ == '__main__':
    pass
    # num_classes = len(classes)
    # inp = Input(shape=(img_h, img_w, img_c))
    # model = tiny_yolo_body(num_anchors_per_layer, num_classes)
    # model.summary()
