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
from yolov3_config import grid_shape


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


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(256, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
        DarknetConv2D_BN_Leaky(128, (1, 1)),
        UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    return Model(inputs, [y1, y2, y3])


def box_iou():
    pass


def get_ignore_mask(y_true_boxes, y_pred_boxes, object_mask, grid_shape_index):
    true_boxes = tf.boolean_mask(y_true_boxes, object_mask[0])
    true_boxes[..., 0:2] = true_boxes[..., 0:2] * grid_shape_index[::-1]


def yolo_loss(y_true, y_pred):
    num_layers = len(anchors) // 3
    for layer in range(num_layers):
        grid_shape_index = grid_shape[layer]
        y_pred_reshape = K.reshape(y_pred[layer], shape=[-1, grid_shape_index[0], grid_shape_index[1], num_anchors // 3,
                                                         num_classes + 5])
        true_xy = y_true[layer][..., 0:2]
        true_wh = y_true[layer][..., 2:4]

        pred_xy = y_pred_reshape[..., 0:2]
        pred_wh = y_pred_reshape[..., 2:4]
        pred_confidence = y_pred_reshape[..., 4:5]

        object_mask = y_true[layer][..., 4:5]
        # 加入ignore_mask主要是为了平衡正负样本的比例，一张图中object的数量相对背景的数量太少了
        ignore_mask = get_ignore_mask(y_true[layer][..., 0: 4], y_pred_reshape[..., 0: 4], object_mask, grid_shape_index)

        xy_loss = object_mask * K.binary_crossentropy(true_xy, pred_xy, from_logits=True)
        wh_loss = object_mask * 0.5 * K.square(true_wh - pred_wh)
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, pred_confidence, from_logits=True) + \
                          1


if __name__ == '__main__':
    from yolov3_config import img_h, img_w, img_c, anchors, classes

    num_anchors = len(anchors) // 3
    num_classes = len(classes)
    inp = Input(shape=(img_h, img_w, img_c))
    model = yolo_body(inp, num_anchors, num_classes)
    model.summary()
