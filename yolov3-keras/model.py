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
from yolov3_config import grid_shape, num_layers
from yolov3_config import img_h, img_w, img_c, anchors, classes, num_classes, num_anchors


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
    return Model(inputs=[inputs], outputs=[y1, y2, y3])




def get_ignore_mask(y_true_boxes, y_pred_boxes,grid_shape_h, grid_shape_w):
    # true_boxes = tf.boolean_mask(y_true_boxes, object_mask[0])
    # true_boxes[..., 0:2] = true_boxes[..., 0:2] * grid_shape_index[::-1]
    batch_size = 32
    layer = grid_shape_h // 26
    result = tf.zeros(shape=(batch_size, grid_shape_h, grid_shape_w, 3, 1))
    for b in range(batch_size):
        # m = tf.shape(b_true_boxes_indexes)[0]
        true_boxes = tf.boolean_mask(y_true_boxes[b], y_true_boxes[b, ..., 4]>0)
        true_boxes_x_index = true_boxes[..., 0] // grid_shape_w
        true_boxes_y_index = true_boxes[..., 1] // grid_shape_h

        true_boxes[..., 2] = tf.exp(true_boxes[..., 2]) #* anchors_arr[layer, anchor_index]
        true_boxes[..., 3] = tf.exp(true_boxes[..., 3])# * grid_shape_w

        b_y_true_x = true_boxes[..., 0]
        b_y_true_y = true_boxes[..., 1]
        b_y_true_w = true_boxes[..., 2]
        b_y_true_h = true_boxes[..., 3]
        b_y_pred_boxes = tf.expand_dims(y_pred_boxes[b], axis=-2)
        b_y_pred_x = b_y_pred_boxes[..., 0]
        b_y_pred_y = b_y_pred_boxes[..., 1]
        b_y_pred_w = b_y_pred_boxes[..., 2]
        b_y_pred_h = b_y_pred_boxes[..., 3]
        xmin = tf.maximum(b_y_true_x - b_y_true_w / 2, b_y_pred_x - b_y_pred_w / 2)
        ymin = tf.maximum(b_y_true_y - b_y_true_h / 2, b_y_pred_y - b_y_pred_h / 2)
        xmax = tf.minimum(b_y_true_x + b_y_true_w / 2, b_y_pred_x + b_y_pred_w / 2)
        ymax = tf.minimum(b_y_true_y + b_y_true_h / 2, b_y_pred_y + b_y_pred_h / 2)
        interact_area = (xmax - xmin) * (ymax - ymin)
        union_area = b_y_pred_w * b_y_pred_h + b_y_true_w * b_y_true_h - interact_area
        iou = interact_area / union_area
        iou_max = tf.reduce_max(iou, axis=-1)
        iou_max = tf.expand_dims(iou_max, axis=-1)
        iou_mask = tf.where(iou_max > 0.5, tf.ones_like(iou_max), tf.zeros_like(iou_max))
        result[b] = iou_mask
    return result


def create_model(num_anchors, num_classes):
    img_input = Input(shape=(img_h, img_w, img_c))
    model_body = yolo_body(img_input, num_anchors, num_classes)

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





    batch_size, grid_shape_h, grid_shape_w = K.shape(y_pred)[0], K.shape(y_pred)[1], K.shape(y_pred)[2]  # batch size, tensor
    # batch_size = K.cast(batch_size, K.dtype(y_pred[0]))
    loss = 0
    y_true = tf.reshape(y_true, shape=[batch_size, grid_shape_h, grid_shape_w, num_anchors // 3, num_classes + 5])
    y_pred_reshape = tf.reshape(y_pred, shape=[batch_size, grid_shape_h, grid_shape_w, num_anchors // 3, num_classes + 5])
    true_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    pred_xy = y_pred_reshape[..., 0:2]
    pred_wh = y_pred_reshape[..., 2:4]
    pred_confidence = y_pred_reshape[..., 4:5]

    object_mask = y_true[..., 4:5]
    # 加入ignore_mask主要是为了平衡正负样本的比例，一张图中object的数量相对背景的数量太少了
    ignore_mask = get_ignore_mask(y_true[..., 0: 5], y_pred_reshape[..., 0: 4], grid_shape_h, grid_shape_w)

    box_loss_scale = 2 - y_true[..., 2:3] * y_true[..., 3:4]

    xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(true_xy, pred_xy, from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * K.square(true_wh - pred_wh)
    confidence_loss = object_mask * K.binary_crossentropy(object_mask, pred_confidence, from_logits=True) + \
                      (1 - object_mask) * ignore_mask * K.binary_crossentropy(object_mask, pred_confidence,
                                                                              from_logits=True)
    class_loss = object_mask * K.binary_crossentropy(y_true[..., 5:], y_pred_reshape[..., 5:], from_logits=True)

    xy_loss = tf.reduce_sum(xy_loss) / batch_size
    wh_loss = tf.reduce_sum(wh_loss) / batch_size
    confidence_loss = tf.reduce_sum(confidence_loss) / batch_size
    class_loss = tf.reduce_sum(class_loss) / batch_size

    loss = loss + xy_loss + wh_loss + confidence_loss + class_loss
    return loss


def yolo_loss_bak(y_true, y_pred):
    num_layers = len(anchors) // 3
    batch_size = K.shape(y_pred[0])[0]  # batch size, tensor
    # batch_size = K.cast(batch_size, K.dtype(y_pred[0]))
    loss = 0
    for layer in range(num_layers):
        grid_shape_index = grid_shape[layer]
        grid = np.zeros(shape=(grid_shape_index[0], grid_shape_index[1], 2))
        # for i in range(grid_shape_index[0]):
        #     for j in range(grid_shape_index[1]):
        #         grid[]
        y_pred_reshape = tf.reshape(y_pred[layer],
                                    shape=[batch_size, grid_shape_index[0], grid_shape_index[1], num_anchors // 3,
                                           num_classes + 5])
        true_xy = y_true[layer][..., 0:2]
        true_wh = y_true[layer][..., 2:4]

        pred_xy = y_pred_reshape[..., 0:2]
        pred_wh = y_pred_reshape[..., 2:4]
        pred_confidence = y_pred_reshape[..., 4:5]

        object_mask = y_true[layer][..., 4:5]
        # 加入ignore_mask主要是为了平衡正负样本的比例，一张图中object的数量相对背景的数量太少了

        ignore_mask = get_ignore_mask(y_true[layer][..., 0: 5], y_pred_reshape[..., 0: 4], object_mask, layer)

        box_loss_scale = 2 - y_true[layer][..., 2:3] * y_true[layer][..., 3:4]

        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(true_xy, pred_xy, from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(true_wh - pred_wh)
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, pred_confidence, from_logits=True) + \
                          (1 - object_mask) * ignore_mask * K.binary_crossentropy(object_mask, pred_confidence,
                                                                                  from_logits=True)
        class_loss = object_mask * K.binary_crossentropy(y_true[layer][..., 5:], y_pred_reshape[..., 5:],
                                                         from_logits=True)

        xy_loss = tf.reduce_sum(xy_loss) / batch_size
        wh_loss = tf.reduce_sum(wh_loss) / batch_size
        confidence_loss = tf.reduce_sum(confidence_loss) / batch_size
        class_loss = tf.reduce_sum(class_loss) / batch_size

        loss = loss + xy_loss + wh_loss + confidence_loss + class_loss
    return loss


if __name__ == '__main__':
    num_classes = len(classes)
    inp = Input(shape=(img_h, img_w, img_c))
    model = yolo_body(num_anchors, num_classes)
    model.summary()
