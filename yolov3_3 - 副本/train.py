# @Time    : 2019/1/18 15:16
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : train.py
import os

from keras.optimizers import Adam
from keras import backend as K, losses

from config import weights_path, data_train, num_train, batch_size, data_test, num_test, logging, reduce_lr, \
    early_stopping, model_dir, checkpoint, grid_shape, num_classes
from yolov3_model import yolo_body, data_generator
import numpy as np


def my_loss(y_true, y_pred, e=0.1):
    # print(K.shape(y_true), K.shape(y_pred))
    # y_true = K.reshape(y_true, (-1, grid_shape[0], grid_shape[1], (5 + num_classes)))
    # y_pred = K.reshape(y_pred, (-1, grid_shape[0], grid_shape[1], (5 + num_classes)))
    object_mask = y_true[..., 4:5]
    object_mask_bool = K.cast(object_mask, 'bool')
    m = K.shape(y_pred[0])[0]  # batch size, tensor
    mf = K.cast(m, K.dtype(y_pred[0]))

    xy_loss = K.sum(K.square(object_mask*(y_pred[..., 0:2])-object_mask*y_true[..., 0:2]))
    wh_loss =K.sum( K.square(object_mask * (y_pred[..., 2:4]) - object_mask * y_true[..., 2:4]))
    confidence_loss = K.sum(K.square(y_pred[..., 4:5] - y_true[..., 4:5]))
    # xy_loss = K.sum(y_pred[..., 4:5]) / mf
    # wh_loss = K.sum(object_mask * K.binary_crossentropy(target=y_pred[..., 2:4], output=y_true[..., 2:4],
    #                                                     from_logits=True)) / mf
    # confidence_loss = K.sum(K.binary_crossentropy(target=y_pred[..., 4:5], output=y_true[..., 4:5],
    #                                               from_logits=True)) / mf
    # category_loss = K.sum(
    # object_mask * K.binary_crossentropy(target=K.sigmoid(y_pred[..., 5:]), output=y_true[..., 5:],
    #                                     from_logits=True)) / mf
    loss = xy_loss + wh_loss + confidence_loss

    loss = loss / K.cast(K.shape(y_true)[0], dtype=K.dtype(y_true[0]))
    return loss


def train():
    model = yolo_body()
    if os.path.exists(weights_path):
        print('+++++++++++++++++++++++++++++++++++++++++++++++++')
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    model.compile(optimizer=Adam(lr=1e-3),
                  loss=my_loss)

    model.fit_generator(generator=data_generator(data_train, batch_size=batch_size),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(data_test, batch_size=batch_size),
                        validation_steps=max(1, num_test // batch_size),
                        epochs=2,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping]
                        )
    model.save_weights(weights_path)


if __name__ == '__main__':
    train()
