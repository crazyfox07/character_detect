# -*- coding:utf-8 -*-
"""
File Name: train
Version:
Description:
Author: liuxuewen
Date: 2018/8/8 16:51
"""
import os
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam

from utils.data_handle import get_classes, get_anchors, data_generator
from yolov3_model import create_model


def train_model():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_train_path = os.path.join(current_dir, 'data', 'data_train.txt')
    model_dir = os.path.join(current_dir, 'model_dir')
    classes_path = os.path.join(current_dir, 'data', 'classes.txt')
    anchors_path = os.path.join(current_dir, 'data', 'yolo_anchors.txt')
    weights_path = os.path.join(model_dir, 'trained_weights_stage_3.h5')
    os.makedirs(model_dir, exist_ok=True)
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    # 图片输入模型的大小hw，是32的倍数
    input_shape = (416, 416)
    model = create_model(input_shape, anchors, num_classes, load_pretrained=False,
                         freeze_body=2, weights_path=weights_path)

    logging = TensorBoard(log_dir=model_dir)
    checkpoint = ModelCheckpoint(model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1)

    with open(data_train_path, encoding='utf8') as f:
        lines = f.readlines()
    test_split = 0.1
    np.random.shuffle(lines)
    num_test = int(len(lines) * test_split)
    num_train = len(lines) - num_test

    model.compile(optimizer=Adam(lr=1e-3),
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    batch_size = 8
    print('Train on {} samples, test on {} samples, with batch size {}.'.format(
        num_train, num_test, batch_size))

    model.fit_generator(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors,
                                                       num_classes),
                        validation_steps=max(1, num_test // batch_size),
                        epochs=2,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    model.save_weights(os.path.join(model_dir, 'trained_weights_stage_3.h5'))
    # model.save(model_dir + 'trained_weights_stage_2.h5')


if __name__ == '__main__':
    train_model()
