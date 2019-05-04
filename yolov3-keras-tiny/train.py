# @Time    : 2019/4/7 10:49
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : train.py
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from model import create_model
from utils import data_generate
from yolov3_config import num_classes, batch_size, train_num, log_dir, num_anchors_per_layer, model_name
import os
import tensorflow as tf
import subprocess

def train_model():
    subprocess.call('rm -rf {}/*'.format(log_dir), shell=True)
    tf.logging.set_verbosity(tf.logging.INFO)
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val-loss:.3f}.h5',
                                 monitor='val-loss', save_weights_only=True, save_best_only=True, period=3)
    model = create_model(num_anchors_per_layer, num_classes)
    model.summary()

    model.compile(optimizer=Adam(lr=1e-3),
                  loss={'yolo-loss': lambda y_true, y_pred: y_pred})  # use custom yolo_loss Lambda layer.

    model.fit_generator(generator=data_generate(batch_size=batch_size),
                        steps_per_epoch=train_num // batch_size,
                        epochs=10,
                        initial_epoch=0,
                        callbacks=[logging])
    model.save_weights(os.path.join(log_dir, model_name))


if __name__ == '__main__':
    train_model()
