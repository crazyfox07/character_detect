# @Time    : 2019/4/7 10:49
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : train.py
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from model import yolo_body, yolo_loss
from utils import data_generate
from yolov3_config import num_anchors, num_classes, batch_size, train_num, log_dir
import os

def train_model():
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    model = yolo_body(num_anchors, num_classes)

    model.compile(optimizer=Adam(lr=1e-3),
                  loss=[yolo_loss] * 3)

    model.fit_generator(generator=data_generate(batch_size=batch_size),
                        steps_per_epoch=train_num // batch_size,
                        epochs=30,
                        initial_epoch=0,
                        callbacks=[logging])
    model.save_weights(os.path.join(log_dir, 'train_weight.h5'))


if __name__ == '__main__':
    train_model()
