# @Time    : 2019/1/18 16:50
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : config.py
import os
import numpy as np
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'model_dir')
os.makedirs(model_dir, exist_ok=True)
data_train_path = os.path.join(current_dir, 'data', 'data_train.txt')
with open(data_train_path, encoding='utf8') as f:
    lines = f.readlines()
test_split = 0.1
np.random.shuffle(lines)
num_test = int(len(lines) * test_split)
num_train = len(lines) - num_test
data_train = lines[: num_train]
data_test = lines[num_train:]
batch_size = 8
num_classes = 1
logging = TensorBoard(log_dir=model_dir)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
weights_path = os.path.join(model_dir, 'trained_weights.h5')

checkpoint = ModelCheckpoint(model_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

input_shape = np.array([416, 416]).astype('int')
HEIGHT, WIDTH = 416, 416
grid_cell_shape = np.array([16, 16]).astype('int')
grid_shape = input_shape // grid_cell_shape  # (h=26, w=26)

# 结果存储
result_dir = os.path.join(current_dir, 'result')
os.makedirs(result_dir, exist_ok=True)
