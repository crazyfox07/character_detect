# @Time    : 2019/4/5 15:36
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : yolov3_config.py
import os

current_dir = os.path.split(os.path.abspath(__file__))[0]
img_w = 416
img_h = 416
img_c = 3
img_dir_train = r'D:\data\pascal-voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
img_dir_test = r''
imgname_label_txt = os.path.join(current_dir, 'imgname-label.txt')
classes = {'diningtable': 0, 'motorbike': 1, 'train': 2, 'sofa': 3, 'aeroplane': 4, 'sheep': 5, 'boat': 6,
           'pottedplant': 7, 'bicycle': 8, 'bird': 9, 'horse': 10, 'chair': 11, 'tvmonitor': 12, 'car': 13,
           'bus': 14, 'bottle': 15, 'cow': 16, 'person': 17, 'cat': 18, 'dog': 19}
num_classes = 20

anchors = [[(202, 199), (190, 109), (133, 194)],
           [(114, 129), (93, 68), (69, 175)],
           [(48, 114), (34, 63), (18, 24)]]
num_anchors_per_layer = 3
num_layers = len(anchors) // num_anchors_per_layer
grid_shape = [(13, 13), (26, 26), (52, 52)]
boxes_max_num = 20
train_num = 17125
batch_size = 32
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
model_name = 'trained_weights_stage_1.h5'
