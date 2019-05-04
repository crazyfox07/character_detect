# @Time    : 2019/4/5 15:36
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : yolov3_config.py
import os
import platform


current_dir = os.path.split(os.path.abspath(__file__))[0]
img_w = 416
img_h = 416
img_c = 3
if platform.system() == 'Windows':
    img_dir_train = r'D:\data\pascal-voc\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\JPEGImages'
    log_dir = r'D:\logs\yolov3'

else:
    img_dir_train = '/home/lxw/data/JPEGImages'
    log_dir = '/home/lxw/logs/yolov3'
img_dir_test = r''
os.makedirs(log_dir, exist_ok=True)
imgname_label_txt = os.path.join(current_dir, 'imgname-label.txt')
classes = {'diningtable': 0, 'motorbike': 1, 'train': 2, 'sofa': 3, 'aeroplane': 4, 'sheep': 5, 'boat': 6,
           'pottedplant': 7, 'bicycle': 8, 'bird': 9, 'horse': 10, 'chair': 11, 'tvmonitor': 12, 'car': 13,
           'bus': 14, 'bottle': 15, 'cow': 16, 'person': 17, 'cat': 18, 'dog': 19}
num_classes = 1

anchors = [[[362, 360], [195, 347], [294, 190]],
           [[114, 250], [93, 138], [37, 54]]]
num_anchors_per_layer = 3
num_layers = len(anchors)
grid_shape = [(13, 13), (26, 26)]
boxes_max_num = 20
train_num = 17125
batch_size = 64

model_name = 'trained_weights_stage_2.h5'
threshold = 0.5
