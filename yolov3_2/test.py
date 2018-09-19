# -*- coding:utf-8 -*-
"""
File Name: test
Version:
Description:
Author: liuxuewen
Date: 2018/8/8 16:51
"""
import colorsys

import numpy as np
import time
from PIL import Image, ImageFont, ImageDraw
import os

from keras import Input
from keras.engine.saving import load_model
from keras.utils import multi_gpu_model
from keras import backend as K
from utils.data_handle import letterbox_image, get_classes, get_anchors
from yolov3_model import yolo_body, yolo_eval


class YOLO(object):
    def __init__(self, **kwargs):
        self.classes_path = 'data/classes.txt'
        self.anchors_path = 'data/yolo_anchors.txt'
        self.model_path =  'model_dir/trained_weights_stage_3.h5'
        self.gpu_num = 1
        self.model_image_size = (416, 416)
        self.score = 0.3
        self.iou = 0.45
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def generate(self):
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        try:
            self.yolo_model = load_model(self.model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match

        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = time.time()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/cambria.ttc',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = time.time()
        print(end - start)
        return image

    def close_session(self):
        self.sess.close()



def run_detect_img(yolo,img_path, out_path):
    imgs = os.listdir(path=img_path)[:10]
    for img in imgs:
        img = os.path.join(img_path,img)
        img_name = os.path.split(img)[-1]
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            result_image = yolo.detect_image(image)
            img_out_path = os.path.join(out_path,'result_{}'.format(img_name))
            result_image.save(img_out_path)

    yolo.close_session()


if __name__ == '__main__':
    img_path = os.path.join(r'D:\tmp\tmp\keras-yolo3-master','utils','img_hanzi')
    out_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'detect_result')
    yolo = YOLO()
    run_detect_img(yolo, img_path, out_path)
