# @Time    : 2019/1/19 11:05
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : test.py
import os
import numpy as np
from PIL import ImageDraw, Image
from skimage import io
from config import data_test, grid_shape, num_classes, grid_cell_shape, input_shape, result_dir, weights_path
from yolov3_model import yolo_body, resize_img


def draw_box(y_pred, img):
    io.imsave(fname='tmp.jpg', arr=img)
    image = Image.open('tmp.jpg')
    draw = ImageDraw.Draw(image)
    out = y_pred
    # out = np.reshape(y_pred, newshape=(1,  grid_shape[0], grid_shape[1], 5 + num_classes))
    for i in range(out.shape[1]):
        for j in range(out.shape[2]):
            print(out[0, i, j, 4],out[0, i, j, 2],out[0, i, j, 3])
            if out[0, i, j, 4] > 0.2:
                print(out[0, i, j, 4],out[0, i, j, 2],out[0, i, j, 3],11111111111111111111111111111)
                x = (out[0, i, j, 0] + j) * grid_cell_shape[1]
                y = (out[0, i, j, 1] + i) * grid_cell_shape[0]
                w = out[0, i, j, 2] * input_shape[1]
                h = out[0, i, j, 3] * input_shape[0]
                box_x1 = x - w / 2
                box_y1 = y - h / 2
                box_x2 = x + w / 2
                box_y2 = y + h / 2
                draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline='red')
    return image


def predict():
    model = yolo_body()
    model.load_weights(weights_path)
    imgs = list()
    for i in range(1):
        img_path = r'D:\project\data-set\test\挨络替糙骡_1547711365.jpg'
        img = resize_img(img_path)
        imgs.append(img)
        # img_expand = np.expand_dims(img, axis=0)
    imgs = np.array(imgs)
    y_pred = model.predict(x=imgs)
    print(y_pred.shape)
    # 给图片画box
    img_box = draw_box(y_pred, img)
    img_name = os.path.split(img_path)[-1]
    print(os.path.join(result_dir, img_name), img_name)
    img_box.save(os.path.join(result_dir, img_name))
    # io.imsave(fname=r'D:\project\data-set\result\p2.jpg', arr=img_box)


if __name__ == '__main__':
    predict()