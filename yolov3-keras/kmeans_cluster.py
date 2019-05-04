# @Time    : 2019/5/3 9:49
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : kmeans_cluster.py
import re
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw
import numpy as np
import os

from yolov3_config import imgname_label_txt, img_w, img_h, anchors, img_dir_train

#  得到样本中的box集和
def get_wh_set():
    boxes_wh = list()
    with open(imgname_label_txt) as fr:
        for line in fr:
            img_size_raw = re.findall('jpg,(\d+),(\d+)', line)
            img_w_raw, img_h_raw = int(img_size_raw[0][0]), int(img_size_raw[0][1])
            items = re.findall('\w+,(\d+),(\d+),(\d+),(\d+)', line)
            for item in items:
                w, h = int(item[2]) - int(item[0]), int(item[3]) - int(item[1])
                w_scale = w / img_w_raw * img_w
                h_scale = h / img_h_raw * img_h
                boxes_wh.append([w_scale, h_scale])
    return np.array(boxes_wh)


#  kmeans聚类
def get_anchors_by_kmeans():
    boxes_wh = get_wh_set()
    kmeans = KMeans(n_clusters=6, random_state=0).fit(boxes_wh)
    # kmeans.fit(boxes_wh)
    result = kmeans.cluster_centers_
    return result.astype('int32').tolist()


# 画出通过kmeans聚类得到的anchor
def draw_anchors():
    anchors_list = get_anchors_by_kmeans()
    anchors_list_sort = sorted(anchors_list, key=lambda item: item[0]*item[1], reverse=True)
    print(anchors_list_sort)
    new_img = Image.new('RGB', (img_w, img_h), '#00ff00')
    draw_obj = ImageDraw.Draw(new_img)

    for anchor in anchors_list_sort:
        x0 = int(img_w / 2 - anchor[0] / 2)
        y0 = int(img_h / 2 - anchor[1] / 2)
        x1 = int(img_w / 2 + anchor[0] / 2)
        y1 = int(img_h / 2 + anchor[1] / 2)

        draw_obj.rectangle(xy=(x0, y0, x1, y1), outline='#ff0000', width=2)
    new_img.save('D:\project\character_detect\p3.jpg')


# 给训练集中的图片话box
def draw_boxes():
    i = 0
    with open(imgname_label_txt) as fr:
        for line in fr:
            img_raw = re.findall('(.*\.jpg),(\d+),(\d+)', line)
            img_name, img_w_raw, img_h_raw = img_raw[0][0], int(img_raw[0][1]), int(img_raw[0][2])
            img_path = os.path.join(img_dir_train, img_name)
            img = Image.open(img_path)
            img = img.resize((img_w, img_h))
            draw_obj = ImageDraw.Draw(img)
            items = re.findall('\w+,(\d+),(\d+),(\d+),(\d+)', line)
            for item in items:
                w, h = int(item[2]) - int(item[0]), int(item[3]) - int(item[1])
                w_scale = w / img_w_raw * img_w
                h_scale = h / img_h_raw * img_h
                x0 = int(item[0]) / img_w_raw * img_w
                y0 = int(item[1]) / img_h_raw * img_h
                x1 = int(item[2]) / img_w_raw * img_w
                y1 = int(item[3]) / img_h_raw * img_h
                print('{}={}, {}={}'.format(x1-x0,w_scale, y1-y0,h_scale))
                draw_obj.rectangle(xy=(x0, y0, x1, y1), outline='#ff0000', width=2)
            img.save(r'D:\project\character_detect\img_with_box\{}'.format(img_name))
            if i == 10:
                break
            else:
                i += 1


if __name__ == '__main__':
    # get_anchors_by_kmeans()
    draw_anchors()
    # draw_boxes()