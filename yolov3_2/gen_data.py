# @Time    : 2019/1/15 17:05
# @Author  : lxw
# @Email   : liuxuewen@inspur.com
# @File    : gen_data.py
import imageio as imageio
from PIL import Image, ImageFont, ImageDraw
import os
from scipy import misc
import random
import time

imageio

current_dir = os.path.dirname(os.path.realpath(__file__))


# 读取汉字集
def read_hanzi():
    with open(os.path.join(current_dir, 'data', 'common_hanzi.txt'), encoding='utf8') as f:
        text = f.read().replace(' ', '')
        return text


# 将图片分为8*8个格子，返回这些格子左上角的坐标
def get_xy_list():
    grid = 8
    side = 344 // grid
    xy = list()
    for x in range(1, grid - 1):
        for y in range(1, grid - 1):
            xy.append((x * side, y * side))
    return xy


# 将文字写入图片
def draw_text():
    # 训练集图片的存储路径
    out_path = os.path.join(r'D:\project\data-set', 'train')
    os.makedirs(out_path, exist_ok=True)
    # 训练集的label，格式path/to/img1.jpg 50,100,150,200,0 30,50,200,120,0
    # Box format：x_min,y_min,x_max,y_max,class_id (no space)
    train_file_name = os.path.join(current_dir, 'data', 'data_train.txt')
    train_file = open(train_file_name, mode='a', encoding='utf8')
    # 获取格子左上角的坐标
    xy = get_xy_list()
    # 获取所有汉字
    hanzi = read_hanzi()
    # 生成1000张图片
    for _ in range(1000):
        # 图片中文字的长度
        text_len = random.randint(2, 6)
        # 从汉字集中随机选取text_len个汉字
        text = random.sample(hanzi, text_len)
        # 根据选取的汉字作为图片的标签
        img_name = '{}_{}.jpg'.format(''.join(text), str(int(time.time())))
        # 要生成图片的路径
        img_path = os.path.join(out_path, img_name)
        # 将图片路径写入到data_train.txt中
        train_file.write(img_path)
        # 生成空白图像
        im = Image.new("RGB", (344, 344), color=(255, 255, 255))
        # 随机选择text_len个格子的左上角坐标，作为每个文字的起始写入点
        xy_sample = random.sample(xy, text_len)
        # 绘图句柄
        draw = ImageDraw.Draw(im)

        for xy_, ch in zip(xy_sample, text):
            x, y = xy_
            # 使用自定义的字体，第二个参数表示字符大小
            font = ImageFont.truetype(r'C:\Windows\Fonts\simfang.ttf',
                                      size=random.randint(20, 50))
            # 将ch从(x,y)处写入图片
            draw.text((x, y), ch, font=font, fill=(0, 0, 0))
            # 获得文字的相对于(x,y)的偏移(offset)位置
            offsetx, offsety = font.getoffset(ch)
            # 获得文字的大小
            width, height = font.getsize(ch)
            # 绘出矩形框
            # draw.rectangle((offsetx+x,offsety+y,offsetx+x+width,offsety+y+height),outline=(255,0,0))
            # 将文字的坐标和类别（x_min,y_min,x_max,y_max,class_id写入到文件中，注意在开始加个空格，用来区分文字
            train_file.write(
                ' {},{},{},{},{}'.format(offsetx + x, offsety + y, offsetx + x + width, offsety + y + height, 0))
            print(' {},{},{},{},{}'.format(offsetx + x, offsety + y, offsetx + x + width, offsety + y + height, 0))
        # 一个图片的label完成，换行，准备写下个图片的label
        train_file.write('\n')
        # 保存生成的图片
        misc.imsave(img_path, im)

if __name__ == '__main__':
    draw_text()
