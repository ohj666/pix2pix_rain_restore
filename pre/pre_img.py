import os
import random

import cv2
import numpy as np
from mycode.over.pre.bg_code import *



def random_img_code(img):
    c=random.randint(1,5)
    if c==1:
        a_img=bg2_code(img)
        bg_path="/home/glint/ruanjiang/data/shujuwajue/mycode/over/pre/bg2.png"
    elif c==2:
        a_img=bg3_code(img)
        bg_path="/home/glint/ruanjiang/data/shujuwajue/mycode/over/pre/bg3.png"
    elif c==3:
        a_img=bg4_code(img)
        bg_path="/home/glint/ruanjiang/data/shujuwajue/mycode/over/pre/bg4.png"
    elif c==4:
        a_img=bg6_code(img)
        bg_path="/home/glint/ruanjiang/data/shujuwajue/mycode/over/pre/bg6.png"
    else:
        a_img=bg7_code(img)
        bg_path="/home/glint/ruanjiang/data/shujuwajue/mycode/over/pre/bg7.png"
    return bg_path,a_img

def enlarge_effect(w,h,img):
    # h, w, n = img.shape
    cx = w/2
    cy = h/2
    radius = 4
    r = int(radius / 2.0)
    new_img = img.copy()
    for i in range(w):
        for j in range(h):
            tx = i - cx
            ty = j - cy
            distance = tx * tx + ty * ty
            if distance < radius * radius:
                x = int(int(tx / 2.0) * (math.sqrt(distance) / r) + cx)
                y = int(int(ty / 2.0) * (math.sqrt(distance) / r) + cy)
                if x < w and y < h:
                    new_img[j, i] = img[y, x]

    return new_img

def add_alpha_channel(img):
    b_channel, g_channel, r_channel = cv2.split(img)  # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255  # 创建Alpha通道
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))  # 融合通道
    return img_new


def merge_img(jpg_img, png_img, y1, y2, x1, x2):

    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
    if x1 < 0:
        xx1 = -x1
        x1 = 0
        if y1 < 0:
            yy1 = - y1
        y1 = 0
        if x2 > jpg_img.shape[1]:
            xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
        if y2 > jpg_img.shape[0]:
            yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
        # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0

    np.set_printoptions()
    alpha_jpg = 1 - alpha_png

    # 开始叠加
    for c in range(0, 3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[yy1:yy2, xx1:xx2, c]))
    return jpg_img

if __name__ == '__main__':  # 定义图像路径
    img_data="/media/glint/F18731345B579A69/data/data/black_men/my_data"
    result_data="/media/glint/F18731345B579A69/data/data/black_men/end_data"
    for i in os.listdir(img_data):
        img_jpg_path = os.path.join(img_data,i)

        # 读取图像
        img_jpg = cv2.imread(img_jpg_path, cv2.IMREAD_UNCHANGED)
        # 设置叠加位置坐标
        img_jpg=cv2.resize(img_jpg,(256,256))

        img_png_path,img_jpg=random_img_code(img_jpg)
        img_png = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)
        img_png=cv2.resize(img_png,(256,256))
        print(i)
        # 开始叠加
        res_img = merge_img(img_jpg, img_png, 0, 256, 0, 256)
        # 显示结果图像
        # 保存结果图像，读者可自行修改文件路径
        cv2.imwrite(os.path.join(result_data,i), res_img)
        cv2.waitKey()
        cv2.destroyAllWindows()