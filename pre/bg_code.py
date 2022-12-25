import math
import random

def t_dig(s):
    r=""
    for i in s:
        if i.isdigit():
            r+=i
    return int(r)

def bg2_code(img_jpg):
    with open("/home/glint/ruanjiang/data/shujuwajue/mycode/over/pre/data_txt/bg1.txt", "r") as fp:
        content=fp.read()
    index_ls=content.split(";")
    for i in index_ls:
        i1,i2=i.split(",")
        i1=t_dig(i1)
        i2=t_dig(i2)
        img_jpg=enlarge_effect(i1*2,i2*2,img_jpg,12)
    return img_jpg


def bg3_code(img_jpg):
    for i in range(10,256,30):
        for j in range(10,256,30):
            img_jpg=enlarge_effect(i,j,img_jpg,4)
    return img_jpg

def bg4_code(img_jpg):
    for i in range(10,256,40):
        for j in range(10,256,40):
            img_jpg=enlarge_effect(i,j,img_jpg,7)
    return img_jpg



def bg6_code(img_jpg):
    for i in range(10, 256, 45):
        for j in range(10, 256, 40):
            img_jpg = enlarge_effect(i, j, img_jpg, random.randint(4,10))
    return img_jpg

def bg7_code(img_jpg):
    for i in range(10, 256, 35):
        for j in range(10, 256, 35):
            img_jpg = enlarge_effect(i, j, img_jpg, random.randint(5,7 ))
    return img_jpg


def enlarge_effect(w, h, img,r):
    # h, w, n = img.shape
    cx = w / 2
    cy = h / 2
    radius = r
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