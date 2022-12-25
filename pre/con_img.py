import numpy as np
import cv2
import os
img_data = "/media/glint/F18731345B579A69/data/data/black_men/my_data"
result_data = "/media/glint/F18731345B579A69/data/data/black_men/traing_data/end_data"
src_imgs=[]
for i in os.listdir(img_data):
    src_img_path=os.path.join(img_data,i)
    rain_img_path=os.path.join(result_data,i)
    rain_img=cv2.imread(rain_img_path)
    if rain_img.shape[1]==512:
        continue
    src_img=cv2.imread(src_img_path)
    src_img=cv2.resize(src_img,(256,256))

    result=np.concatenate([src_img,rain_img],axis=1)
    cv2.imwrite(os.path.join(result_data,i),result)




