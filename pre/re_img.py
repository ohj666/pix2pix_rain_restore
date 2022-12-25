import os

import cv2
#处理后的路径
result_path="/media/glint/F18731345B579A69/data/data/black_men/my_data"
#要处理的图像
base_path="/media/glint/F18731345B579A69/data/data/black_men/my_data"
#把图片裁剪成长宽一致
for i in os.listdir(base_path):
    img_name=os.path.join(base_path,i)
    img=cv2.imread(img_name)
    print(i)
    s=min(img.shape[:-1])

    cv2.imwrite(img_name,img[:s,:s])