# from skimage.color.colorlabel import label2rgb

import gdal
import os
import numpy as np
import cv2
from PIL import Image

classid2rgb_map = {
    1: [0, 255, 255],  # "cyan", 土地
    2: [255, 0, 0],  # "red",
    3: [0, 255, 0],  # "green", 草地
    4: [255, 0, 255],  # "purple", 背景
    5: [0, 0, 255],  # "blue", 房子
    6: [255, 255, 0],  # "yello" 汽车
}

def label2rgb(pred_y):
    #print(set(list(pred_y.reshape(-1))))
    rgb_img = np.zeros((pred_y.shape[0], pred_y.shape[1], 3))
    for i in range(len(pred_y)):
        for j in range(len(pred_y[0])):
            rgb_img[i][j] = classid2rgb_map.get(pred_y[i][j], [255, 255, 255])
    return rgb_img.astype(np.uint8)

# tiff.imsave(dump_file_name, img)

tifpath='/home/zjh/AIDataset/gt'
tif = os.listdir(tifpath)
# change grey to rgb
# for file in tif:
#     if file.endswith('.tif'):
#         ds = gdal.Open(os.path.join(tifpath, file))
#         data = ds.ReadAsArray()
#         img = label2rgb(data)
#         png_name = '/home/zjh/AIDataset/gt_png/%s.png'%file[:-4]
#         #tiff.imsave(dump_file_name, img)
#         print(png_name)
#         cv2.imwrite(png_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

input='/home/zjh/AIDataset/gt'
output = '/home/zjh/LULC_CNIC_BAIDU/gt'
files = os.listdir(input)
files = list(filter(lambda x: x.endswith('.tif'), files))
for file in files:
# change tif to png
    ds = gdal.Open(os.path.join(input, file))

    data = ds.ReadAsArray()      
    im = Image.fromarray(data)
    im = im.convert('P')
    im.save(os.path.join(output,file.split('.')[0]+'.png'))

