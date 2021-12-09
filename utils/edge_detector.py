# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 14:00:19 2020

@author: ink
"""

import numpy as np
import cv2 
from matplotlib import pyplot as plt
import os

input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
img = cv2.imread(input_path,0)
edges = cv2.Canny(img,200,400)
cv2.imwrite('E:/tmp/caany_%s_%s.jpg'%(200,400), edges)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()


# input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'

