# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 07:36:34 2020

@author: zjh
"""

import numpy as np
import pandas as pd
import time, os, sys, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
import gdal
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as functional
import cv2
# import tifffile as tiff
from torch.utils.data import DataLoader
from PIL import Image

random.seed(20180122)
np.random.seed(20180122)

   
class preddataset(Dataset):
    def __init__(self, file_path=None):
        super(preddataset, self).__init__()
        self.img = os.listdir(file_path)
        self.file_path = file_path
#         self.img = [x.replace('/gt/', '/imgs/') for x in self.gt]

    def __getitem__(self, index):
        tiffile = self.img[index]
        tiffile = os.path.join(self.file_path,tiffile)
        # ds = gdal.Open(tiffile,gdal.GA_ReadOnly)
        # fy4a_tile_data = ds.ReadAsArray(xoff, yoff, 256, 256)
        # tmp = np.zeros(fy4a_tile_data.shape, dtype=np.float32)
        # data_x = cv2.normalize(fy4a_tile_data,tmp,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        
        # img_path = '/data/data/cloud_tif/img/'
        # imgfile = os.path.join(img_path, tiffile)
        # img_tif = cv2.imread(imgfile, -1)
        
        img_tif = cv2.imread(tiffile, -1)
        img_tif = np.array(img_tif).astype(np.float32)
        img_tif[np.isnan(img_tif)] == 0
        imax = 0
        imin = 255
        
        for i in range(img_tif.shape[0]):
            for j in range(img_tif.shape[1]):
                for k in range(img_tif.shape[2]):
                    if img_tif[i,j,k] > imax:
                        imax = img_tif[i,j,k]
                    if img_tif[i,j,k] < imin:
                        imin = img_tif[i,j,k]
        print(imax, imin)
        if imax>255:
            print(tiffile)
    
        # # imax = img_tif.max()
        # # imin = img_tif.min()
        # img_tif = (img_tif - imin)/(imax - imin)
        # img_tif *= 255
        # # img_tif = img_tif.astype(np.uint8)
        
        # jpgfile = tiffile.replace('tif','jpg')

        # # img_jpg_path = '/tmp/'
        # # cv2.imwrite(img_jpg_path + newfilename, img_tif)
        # cv2.imwrite(jpgfile, img_tif)

        # # ds = Image.open(img_jpg_path + newfilename).convert('RGB')
        # ds = Image.open(jpgfile).convert('RGB')
        # img = ds.crop((xoff, yoff, xoff + 256, yoff + 256))
        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        # img = np.array(img).astype(np.float32)
        # img[np.isnan(img)] = 0
        # img /= 255.0
        # img -= mean
        # img /= std
        # img = img.transpose((2, 0, 1))
        input = torch.from_numpy(img_tif).float()
        
        # # data_x[np.isnan(data_x)] = 0
        # # input = torch.from_numpy(data_x)

        return input, index

    def __len__(self):
        return len(self.img)

    def filelist(self):
        return self.img         

class predprefetcher():
    def __init__(self, loader, use_cuda=True):
        self.loader = iter(loader)
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.stream = torch.cuda.Stream()    # 选择给定流的上下文管理器。在其上下文中排队的所有CUDA核心将在所选流上入队。
        self.preload()

    def preload(self):
        try:
#             self.next_inputs, self.next_targets, self.next_index = next(self.loader)
            self.next_inputs, self.next_index = next(self.loader)
        except StopIteration:
            self.next_inputs = None
            self.next_targets = None
            self.next_index = None
            return
        if self.use_cuda:
            with torch.cuda.stream(self.stream):
                self.next_inputs = self.next_inputs.cuda(non_blocking=True)
                # self.next_targets = self.next_targets.cuda(non_blocking=True)

    def next(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
#         targets = self.next_targets
        index = self.next_index
        self.preload()
#         return inputs, targets, index
        return inputs, index


if __name__ == "__main__":
    # dataset = preddataset('/home/zjh/AIDataset/img_jpg')
    # dataloader = DataLoader(dataset,batch_size=1, shuffle=False)
    # x = predprefetcher(dataloader,use_cuda=False)
    # for i in range(dataset.__len__()):
    #     input,_=x.next()
    import os
    data_dir = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/9_labels_id/'
    label_files = list(filter(lambda x: x.endswith('_label.tif'), os.listdir(data_dir)))
    for label_id in label_files:
        # top_potsdam_2_10_label.tif  top_potsdam_2_10_RGB.tif
        # id = label_file.split('.')[0].split('_')
        label_file=os.path.join(data_dir,label_id)
        lds=gdal.Open(label_file)
        arr = lds.ReadAsArray()
        origin_file = os.path.join('E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/',label_id.replace('label','RGB'))
        print(origin_file)
        ds = gdal.Open(origin_file)
        gt = ds.GetGeoTransform()
        proj = ds.GetProjectionRef()

        dst_format = 'GTiff'
        driver = gdal.GetDriverByName(dst_format)
        dst_nbands = 1
        file_dst=os.path.join(data_dir, label_id.replace('label','label_proj'))
        print(file_dst)
        dst_ds = driver.Create(file_dst, 6000, 6000, dst_nbands, 1,['COMPRESS=LZW'])
        # dst_ds = driver.Create(file_dst, xsize, ysize, dst_nbands, 6,['COMPRESS=LZW'])
        dst_ds.SetGeoTransform(gt)
        dst_ds.SetProjection(proj)

        if dst_nbands == 1:
            dst_ds.GetRasterBand(1).WriteArray(arr)
        # del dst_ds
        
   # data = 'E:/tmp/top_potsdam2_2_10_RGB_deeplabv3+_pca.tif'
   #  import gdal
   #  ds = gdal.Open(data)
   #  tile = ds.ReadAsArray(0,0,512,512)
   #  import cv2
   #  cv2.imwrite('E:/tmp/00.jpg',tile)
