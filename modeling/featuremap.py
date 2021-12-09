# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:28:25 2021

@author: zjh
"""
import torch
import numpy as np
import cv2
from PIL import Image
import UNet_SNws
import gdal
import itertools
import os

def featuremap(input):
    

    input = input.cuda()
    # print(input.shape)
    
    model = UNet_SNws( n_channels=3, n_filters=64, n_class=6, using_movavg=1, using_bn=1)
    model =model.cuda()
    model.eval()
    output = model(input)
    _, predict = torch.max(output.data, 1)
    pred = predict[0].cpu()
    pred = pred.numpy()
    print(pred)
    # stinterp = stinterp.cpu()
    # ftinterp = torch.squeeze(stinterp, 0).detach().numpy()
    # print('before argmax:',stinterp)
    # outputs = np.argmax(ftinterp, axis=1)
    # print('after argmax:',outputs)
    # sys.exit()
    # ftinterp = torch.squeeze(stinterp, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ftinterp.jpg', ftinterp)
    # print(stinterp.detach().numpy().shape)!
    # return ftinterp
    return pred

# itertools模块提供的全部是处理迭代功能的函数,它们的返回值不是list,而是迭代对象,只有用for循环迭代的时候才真正计算
def gen_tiles_offs(xsize, ysize, BLOCK_SIZE=512,OVERLAP_SIZE=0):
    xoff_list = []
    yoff_list = []
    
    cnum_tile = int((xsize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    rnum_tile = int((ysize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    
    for j in range(cnum_tile + 1):
        xoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * j                  
        if j == cnum_tile:
            xoff = xsize - BLOCK_SIZE
        xoff_list.append(xoff)
        
    for i in range(rnum_tile + 1):
        yoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * i
        if i == rnum_tile:
            yoff = ysize - BLOCK_SIZE
        yoff_list.append(yoff)
    
    if xoff_list[-1] == xoff_list[-2]:
        xoff_list.pop()    # pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值
    if yoff_list[-1] == yoff_list[-2]:    # the last tile overlap with the last second tile
        yoff_list.pop()
    # print(xoff_list, yoff_list)
    return [d for d in itertools.product(xoff_list,yoff_list)]

def gen_file_list(geotif):
    file_list = []
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    off_list = gen_tiles_offs(xsize, ysize, 512, 0)
   
    for xoff,yoff in off_list:
        file_list.append((geotif, xoff, yoff))
    return file_list

def main():    
    input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    # input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    
    rgb_dir = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB'
    output_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/11_deeplab_feature'
    rsts = list(filter(lambda x: x.endswith('.tif') and x.startswith('top_potsdam_2_10'), os.listdir(rgb_dir))) 
    for rastername in rsts:
        print(rastername)
        rasterfile = os.path.join(rgb_dir,rastername)
        # rasterfile='/home/zjh/xa098_b4321.tif'
        file_dst=os.path.join(output_path, rastername.split('.tif')[0]+'_deeplabv3+.tif')
        
        dataset = gdal.Open(rasterfile)
        ysize=dataset.RasterYSize
        xsize=dataset.RasterXSize
        print(xsize,ysize)
        img = dataset.ReadAsArray()
        img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        
        gt=dataset.GetGeoTransform()
        out_proj=dataset.GetProjectionRef()
        
        files_list = gen_file_list(rasterfile)

        # pred_arr = np.zeros([6,ysize,xsize])
        # pred_arr = np.zeros([512,512])
        num = len(files_list)
        for i in range(num):
            print("processing:",i)
            _,xoff,yoff = files_list[i]
            print(xoff, yoff)
            
            tile = np.array(img[:,yoff:yoff+512, xoff:xoff+512]).astype(np.float32)
            
            input = torch.from_numpy(tile).float()
            input = torch.unsqueeze(input, 0)
            
            outputs= featuremap(input)
            
            out_gt=(gt[0]+xoff*gt[1],gt[1],gt[2],gt[3]+gt[5]*yoff, gt[4],gt[5])
    #         tile_pred = predict.cpu().numpy()
            # pred_arr[yoff:yoff+512,xoff:xoff+512]=outputs
            # pred_arr[:,yoff:yoff+512,xoff:xoff+512]=outputs
            
            dst_format = 'GTiff'
            driver = gdal.GetDriverByName(dst_format)
            dst_nbands = 1
            # dst_nbands = 6
            file_dst=os.path.join(output_path, rastername.split('.tif')[0]+'_%s_unetv2.tif'%i)
            print(file_dst)
            dst_ds = driver.Create(file_dst, 512, 512, dst_nbands, 6,['COMPRESS=LZW'])
            # dst_ds = driver.Create(file_dst, xsize, ysize, dst_nbands, 6,['COMPRESS=LZW'])
            dst_ds.SetGeoTransform(out_gt)
            dst_ds.SetProjection(out_proj)

            if dst_nbands == 1:
                dst_ds.GetRasterBand(1).WriteArray(outputs)
            else:
                for i in range(dst_nbands):
                    arr=outputs[i, :, :]
                    print(arr.shape)
                    dst_ds.GetRasterBand(i + 1).WriteArray(arr)
            del dst_ds
            
        # dst_nbands = 6
        # dst_format = 'GTiff'
        # driver = gdal.GetDriverByName(dst_format)
        # dst_ds = driver.Create(file_dst, 512, 512, dst_nbands, 6,['COMPRESS=LZW'])
        # # dst_ds = driver.Create(file_dst, xsize, ysize, dst_nbands, 6,['COMPRESS=LZW'])
        # dst_ds.SetGeoTransform(out_gt)
        # dst_ds.SetProjection(out_proj)
        # if dst_nbands == 1:
        #     dst_ds.GetRasterBand(1).WriteArray(pred_arr)
        # else:
        #     for i in range(dst_nbands):
        #         arr=pred_arr[i, :, :]
        #         print(arr.shape)
        #         dst_ds.GetRasterBand(i + 1).WriteArray(arr)
        # del dst_ds
        
if __name__ == "__main__":
    main()
    
# if __name__ == "__main__":
#     import torch
#     import numpy as np
#     import cv2
#     from PIL import Image
#     import os
#     # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#     input_path = 'D:/data/xa098_b4321.tif'
#     import gdal
#     ds = gdal.Open(input_path)
#     img = ds.ReadAsArray(50,60,256,256)
#     img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    
#     # img = Image.open(input_path).convert('RGB')
#     # xoff = 0
#     # yoff = 0
#     # # xoff = np.random.randint(0,725)
#     # # yoff = np.random.randint(0,944)
#     # img = img.crop((xoff, yoff, xoff + 981, yoff + 981))
#     # mean = (0.485, 0.456, 0.406)
#     # std = (0.229, 0.224, 0.225)

#     img = np.array(img).astype(np.float32)
#     # img /= 255
#     print(img.shape)
#     imgx = img.transpose((1, 2, 0))
    
#     output_path = 'D:/tmp/ft_from_unetlayers/'
#     cv2.imwrite(output_path + 'img.jpg', imgx)
    
#     input = torch.from_numpy(img).float()
#     input = torch.unsqueeze(input, 0)
#     # print(input.shape)
#     model = UNet_SNws(n_channels=3, n_filters=64, n_class=6, using_movavg=True, using_bn=True)
#     # input = torch.rand(1, 3, 512, 512)
#     # print(model.block1)
    
#     st1 = model.conv1_1(input)
#     ft1 = torch.squeeze(st1, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft1.jpg', ft1)
#     print(st1.detach().numpy().shape)
    
#     st2 = model.conv1_2(st1)
#     ft2 = torch.squeeze(st2, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft2.jpg', ft2)
#     print(st2.detach().numpy().shape)
    
#     st3 = model.maxPool(st2)
#     ft3 = torch.squeeze(st3, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft3.jpg', ft3)
#     print(st3.detach().numpy().shape)
    
#     st4 = model.conv2_1(st3)
#     ft4 = torch.squeeze(st4, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft4.jpg', ft4)
#     print(st4.detach().numpy().shape)
    
#     st5 = model.conv2_2(st4)
#     ft5 = torch.squeeze(st5, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft5.jpg', ft5)
#     print(st5.detach().numpy().shape)
    
#     st6 = model.maxPool(st5)
#     ft6 = torch.squeeze(st6, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft6.jpg', ft6)
#     print(st6.detach().numpy().shape)
    
#     st7 = model.conv3_1(st6)
#     ft7 = torch.squeeze(st7, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft7.jpg', ft7)
#     print(st7.detach().numpy().shape)
    
#     st8 = model.conv3_2(st7)
#     ft8 = torch.squeeze(st8, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft8.jpg', ft8)
#     print(st8.detach().numpy().shape)
#     low_level_feat = st8
    
#     st9 = model.maxPool(st8)
#     ft9 = torch.squeeze(st9, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft9.jpg', ft9)
#     print(st9.detach().numpy().shape)
    
#     st10 = model.conv4_1(st9)
#     ft10 = torch.squeeze(st10, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft10.jpg', ft10)
#     print(st10.detach().numpy().shape)
    
#     st11 = model.conv4_2(st10)
#     ft11 = torch.squeeze(st11, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft11.jpg', ft11)
#     print(st11.detach().numpy().shape)
    
#     st12 = model.upSample(st11)
#     ft12 = torch.squeeze(st12, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft12.jpg', ft12)
#     print(st12.detach().numpy().shape)
    
#     st13 = model.conv5_1(st12)
#     ft13 = torch.squeeze(st13, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft13.jpg', ft13)
#     print(st13.detach().numpy().shape)
    
#     st13 = CropConcat(st8, st13)
    
#     st14 = model.conv5_2(st13)
#     ft14 = torch.squeeze(st14, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft14.jpg', ft14)
#     print(st14.detach().numpy().shape)
    
#     st15 = model.conv5_3(st14)
#     ft15 = torch.squeeze(st15, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft15.jpg', ft15)
#     print(st15.detach().numpy().shape)
    
#     st16 = model.upSample(st15)
#     ft16 = torch.squeeze(st16, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft16.jpg', ft16)
#     print(st16.detach().numpy().shape)
    
#     st17 = model.conv6_1(st16)
#     ft17 = torch.squeeze(st17, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft17.jpg', ft17)
#     print(st17.detach().numpy().shape)
    
#     st17=CropConcat(st5,st17)
    
#     st18 = model.conv6_2(st17)
#     ft18 = torch.squeeze(st18, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft18.jpg', ft18)
#     print(st18.detach().numpy().shape)
    
#     st19 = model.conv6_3(st18)
#     ft19 = torch.squeeze(st19, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft19.jpg', ft19)
#     print(st19.detach().numpy().shape)
    
#     st20 = model.upSample(st19)
#     ft20 = torch.squeeze(st20, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft20.jpg', ft20)
#     print(st20.detach().numpy().shape)
    
#     st21 = model.conv7_1(st20)
#     ft21 = torch.squeeze(st21, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft21.jpg', ft21)
#     print(st21.detach().numpy().shape)
    
#     st21=CropConcat(st2,st21)
    
#     st22 = model.conv7_2(st21)
#     ft22 = torch.squeeze(st22, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft22.jpg', ft22)
#     print(st22.detach().numpy().shape)
    
#     st23 = model.conv7_3(st22)
#     ft23 = torch.squeeze(st23, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ft23.jpg', ft23)
#     print(st23.detach().numpy().shape)