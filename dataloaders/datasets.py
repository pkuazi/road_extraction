# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 16:24:18 2021

@author: zjh
"""
import sys
import os
# import cv2
import gdal
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
sys.path.append('..')
from configure import BASEDIR
from PIL import Image
from dataloaders import transforms
# from . import transforms

# transformed = transforms.train_transform_1(image=image,mask=mask)
# transformed_image=transformed['image']
# transformed_mask=transformed['mask']
def reclassify(target,origin_labels, output_labels):
    h, w = target.shape
    label = np.zeros((h, w), dtype=np.int8)
    # label[target == 6] = 1
    # label[target == 9] = 2
    # label[target == 7] = 3
    # label[target == 1] = 4
    # label[target == 3] = 5
    for i in range(len(origin_labels)):
        label[target==origin_labels[i]]=output_labels[i]

    # label = torch.as_tensor(label, dtype=torch.long)
    return label

class SegmentationDataset(Dataset):
    def __init__(self,tile_names, images_dir, masks_dir, tile_size, transform_name="train_transform_1"):
        super().__init__()
        self.tile_names=tile_names
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.tile_size=tile_size
        self.transform=transforms.__dict__[transform_name] if transform_name else None
    
    def __len__(self):
        return len(self.tile_names)
    
    def __getitem__(self, id):       
        img_name=self.tile_names[id][0]
        xoff=self.tile_names[id][1]
        yoff=self.tile_names[id][2]
        
        mask_name=img_name.split('.')[0]+"_mask."+img_name.split('.')[1]
        image_path=os.path.join(self.images_dir,img_name )
        # print(image_path)
        mask_path = os.path.join(self.masks_dir,mask_name)
        
        # read data sample
        sample=dict(
            id=id, 
            image=self.read_image(image_path,xoff,yoff),
            mask=self.read_mask(mask_path,xoff,yoff),
            )
        
        #background 0; railway 1; road 2; country road 3; bridge 4
        # out_mask = reclassify(sample['mask'],[0,1,2,3,4],[0,0,0,0,1] )
        # sample['mask']=out_mask
        # test transform
        # print('before transform..',sample)
        # img = Image.fromarray(np.uint8(sample['image'].transpose(1,2,0)))
        # img.save(os.path.join(masks_dir,'%s_before.bmp'%id)) 
        # gt_ = Image.fromarray(sample['mask'])
        # gt_ =gt_.convert("L")
        # gt_.save(os.path.join(masks_dir,'%s_gt_before.bmp'%id)) 
        
        # print(np.unique(sample['mask']))
        # apply augmentation
        # augmentation transform input image must be in HWC format
        if self.transform is not None:
            sample=self.transform(**sample) #(**)将参数以字典的形式导入
        
        # test transform
        # img = Image.fromarray(np.uint8(sample['image'].transpose(1,2,0)))
        # img.save(os.path.join(masks_dir,'%s_after.bmp'%id)) 
        
        # gt_ = Image.fromarray(sample['mask'])
        # gt_ =gt_.convert("L")
        # gt_.save(os.path.join(masks_dir,'%s_gt_after.bmp'%id)) 
        
        # _img = torch.FloatTensor(sample['image'])
        # _target = torch.LongTensor(sample['mask'])
        # sample = {'image': _img, 'mask': _target}
        # print('after transform..',sample)
        # print(np.unique(sample['mask']))
        
        return sample
        # return _img, _target
        
    
    def read_image(self, path,xoff, yoff):
        # # cv2.imread(filename, flags) ,flags：标志位，默认{cv2.IMREAD_COLOR，cv2.IMREAD_GRAYSCALE 0，cv2.IMREAD_UNCHANGED -1}
        # image = cv2.imread(path)
        # # OpenCV reads an image in BGR format (so color channels of the image have the following order: Blue, Green, Red). Albumentations uses the most common and popular RGB image format. So when using OpenCV, we need to convert the image format to RGB explicitly.
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgds=gdal.Open(path)
        # CHW(channel, height, width) format, can be directly changed into tensor
        image = imgds.ReadAsArray(xoff, yoff, self.tile_size, self.tile_size).astype(np.float32) 
        return image
    
    def read_mask(self, path,xoff, yoff):
        # mask = cv2.imread(path,-1)
        gtds=gdal.Open(path)
        mask = gtds.ReadAsArray(xoff, yoff, self.tile_size, self.tile_size)
        return mask
    
def main():
    images_dir = os.path.join(BASEDIR,'data/raw/')
    masks_dir = os.path.join(BASEDIR,'data/processed/')
    train_set = SegmentationDataset(tile_names= [["4.bmp", 11088, 7256], ["4.bmp", 5082, 7256]],images_dir=images_dir,masks_dir=masks_dir,tile_size = 512, transform_name = 'train_transform_1')
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True) 
    # num=0
    for batch_idx, data in enumerate(train_loader):
        img,label=data['image'],data['mask']
        print(img.shape)
        
if __name__=="__main__":
    main()
    

        
            
        
        