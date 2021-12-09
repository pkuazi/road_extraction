# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:31:20 2021

@author: zjh
"""

import albumentations as A

# gdal read in as CHW, need to be transformed into HWC
def pre_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(1, 2, 0).astype("float32")
    else:
        return image.astype("float32")

#cv2/transform format HWC need to be transformed into CHW to tensor
def post_transform(image, **kwargs):
    if image.ndim == 3:
        return image.transpose(2, 0, 1).astype("float32")
    else:
        return image.astype("float32")
  
# --------------------------------------------------------------------
# Segmentation transforms
# --------------------------------------------------------------------
pre_transform = A.Lambda(name="pre_transform", image=pre_transform, mask=pre_transform)
post_transform = A.Lambda(name="post_transform", image=post_transform, mask=post_transform)

# crop 512
train_transform_1=A.Compose([
    pre_transform,
    # A.RandomCrop(512,512,p=1.),
    A.Flip(p=0.75),
    A.RandomBrightnessContrast(p=0.2),

    post_transform,
    ])

# crop 768 and hard augs
train_transform_2=A.Compose([
    pre_transform,
    A.RandomScale(scale_limit=0.3,p=0.5),
    A.PadIfNeeded(768,768,p=1),
    A.RandomCrop(768,768,p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75,p=0.05),
    
    # color transform
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1.),
            A.RandomGamma(p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1),
            ],
        p=0.5,),
    
    # noise transform
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.IAASharpen(p=1),
            A.GaussianBlur(p=1),
            ],
        p=0.2,
        ),
    post_transform,
    ])

# crop 1024 and hard augs
valid_transform_3=A.Compose([
    pre_transform,
    A.RandomScale(scale_limit=0.3, p=0.5),
    A.PadIfNeeded(1024, 1024, p=1),
    A.RandomCrop(1024, 1024, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(p=1),
            A.RGBShift(p=1),
        ],
        p=0.5,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.IAASharpen(p=1),
            # A.ImageCompression(quality_lower=0.7, p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
        ),
    post_transform,
    ])

# crop 768 and very hard augs
train_transform_4 = A.Compose([
    pre_transform,
    A.ShiftScaleRotate(scale_limit=0.2, rotate_limit=45, border_mode=0, value=0, p=0.7),
    A.PadIfNeeded(768, 768, border_mode=0, value=0, p=1.),
    A.RandomCrop(768, 768, p=1.),
    A.Flip(p=0.75),
    A.Downscale(scale_min=0.5, scale_max=0.75, p=0.05),
    A.MaskDropout(max_objects=3, image_fill_value=0, mask_fill_value=0, p=0.1),

    # color transforms
    A.OneOf(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.RandomGamma(gamma_limit=(70, 130), p=1),
            A.ChannelShuffle(p=0.2),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1),
            A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=1),
        ],
        p=0.8,
    ),

    # distortion
    A.OneOf(
        [
            A.ElasticTransform(p=1),
            A.OpticalDistortion(p=1),
            A.GridDistortion(p=1),
            A.IAAPerspective(p=1),
        ],
        p=0.2,
    ),

    # noise transforms
    A.OneOf(
        [
            A.GaussNoise(p=1),
            A.MultiplicativeNoise(p=1),
            A.IAASharpen(p=1),
            A.GaussianBlur(p=1),
        ],
        p=0.2,
    ),
    post_transform,
])
# another_transformed_image=transform(image=another_image)['image']
