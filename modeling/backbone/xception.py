import sys
sys.path.append('/home/zjh/lulc_dl')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False, BatchNorm=None):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, BatchNorm=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = BatchNorm(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(inplanes, planes, 3, 1, dilation, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 2, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if stride == 1 and is_last:
            rep.append(self.relu)
            rep.append(SeparableConv2d(planes, planes, 3, 1, BatchNorm=BatchNorm))
            rep.append(BatchNorm(planes))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip

        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, output_stride, BatchNorm,
                 pretrained=False):
        super(AlignedXception, self).__init__()

        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError


        # Entry flow
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(64)

        self.block1 = Block(64, 128, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False)
        
        self.block2 = Block(128, 256, reps=2, stride=2, BatchNorm=BatchNorm, start_with_relu=False,
                            grow_first=True)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, BatchNorm=BatchNorm,
                            start_with_relu=True, grow_first=True, is_last=True)

        # Middle flow
        self.block4  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block5  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block6  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block7  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block8  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block9  = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block13 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block14 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block15 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block16 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block17 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block18 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)
        self.block19 = Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation,
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                             BatchNorm=BatchNorm, start_with_relu=True, grow_first=False, is_last=True)

        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn3 = BatchNorm(1536)

        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn4 = BatchNorm(1536)

        self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1], BatchNorm=BatchNorm)
        self.bn5 = BatchNorm(2048)

        # Init weights
        self._init_weight()

        # Load pretrained model
        if pretrained:
            self._load_pretrained_model()

    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth')
        model_dict = {}
        state_dict = self.state_dict()

        for k, v in pretrain_dict.items():
            if k in state_dict:
                if 'pointwise' in k:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                if k.startswith('block11'):
                    model_dict[k] = v
                    model_dict[k.replace('block11', 'block12')] = v
                    model_dict[k.replace('block11', 'block13')] = v
                    model_dict[k.replace('block11', 'block14')] = v
                    model_dict[k.replace('block11', 'block15')] = v
                    model_dict[k.replace('block11', 'block16')] = v
                    model_dict[k.replace('block11', 'block17')] = v
                    model_dict[k.replace('block11', 'block18')] = v
                    model_dict[k.replace('block11', 'block19')] = v
                elif k.startswith('block12'):
                    model_dict[k.replace('block12', 'block20')] = v
                elif k.startswith('bn3'):
                    model_dict[k] = v
                    model_dict[k.replace('bn3', 'bn4')] = v
                elif k.startswith('conv4'):
                    model_dict[k.replace('conv4', 'conv5')] = v
                elif k.startswith('bn4'):
                    model_dict[k.replace('bn4', 'bn5')] = v
                else:
                    model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def featuremap(rasterfile,file_dst):
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    import os
    import gdal
    
    ds = gdal.Open(rasterfile)
    img = ds.ReadAsArray()
    ysize=ds.RasterYSize
    xsize=ds.RasterXSize
    out_gt=ds.GetGeoTransform()
    out_proj=ds.GetProjectionRef()
    # img = ds.ReadAsArray(50,60,256,256)
    img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    img = np.array(img).astype(np.float32)
    # img /= 255
    print(img.shape)
    imgx = img.transpose((1, 2, 0))
    
    # output_path = 'E:/tmp/ft_from_layer/'
    # cv2.imwrite(output_path + 'img.jpg', imgx)
    
    input = torch.from_numpy(img).float()
    input = torch.unsqueeze(input, 0)
    # print(input.shape)
    model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    # input = torch.rand(1, 3, 512, 512)
    # print(model.block1)
    
    st1 = model.conv1(input)
    # ft1 = torch.squeeze(st1, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft1.jpg', ft1)
    # print(st1.detach().numpy().shape)
    
    st2 = model.bn1(st1)
    # ft2 = torch.squeeze(st2, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft2.jpg', ft2)
    # print(st2.detach().numpy().shape)
    
    st3 = model.relu(st2)
    # ft3 = torch.squeeze(st3, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft3.jpg', ft3)
    # print(st3.detach().numpy().shape)
    
    st4 = model.conv2(st3)
    # ft4 = torch.squeeze(st4, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft4.jpg', ft4)
    # print(st4.detach().numpy().shape)
    
    st5 = model.bn2(st4)
    # ft5 = torch.squeeze(st5, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft5.jpg', ft5)
    # print(st5.detach().numpy().shape)
    
    st5_upsample = F.interpolate(st5, scale_factor=2, mode='bilinear', align_corners=True)
    ft5_upsample = torch.squeeze(st5_upsample, 0).detach().numpy()
    # cv2.imwrite(output_path + 'ft5_up1.jpg', ft5_upsample)
    print(ft5_upsample.shape)
    dst_nbands = ft5_upsample.shape[0]

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(file_dst, ysize, xsize, dst_nbands, 1,['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(out_gt)
    dst_ds.SetProjection(out_proj)
    
    if dst_nbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(ft5_upsample)
    else:
        for i in range(dst_nbands):
            print(i)
            dst_ds.GetRasterBand(i + 1).WriteArray(ft5_upsample[i, :, :])
    del dst_ds
    # st6 = model.relu(st5)
    # ft6 = torch.squeeze(st6, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft6.jpg', ft6)
    # print(st6.detach().numpy().shape)
    
    # st7 = model.block1(st6)
    # ft7 = torch.squeeze(st7, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft7.jpg', ft7)
    # print(st7.detach().numpy().shape)
    
    # st8 = model.relu(st7)
    # ft8 = torch.squeeze(st8, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft8.jpg', ft8)
    # print(st8.detach().numpy().shape)
    # low_level_feat = st8
    
    # st9 = model.block2(st8)
    # ft9 = torch.squeeze(st9, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft9.jpg', ft9)
    # print(st9.detach().numpy().shape)
    
    # st10 = model.block3(st9)
    # ft10 = torch.squeeze(st10, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft10.jpg', ft10)
    # print(st10.detach().numpy().shape)
    
    # st11 = model.block4(st10)
    # ft11 = torch.squeeze(st11, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft11.jpg', ft11)
    # print(st11.detach().numpy().shape)
    
    # st12 = model.block5(st11)
    # ft12 = torch.squeeze(st12, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft12.jpg', ft12)
    # print(st12.detach().numpy().shape)
    
    # st13 = model.block6(st12)
    # ft13 = torch.squeeze(st13, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft13.jpg', ft13)
    # print(st13.detach().numpy().shape)
    
    # st14 = model.block7(st13)
    # ft14 = torch.squeeze(st14, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft14.jpg', ft14)
    # print(st14.detach().numpy().shape)
    
    # st15 = model.block8(st14)
    # ft15 = torch.squeeze(st15, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft15.jpg', ft15)
    # print(st15.detach().numpy().shape)
    
    # st16 = model.block9(st15)
    # ft16 = torch.squeeze(st16, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft16.jpg', ft16)
    # print(st16.detach().numpy().shape)
    
    # st17 = model.block10(st16)
    # ft17 = torch.squeeze(st17, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft17.jpg', ft17)
    # print(st17.detach().numpy().shape)
    
    # st18 = model.block11(st17)
    # ft18 = torch.squeeze(st18, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft18.jpg', ft18)
    # print(st18.detach().numpy().shape)
    
    # st19 = model.block12(st18)
    # ft19 = torch.squeeze(st19, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft19.jpg', ft19)
    # print(st19.detach().numpy().shape)
    
    # st20 = model.block13(st19)
    # ft20 = torch.squeeze(st20, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft20.jpg', ft20)
    # print(st20.detach().numpy().shape)
    
    # st21 = model.block14(st20)
    # ft21 = torch.squeeze(st21, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft21.jpg', ft21)
    # print(st21.detach().numpy().shape)
    
    # st22 = model.block15(st21)
    # ft22 = torch.squeeze(st22, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft22.jpg', ft22)
    # print(st22.detach().numpy().shape)
    
    # st23 = model.block16(st22)
    # ft23 = torch.squeeze(st23, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft23.jpg', ft23)
    # print(st23.detach().numpy().shape)
    
    # st24 = model.block17(st23)
    # ft24 = torch.squeeze(st24, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft24.jpg', ft24)
    # print(st24.detach().numpy().shape)
    
    # st25 = model.block18(st24)
    # ft25 = torch.squeeze(st25, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft25.jpg', ft25)
    # print(st25.detach().numpy().shape)
    
    # st26 = model.block19(st25)
    # ft26 = torch.squeeze(st26, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft26.jpg', ft26)
    # print(st26.detach().numpy().shape)
    
    # st27 = model.block20(st26)
    # ft27 = torch.squeeze(st27, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft27.jpg', ft27)
    # print(st27.detach().numpy().shape)
    
    # st28 = model.relu(st27)
    # ft28 = torch.squeeze(st28, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft28.jpg', ft28)
    # print(st28.detach().numpy().shape)
    
    # st29 = model.conv3(st28)
    # ft29 = torch.squeeze(st29, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft29.jpg', ft29)
    # print(st29.detach().numpy().shape)
    
    # st30 = model.bn3(st29)
    # ft30 = torch.squeeze(st30, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft30.jpg', ft30)
    # print(st30.detach().numpy().shape)
    
    # st31 = model.relu(st30)
    # ft31 = torch.squeeze(st31, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft31.jpg', ft31)
    # print(st31.detach().numpy().shape)
    
    # st32 = model.conv4(st31)
    # ft32 = torch.squeeze(st32, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft32.jpg', ft32)
    # print(st32.detach().numpy().shape)
    
    # st33 = model.bn4(st32)
    # ft33 = torch.squeeze(st33, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft33.jpg', ft33)
    # print(st33.detach().numpy().shape)
    
    # st34 = model.relu(st33)
    # ft34 = torch.squeeze(st34, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft34.jpg', ft34)
    # print(st34.detach().numpy().shape)
    
    # st35 = model.conv5(st34)
    # ft35 = torch.squeeze(st35, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft35.jpg', ft35)
    # print(st35.detach().numpy().shape)
    
    # st36 = model.bn5(st35)
    # ft36 = torch.squeeze(st36, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft36.jpg', ft36)
    # print(st36.detach().numpy().shape)
    
    # st37 = model.relu(st36)
    # ft37 = torch.squeeze(st37, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft37.jpg', ft37)
    # print(st37.detach().numpy().shape)
    
    # output, low_level_feat = model(input)
    # print(output.size())
    # print(output)
    # print(low_level_feat.size())
def main():
    import os
    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    
    # rgb_dir = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB'
    rgb_dir = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB'
    output_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/10_xception_feature'
    rsts = list(filter(lambda x: x.endswith('.tif'), os.listdir(rgb_dir))) 
    for rastername in rsts:
        print(rastername)
        rasterfile = os.path.join(rgb_dir,rastername)
        file_dst=os.path.join(output_path, rastername.split('.tif')[0]+'_ft5.tif')
        featuremap(rasterfile, file_dst)

if __name__ == "__main__":
    main()
    
    # import torch
    # import numpy as np
    # import cv2
    # from PIL import Image
    # import os
    # # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    # # input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    # import gdal
    # ds = gdal.Open(input_path)
    # img = ds.ReadAsArray()
    # # img = ds.ReadAsArray(50,60,256,256)
    # img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    
    # # img = Image.open(input_path).convert('RGB')
    # # xoff = 0
    # # yoff = 0
    # # # xoff = np.random.randint(0,725)
    # # # yoff = np.random.randint(0,944)
    # # img = img.crop((xoff, yoff, xoff + 981, yoff + 981))
    # # mean = (0.485, 0.456, 0.406)
    # # std = (0.229, 0.224, 0.225)

    # img = np.array(img).astype(np.float32)
    # # img /= 255
    # print(img.shape)
    # imgx = img.transpose((1, 2, 0))
    
    # output_path = '/home/zjh/tmp/ft_from_layers/'
    # # output_path = 'E:/tmp/ft_from_layer/'
    # cv2.imwrite(output_path + 'img.jpg', imgx)
    
    # input = torch.from_numpy(img).float()
    # input = torch.unsqueeze(input, 0)
    # # print(input.shape)
    # model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=True, output_stride=8)
    # # input = torch.rand(1, 3, 512, 512)
    # # print(model.block1)
    
    # st1 = model.conv1(input)
    # ft1 = torch.squeeze(st1, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft1.jpg', ft1)
    # print(st1.detach().numpy().shape)
    
    # st2 = model.bn1(st1)
    # ft2 = torch.squeeze(st2, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft2.jpg', ft2)
    # print(st2.detach().numpy().shape)
    
    # st3 = model.relu(st2)
    # ft3 = torch.squeeze(st3, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft3.jpg', ft3)
    # print(st3.detach().numpy().shape)
    
    # st4 = model.conv2(st3)
    # ft4 = torch.squeeze(st4, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft4.jpg', ft4)
    # print(st4.detach().numpy().shape)
    
    # st5 = model.bn2(st4)
    # ft5 = torch.squeeze(st5, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft5.jpg', ft5)
    # print(st5.detach().numpy().shape)
    
    # st5_upsample = F.interpolate(st5, scale_factor=2, mode='bilinear', align_corners=True)
    # ft5_upsample = torch.squeeze(st5_upsample, 0).detach().numpy()[3:6].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft5_up1.jpg', ft5_upsample)
    
    
    # st6 = model.relu(st5)
    # ft6 = torch.squeeze(st6, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft6.jpg', ft6)
    # print(st6.detach().numpy().shape)
    
    # st7 = model.block1(st6)
    # ft7 = torch.squeeze(st7, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft7.jpg', ft7)
    # print(st7.detach().numpy().shape)
    
    # st8 = model.relu(st7)
    # ft8 = torch.squeeze(st8, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft8.jpg', ft8)
    # print(st8.detach().numpy().shape)
    # low_level_feat = st8
    
    # st9 = model.block2(st8)
    # ft9 = torch.squeeze(st9, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft9.jpg', ft9)
    # print(st9.detach().numpy().shape)
    
    # st10 = model.block3(st9)
    # ft10 = torch.squeeze(st10, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft10.jpg', ft10)
    # print(st10.detach().numpy().shape)
    
    # st11 = model.block4(st10)
    # ft11 = torch.squeeze(st11, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft11.jpg', ft11)
    # print(st11.detach().numpy().shape)
    
    # st12 = model.block5(st11)
    # ft12 = torch.squeeze(st12, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft12.jpg', ft12)
    # print(st12.detach().numpy().shape)
    
    # st13 = model.block6(st12)
    # ft13 = torch.squeeze(st13, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft13.jpg', ft13)
    # print(st13.detach().numpy().shape)
    
    # st14 = model.block7(st13)
    # ft14 = torch.squeeze(st14, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft14.jpg', ft14)
    # print(st14.detach().numpy().shape)
    
    # st15 = model.block8(st14)
    # ft15 = torch.squeeze(st15, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft15.jpg', ft15)
    # print(st15.detach().numpy().shape)
    
    # st16 = model.block9(st15)
    # ft16 = torch.squeeze(st16, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft16.jpg', ft16)
    # print(st16.detach().numpy().shape)
    
    # st17 = model.block10(st16)
    # ft17 = torch.squeeze(st17, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft17.jpg', ft17)
    # print(st17.detach().numpy().shape)
    
    # st18 = model.block11(st17)
    # ft18 = torch.squeeze(st18, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft18.jpg', ft18)
    # print(st18.detach().numpy().shape)
    
    # st19 = model.block12(st18)
    # ft19 = torch.squeeze(st19, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft19.jpg', ft19)
    # print(st19.detach().numpy().shape)
    
    # st20 = model.block13(st19)
    # ft20 = torch.squeeze(st20, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft20.jpg', ft20)
    # print(st20.detach().numpy().shape)
    
    # st21 = model.block14(st20)
    # ft21 = torch.squeeze(st21, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft21.jpg', ft21)
    # print(st21.detach().numpy().shape)
    
    # st22 = model.block15(st21)
    # ft22 = torch.squeeze(st22, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft22.jpg', ft22)
    # print(st22.detach().numpy().shape)
    
    # st23 = model.block16(st22)
    # ft23 = torch.squeeze(st23, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft23.jpg', ft23)
    # print(st23.detach().numpy().shape)
    
    # st24 = model.block17(st23)
    # ft24 = torch.squeeze(st24, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft24.jpg', ft24)
    # print(st24.detach().numpy().shape)
    
    # st25 = model.block18(st24)
    # ft25 = torch.squeeze(st25, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft25.jpg', ft25)
    # print(st25.detach().numpy().shape)
    
    # st26 = model.block19(st25)
    # ft26 = torch.squeeze(st26, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft26.jpg', ft26)
    # print(st26.detach().numpy().shape)
    
    # st27 = model.block20(st26)
    # ft27 = torch.squeeze(st27, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft27.jpg', ft27)
    # print(st27.detach().numpy().shape)
    
    # st28 = model.relu(st27)
    # ft28 = torch.squeeze(st28, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft28.jpg', ft28)
    # print(st28.detach().numpy().shape)
    
    # st29 = model.conv3(st28)
    # ft29 = torch.squeeze(st29, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft29.jpg', ft29)
    # print(st29.detach().numpy().shape)
    
    # st30 = model.bn3(st29)
    # ft30 = torch.squeeze(st30, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft30.jpg', ft30)
    # print(st30.detach().numpy().shape)
    
    # st31 = model.relu(st30)
    # ft31 = torch.squeeze(st31, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft31.jpg', ft31)
    # print(st31.detach().numpy().shape)
    
    # st32 = model.conv4(st31)
    # ft32 = torch.squeeze(st32, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft32.jpg', ft32)
    # print(st32.detach().numpy().shape)
    
    # st33 = model.bn4(st32)
    # ft33 = torch.squeeze(st33, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft33.jpg', ft33)
    # print(st33.detach().numpy().shape)
    
    # st34 = model.relu(st33)
    # ft34 = torch.squeeze(st34, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft34.jpg', ft34)
    # print(st34.detach().numpy().shape)
    
    # st35 = model.conv5(st34)
    # ft35 = torch.squeeze(st35, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft35.jpg', ft35)
    # print(st35.detach().numpy().shape)
    
    # st36 = model.bn5(st35)
    # ft36 = torch.squeeze(st36, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft36.jpg', ft36)
    # print(st36.detach().numpy().shape)
    
    # st37 = model.relu(st36)
    # ft37 = torch.squeeze(st37, 0).detach().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ft37.jpg', ft37)
    # print(st37.detach().numpy().shape)
    
    # output, low_level_feat = model(input)
    # # print(output.size())
    # # print(output)
    # # print(low_level_feat.size())
    
