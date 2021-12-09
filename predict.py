import os
import numpy as np
import torch
from modeling.deeplab import *
import gc 
from modeling.UNet_SNws import *
import time
from datetime import datetime 
from configure import BASEDIR,BLOCK_SIZE,NUM_CLASSES
# Test the trained model
def unet_predict(array):
#     gc.enable()
#     gc.set_debug(gc.DEBUG_UNCOLLECTABLE ) # gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | 
    
    # Define network
    net = UNet_SNws(3, 64,NUM_CLASSES, using_movavg=1, using_bn=1).cuda()
    
    model = os.path.join(os.getcwd(),'run/road/unet/model_best.pth.tar')
    print('[%s] Start test using: %s.' % (datetime.now(), model.split('/')[-1]))
    # net = net.cuda()
    if torch.cuda.is_available():
        checkpoint = torch.load(model)
        net.load_state_dict(checkpoint['state_dict'])
        checkpoint= None 
    else:
        checkpoint = torch.load(model,map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'])
        checkpoint= None
    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    # start test
    net.eval()
    
    arr = np.expand_dims(array,axis=0)
    inputs = torch.tensor(arr, dtype=torch.float32)

    inputs = inputs.cuda() # add this line
    
    outputs = net(inputs)  # with shape NCHW
    del inputs
    
    _, predict = torch.max(outputs.data, 1)
    del outputs
        
    pred = predict[0].cpu().numpy()
    print('[%s] Finished test.' % datetime.now())
    
    del net 
    return pred
    
if __name__ == '__main__':  
    input_path = 'D:/data/xa098_b4321.tif'
    import gdal,cv2
#     ds = gdal.Open(input_path)
#     img = ds.ReadAsArray(50,60,256,256)
#     img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    
# #     img = '/tmp/data/6405_2015_56_02_LC81290342015244LGN00.jpg'
# #     import cv2 
# #     array = cv2.imread(img,-1)
# #     array = np.transpose(array,(2,0,1))
#     pred_arr = unet_predict(img)
#     from PIL import Image
#     pred_arr = pred_arr.astype('int8')
#     im = Image.fromarray(pred_arr)
#     im =im.convert("L")
#     im.save('D:/tmp/unet.jpg')
    import json
    from PIL import Image
    masks_dir = os.path.join(BASEDIR,'data/processed/')
    images_dir = os.path.join(BASEDIR,'data/raw/')
    test_tiles_list = os.path.join(masks_dir,'test_file_list_11_15_14_06.txt')
    d = open(test_tiles_list, "r",encoding='UTF-8')
    out_test = d.read()
    test_file_names = json.loads(out_test)
    
    ds = gdal.Open(os.path.join(images_dir,test_file_names[0][0]))
    img = ds.ReadAsArray(test_file_names[0][1],test_file_names[0][2],BLOCK_SIZE,BLOCK_SIZE)
    
    img_ = Image.fromarray(np.uint8(img.transpose(1,2,0)))
    img_.save('/home/zjh/tmp/road_img.bmp') 
    
    gt_ds = gdal.Open(os.path.join(masks_dir,'4_mask.bmp'))
    gt = gt_ds.ReadAsArray(test_file_names[0][1],test_file_names[0][2],BLOCK_SIZE,BLOCK_SIZE)
    print(gt)
    
    gt_ = Image.fromarray(gt)
    gt_ =gt_.convert("L")
    gt_.save('/home/zjh/tmp/road_gt.bmp') 
    
    # img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    pred_arr = unet_predict(img)
    pred_arr = pred_arr.astype('int8')
    im = Image.fromarray(pred_arr)
    im =im.convert("L")
    im.save('/home/zjh/tmp/road_test.bmp')
    
    
    
    

#     
#     array1 = ds.ReadAsArray(500,200, 256,256)
#     array1 = np.array(array1, dtype='int16')
#     pred_a = deeplabv3_predict(array1)
    
  
     
    del ds

#     
#     gc.collect()
#     time.sleep(5)
    



    
    
