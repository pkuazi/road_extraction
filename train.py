# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 09:07:08 2021

@author: zjh
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from configure import BASEDIR,BLOCK_SIZE,NUM_CLASSES,IN_CHANNELS
from dataloaders.datasets import SegmentationDataset
from torch.utils.data import DataLoader
import json
import torch
from modeling.myscse import SCSEUnet
from modeling.loss import SegmentationLosses
from utils.metrics import Evaluator
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data():
    images_dir = os.path.join(BASEDIR,'data/raw/')
    masks_dir = os.path.join(BASEDIR,'data/processed/')
    
    b = open(os.path.join(masks_dir,'train_file_list_11_15_14_06.txt'), "r",encoding='UTF-8')
    out_train = b.read()
    train_file_names = json.loads(out_train)
    
    c = open(os.path.join(masks_dir,'val_file_list_11_15_14_06.txt'), "r",encoding='UTF-8')
    out_val = c.read()
    val_file_names = json.loads(out_val)
    
    d = open( os.path.join(masks_dir,'test_file_list_11_15_14_06.txt'), "r",encoding='UTF-8')
    out_test = d.read()
    test_file_names = json.loads(out_test)
    
    # 构造训练loader
    train_set = SegmentationDataset(train_file_names,images_dir,masks_dir,BLOCK_SIZE, transform_name = 'train_transform_1')   
    
    # 构造验证loader
    val_set = SegmentationDataset(val_file_names,images_dir,masks_dir,BLOCK_SIZE, transform_name = None)
    
    # 构造测试loader
    test_set = SegmentationDataset(test_file_names,images_dir,masks_dir,BLOCK_SIZE, transform_name = None)
    
    return train_set, val_set, test_set

def train(learning_rate,batch_size):
    net = SCSEUnet(IN_CHANNELS,NUM_CLASSES)
    net = net.to(device)
    summary(net,(IN_CHANNELS,BLOCK_SIZE,BLOCK_SIZE))
    
    #loss and optimizer
    criterion = SegmentationLosses().FocalLoss
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
    
    train_set, val_set, test_set=load_data()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    evaluator = Evaluator(NUM_CLASSES)

    #train the model
    num_epochs=200
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}\n------------------")
        
        #training
        running_loss=0.0
        for batch_idx, data in enumerate(train_loader):
            inputs,labels=data['image'],data['mask']
            inputs, labels = inputs.to(device), labels.to(device)
            
            
            #compute prediction and loss
            outputs=net(inputs)
            
            # print('train loop output shape',outputs.shape)
            loss = criterion(outputs, labels)
            # loss = criterion(outputs, labels,gamma=0.1,alpha=3)#bridge 
            
            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss+=loss.item()
        running_loss/=len(train_loader)
        print(f"loss:{running_loss:>7f}")
                # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # validation loss
        evaluator.reset()  # 创建全为0的混淆矩阵
        num_batches = len(valid_loader)
        # print('length of dataloader', num_batches)
        # num_images = len(valid_loader.dataset)
        # print('length of dataset', num_images)
        val_loss=0
        
        with torch.no_grad():
            for data in valid_loader:
                X,target=data['image'],data['mask']
                X,target=X.to(device),target.to(device)
                outputs=net(X)
                
                val_loss+= criterion(outputs,target).item()
                
                outputs=outputs.argmax(1)   
                # print('gt ',target)
                # print('output ', outputs)
                # break
                
                # Add batch sample into evaluator
                target = target.cpu().numpy()
                pred = outputs.data.cpu().numpy()
                
                evaluator.add_batch(target, pred)
                
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比        
        val_loss/=num_batches
        # correct/=num_images
        print(f"Validation Error: FWIoU Accuracy: {(100*FWIoU):>0.3f}%, Avg loss: {val_loss:>8f} \n")
        
    print('Finished Training')
    torch.save(net, '/home/zjh/road_extraction/road_extraction/run/batch16_model.pth')

def test_accuracy(net, device='cpu'):  
    from PIL import Image
    net=net.to(device)
    train_set, val_set, test_set=load_data()
    test_loader = DataLoader(test_set,batch_size=1, shuffle=False)    
    #test the model
    evaluator = Evaluator(NUM_CLASSES)
    with torch.no_grad():
        id=0
        for data in test_loader:
            inputs,labels=data['image'],data['mask']
            
            img=inputs[0].numpy()
            img_ = Image.fromarray(np.uint8(img.transpose(1,2,0)))
            img_.save('/home/zjh/tmp/road_img_%s.bmp'%id) 
            mask=labels[0].numpy()
            gt = Image.fromarray(mask)
            gt =gt.convert("L")
            gt.save('/home/zjh/tmp/road_gt_%s.bmp'%id)

            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs=net(inputs)# with shape NCHW
            
            # _, predicted = torch.max(outputs.data, 1)   
            predicted=outputs.argmax(1)   
            
            # Add batch sample into evaluator
            target = labels.cpu().numpy()
            pred = predicted.data.cpu().numpy()
            # print('predicted max shape ', pred.shape)
            
            evaluator.add_batch(target, pred)
            
            pred_arr = pred[0].astype('int8')
            im = Image.fromarray(pred_arr)
            im =im.convert("L")
            im.save('/home/zjh/tmp/road_test_bridge_%s.bmp'%id)
            
            id+=1
            
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
            
        print(f'FWIoU Accuracy of the network on the test images: {(FWIoU*100):>0.3f} %')
def tta_test(net):
    from PIL import Image
    model=net.to(device)
    train_set, val_set, test_set=load_data()
    test_loader = DataLoader(test_set,batch_size=1, shuffle=False)    
    #test the model
    evaluator = Evaluator(NUM_CLASSES)
    with torch.no_grad():
        id=0
        for data in test_loader:
            inputs,labels=data['image'],data['mask']
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            output=model(inputs)
            #horizontalflip
            hfliped_image=inputs.flip(3)
            hfliped_label=model(hfliped_image)
            label1=hfliped_label.flip(3)
            # print('label1, ',label1)
            
            #verticalflip
            vfliped_image=inputs.flip(2)
            vfliped_label=model(vfliped_image)
            label2=vfliped_label.flip(2)
            
            #rotate90
            rotated_image=torch.rot90(inputs, 1,(2,3))
            rotated_label=model(rotated_image)
            label3=torch.rot90(rotated_label, 3,(2,3))
            
            # outputs=net(inputs)# with shape NCHW
            
            # augmented labels merge
            outputs =(output+label1+label2+label3)/4
            # print('mean label, ',outputs)
            # _, predicted = torch.max(outputs.data, 1)   
            predicted=outputs.argmax(1)   
            
            # Add batch sample into evaluator
            target = labels.cpu().numpy()
            pred = predicted.data.cpu().numpy()
            # print('predicted max shape ', pred.shape)
            
            evaluator.add_batch(target, pred)
            
            pred_arr = pred[0].astype('int8')
            im = Image.fromarray(pred_arr)
            im =im.convert("L")
            im.save('/home/zjh/tmp/road_test_bridge_%s.bmp'%id)
            
            id+=1
            
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
            
        print(f'FWIoU Accuracy of TTA: {(FWIoU*100):>0.3f} %')
        
    # import ttach as tta
    # # transforms = tta.Compose(
    # #     [
    # #         tta.HorizontalFlip(),
    # #         tta.VerticalFlip(),
    # #         tta.Rotate90(angles=[0, 180]),
    # #         # tta.Scale(scales=[1, 2, 4]),
    # #         # tta.Multiply(factors=[0.9, 1, 1.1]),        
    # #     ]
    # # )
    # # assert len(transforms) == 2 * 2 * 2   # all combinations for aug parameters
    
    # model=model.to(device)
    # train_set, val_set, test_set=load_data()
    # test_loader = DataLoader(test_set,batch_size=1, shuffle=False)    
    # #test the model
    # evaluator = Evaluator(NUM_CLASSES)
    # transforms=[tta.HorizontalFlip(),
    #         tta.VerticalFlip(),
    #         tta.Rotate90(angles=[0, 180]),]
    
    # with torch.no_grad():
    #     id=0
    #     for data in test_loader:
    #         image,labels=data['image'],data['mask']
            
    #         #horizontalflip
    #         hfliped_image=image.flip(3)
    #         hfliped_label=model(hfliped_image)
    #         label1=hfliped_label.flip(3)
            
    #         #verticalflip
    #         vfliped_image=image.flip(2)
    #         vfliped_label=model(vfliped_image)
    #         label2=vfliped_label.flip(2)
            
    #         #rotate90
    #         rotated_image=torch.rot90(image, 1,(2,3))
    #         rotated_label=model(rotated_image)
    #         label3=torch.rot90(rotated_label, 3,(2,3))
            
    #         #TODO 取三个label各元素的众数，考虑使用pandas
            
            
    #         for transformer in transforms: # custom transforms or e.g. tta.aliases.d4_transform()  
    #             # augment image
    #             augmented_image = transformer.apply_aug_image(image,apply=True)  
            
    #             # pass to model
    #             model_output = model(augmented_image)   
                
    #             # reverse augmentation for mask and label
    #             # deaug_mask = transformer.deaugment_mask(model_output)
    #             deaug_label = transformer.deaugment_label(model_output)     
                # save results
    #             labels.append(deaug_mask)
    #             masks.append(deaug_label)
        
    # # reduce results as you want, e.g mean/max/min
    # label = mean(labels)
    # mask = mean(masks)
# valid_epoch = smp.utils.train.ValidEpoch(
#     model, 
#     loss=loss, 
#     metrics=metrics, 
#     device=DEVICE,
#     verbose=True,
# )

# # train model for 40 epochs
# max_score = 0

# for i in range(0, 40):
#     print('\nEpoch: {}'.format(i))
#     train_logs = train_epoch.run(train_loader)
#     valid_logs = valid_epoch.run(valid_loader)
    
#     # do something (save model, change lr, etc.)
#     if max_score < valid_logs['iou_score']:
#         max_score = valid_logs['iou_score']
#         torch.save(model, os.path.join(masks_dir,'best_model.pth'))
#         print('Model saved!')
        
#     if i == 25:
#         optimizer.param_groups[0]['lr'] = 1e-5
#         print('Decrease decoder learning rate to 1e-5!')


# # load best saved checkpoint
# best_model = torch.load( os.path.join(masks_dir,'best_model.pth'))


# logs = test_epoch.run(test_loader)
def main():
    #hyper-parameters
    train(0.0001,16)
    net=torch.load('/home/zjh/road_extraction/road_extraction/run/batch16_model.pth')
    net.eval()
    test_accuracy(net,'cuda')
    tta_test(net)
    print('batch size is 16')
    
    
if __name__=="__main__":
    main()

    
    #使用真实bridge切片,测试focalloss的参数
#     import os,gdal
#     images_dir = os.path.join(BASEDIR,'data/raw/')
#     masks_dir = os.path.join(BASEDIR,'data/processed/')
#     imgds=gdal.Open(os.path.join(images_dir,'4.bmp'))
                    
#     inputs = imgds.ReadAsArray(0,0,512,512).astype(np.float32)
#     gtds=gdal.Open(os.path.join(masks_dir,'4_mask.bmp'))
#     gt=gtds.ReadAsArray(0,0,512,512).astype(np.float32)
    
#     def reclassify(target,origin_labels, output_labels):
#         h, w = target.shape
#         label = np.zeros((h, w), dtype=np.int8)
#         # label[target == 6] = 1
#         # label[target == 9] = 2
#         # label[target == 7] = 3
#         # label[target == 1] = 4
#         # label[target == 3] = 5
#         for i in range(len(origin_labels)):
#             label[target==origin_labels[i]]=output_labels[i]
    
#         # label = torch.as_tensor(label, dtype=torch.long)
#         return label
    
#     gt = reclassify(gt,[0,1,2,3,4],[0,0,0,0,1] )
    
    
#     inputs, gt = torch.tensor(np.array([inputs])).to('cuda'),  torch.tensor(np.array([gt])).to('cuda')
    
#     net=torch.load(os.path.join(BASEDIR,'run/bridge_model.pth'))
#     net.eval()
#     model=net.to(device)
#     pred2=model(inputs)
    
#     loss = SegmentationLosses(cuda=False)
#     print('CE ', loss.CrossEntropyLoss(pred2,gt).item())
#     print('0,None ',loss.FocalLoss(pred2,gt, gamma=0, alpha=None).item())
#     print('0.01,0.5 ',loss.FocalLoss(pred2,gt, gamma=0.01, alpha=0.5).item())
#     print('0.1,0.5 ',loss.FocalLoss(pred2,gt, gamma=0.1, alpha=0.5).item())
#     print('0.1,2 ',loss.FocalLoss(pred2,gt, gamma=0.1, alpha=2).item())
#     print('0.1,3 ',loss.FocalLoss(pred2,gt, gamma=0.1, alpha=3).item())
#     print('1,0.5 ',loss.FocalLoss(pred2,gt, gamma=1, alpha=0.5).item())
#     print('2,0.5 ',loss.FocalLoss(pred2,gt, gamma=2, alpha=0.5).item())
#     print('3,0.5 ',loss.FocalLoss(pred2,gt, gamma=3, alpha=0.5).item())
#     print('2,1 ',loss.FocalLoss(pred2,gt, gamma=2, alpha=1).item())
#     print('2,2 ',loss.FocalLoss(pred2,gt, gamma=2, alpha=2).item())
    
#     CE  0.3212392330169678
# 0,None  0.3212392330169678
# 0.01,0.5  0.1585579216480255
# 0.1,0.5  0.1411537379026413
# 0.1,2  0.5646149516105652
# 0.1,3  0.8469224572181702  #small gamma, large alpha, large loss
# 1,0.5  0.04413028806447983
# 2,0.5  0.012124809436500072
# 3,0.5  0.003331294981762767
# 2,1  0.024249618873000145
# 2,2  0.04849923774600029
