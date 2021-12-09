import argparse
import os,json
import numpy as np
from tqdm import tqdm

# import utils.preddataset as preddataset

# from mypath import Path
# from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
# from modeling.deeplab import *
from modeling.UNet_SNws import *
# from modeling.UNet3Plus import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from utils.TruncatedLoss import TruncatedLoss
from utils.progress_bar import progress_bar
from configure import BASEDIR,BLOCK_SIZE,NUM_CLASSES
from dataloaders.datasets import SegmentationDataset
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def make_data_loader():
    images_dir = os.path.join(BASEDIR,'data/raw/')
    masks_dir = os.path.join(BASEDIR,'data/processed/')
    
    train_tiles_list=os.path.join(masks_dir,'train_file_list_11_15_14_06.txt')
    val_tiles_list = os.path.join(masks_dir,'val_file_list_11_15_14_06.txt')
    test_tiles_list = os.path.join(masks_dir,'test_file_list_11_15_14_06.txt')
    
    b = open(train_tiles_list, "r",encoding='UTF-8')
    out_train = b.read()
    train_file_names = json.loads(out_train)
    
    c = open(val_tiles_list, "r",encoding='UTF-8')
    out_val = c.read()
    val_file_names = json.loads(out_val)
    
    d = open(test_tiles_list, "r",encoding='UTF-8')
    out_test = d.read()
    test_file_names = json.loads(out_test)
    
    # 构造训练loader
    train_set = SegmentationDataset(train_file_names,images_dir,masks_dir,BLOCK_SIZE, transform_name = 'train_transform_1')
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # shuffle=False
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    
    # 构造验证loader
    val_set = SegmentationDataset(val_file_names,images_dir,masks_dir,BLOCK_SIZE, transform_name = 'valid_transform_3')
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    valid_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=4)
    
    # 构造测试loader
    test_set = SegmentationDataset(test_file_names,images_dir,masks_dir,BLOCK_SIZE, transform_name = None)
    # test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set,batch_size=1, shuffle=False)
    return train_loader,valid_loader,test_loader

class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        # 使用tensorboardX可视化
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}

        self.train_loader, self.val_loader, self.test_loader = make_data_loader()
        print('num of training',len(self.train_loader))
        if args.model_name == 'unet':
            model = UNet_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)     
        # elif args.model_name=='unet3+':
        #     model = UNet3Plus(args.n_channels,args.n_class)
        # #net = UNet3Plus_DeepSup(n_channels=3, n_classes=1)
        # #net = UNet3Plus_DeepSup_CGM(n_channels=3, n_classes=1)
        #     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)     
        
        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(masks_dir, args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, NUM_CLASSES)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        # self.criterion = SegmentationLosses(weight=weight, ignore_index=255, cuda=args.cuda).build_loss(mode=args.loss_type)
        # use TruncatedLoss for noisy labels
        # self.criterion = TruncatedLoss(trainset_size=len(self.train_loader)).cuda()
        self.model, self.optimizer = model, optimizer
        
        # Define Evaluator
        self.evaluator = Evaluator(NUM_CLASSES)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        
    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
#            image, target = sample[0], sample[1]
            image, target = sample['image'], sample['mask']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
#                 self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)  # 保存标量值
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % (train_loss / len(tbar)))
        # print('Loss: %.3f' % (train_loss / i))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()  # 创建全为0的混淆矩阵
        tbar = tqdm(self.val_loader, desc='\r')  # 回车符
        val_loss = 0.0
        # i=0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['mask']
#            image, target = sample[0], sample[1]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            tbar.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)  # 按行
            
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % (val_loss / len(tbar)))

        new_pred = mIoU  # FWIoU
        
        # log
        logfile = os.path.join(os.getcwd(),'log.txt')
        log_file = open(logfile, 'a')
        if epoch == 0:
            log_file.seek(0)
            log_file.truncate()
            log_file.write(self.args.model_name + '\n')
            log_file.write(str(self.args.lr) + '\n')
            log_file.write(str(NUM_CLASSES) + 'classes \n')
        log_file.write('Epoch: %d, ' % (epoch + 1))
        if new_pred < self.best_pred:
            log_file.write('Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, best_mIoU: {}, '.format(Acc, Acc_class, mIoU, FWIoU, self.best_pred))
        else:
            log_file.write('Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, best_mIoU: {}, '.format(Acc, Acc_class, mIoU, FWIoU, new_pred))
        log_file.write('Loss: %.3f\n' % (val_loss / len(tbar)))
        if epoch == 199:   # 499
            log_file.close()     
        
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")  # 创建解析器

    parser.add_argument('--dataset', type=str, default='road',
                        choices=['pascal', 'coco', 'cityscapes', 'potsdam','road'],
                        help='dataset name (default: pascal)')

    parser.add_argument('--workers', type=int, default=4,  # default=4
                        metavar='N', help='dataloader threads')


    # training hyper params
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: auto)')  # metavar参数:用来控制部分命令行参数的显示
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')

    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')  # 0.005
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')

    # cuda, seed and logging
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')  # 恢复文件
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
                        # action='store_true':只要运行时该变量有传参就将该变量设为True
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')                
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,  # False
                        help='whether to use balanced weights (default: False)')
                        # 'balanced'计算出来的结果很均衡，使得惩罚项和样本量对应，惩罚项用的样本数的倒数
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')  # comma-separated:逗号分割
    parser.add_argument('--model_name', type=str, default='unet', choices=['deeplabv3+', 'unet3+','unet'])
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--n_filters', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=5)
    parser.add_argument('--using_movavg', type=int, default=1)
    parser.add_argument('--using_bn', type=int, default=1)

    args = parser.parse_args()
    # parser.parse_args():把parser中设置的所有"add_argument"返回到args子类实例当中，那么parser中增加的属性内容都会在args实例中，使用即可
    args.cuda = torch.cuda.is_available()
    
    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'potsdam':200,
            'road':20,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        # args.batch_size = 8 * len(args.gpu_ids)
        args.batch_size =8

    if args.lr is None:
        lrs = {
            'potsdam':0.01,
            'road':0.001
        }
        args.lr = lrs[args.dataset.lower()] / 4  * args.batch_size
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')  # 用raise语句来引发一个异常
    if args.checkname is None:
        if args.model_name == 'unet':
            args.checkname = 'unet'
        elif args.model_name == 'unet3+':
            args.checkname = 'unet3+'
        elif args.model_name == 'unet3+_aspp':
            args.checkname = 'unet3+_aspp'
    print(args)
    torch.manual_seed(args.seed)  # 设置 (CPU/GPU) 生成随机数的种子，并返回一个torch.Generator对象
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # trainer.training_truncatedloss(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    test_loader = trainer.test_loader
    
    
    #predict on the test dataset
    trainer.model.eval()
    trainer.evaluator.reset()  # 创建全为0的混淆矩阵
    tbar = tqdm(trainer.test_loader, desc='\r')  # 回车符
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['mask']
#            image, target = sample[0], sample[1]
        if trainer.args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output = trainer.model(image)
        loss = trainer.criterion(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)  # 按行
        # Add batch sample into evaluator
        trainer.evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = trainer.evaluator.Pixel_Accuracy()
    Acc_class = trainer.evaluator.Pixel_Accuracy_Class()
    mIoU = trainer.evaluator.Mean_Intersection_over_Union()
    FWIoU = trainer.evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
    print('test:')
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % (test_loss / len(tbar)))

    # log
    logfile = os.path.join(os.getcwd(),'log.txt')
    log_file = open(logfile, 'a')
    log_file.write('For test dataset, Acc: {}, Acc_class: {}, mIoU: {}, fwIoU: {}, best_mIoU: {}, '.format(Acc, Acc_class, mIoU, FWIoU, trainer.best_pred))
    log_file.close()

    trainer.writer.close()


if __name__ == "__main__":
    main()
