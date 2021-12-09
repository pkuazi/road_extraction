import torch
import torch.nn as nn

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


if __name__ == "__main__":
    # loss = SegmentationLosses(cuda=True)
    pred = torch.tensor([[[[0.9,0.4],[0.1,0.1]],[[0.1,0.6],[0.9,0.9]]]])#1个负样本分为了正样本，共2个正样本，2个负样本
    gt = torch.tensor([[[0,0],[1,1]]])
    # a = torch.rand(1, 3, 7, 7).cuda()
    # b = torch.rand(1, 7, 7).cuda()
    a = torch.rand(1, 3, 7, 7)
    print(gt.size())
    b = torch.rand(1, 7, 7)
    print(b.size())
    # print(a)
    # print(b)
    loss = SegmentationLosses(cuda=False)
    print(loss.CrossEntropyLoss(pred,gt).item())
    # print(loss.FocalLoss(pred,gt, gamma=0, alpha=None).item())
    print(loss.FocalLoss(pred,gt, gamma=2, alpha=0.5).item())
    
    pred2 = torch.tensor([[[[0.9,0.9],[0.6,0.1]],[[0.1,0.1],[0.4,0.9]]]])#1个正样本分为了负样本
    print(loss.CrossEntropyLoss(pred2,gt).item())
    # print(loss.FocalLoss(pred2,gt, gamma=0, alpha=None).item())
    print(loss.FocalLoss(pred2,gt, gamma=0.1, alpha=0.5).item())
    
    


