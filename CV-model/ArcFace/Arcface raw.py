import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


import math


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
        

class F1Loss(nn.Module):
    def __init__(self, classes=18, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()



class F1FocalLoss(nn.Module):
    """
    [F1 + Focal Loss]
    weight = ratio of loss
    classes = # of classes
    epsilon = epsilon to prevent division error
    gamma = gamma for focal
    alpha = loss weights of focal loss for each classes
    size_average = if True, reduction method is average
    """
    def __init__(self, weight=[0.4, 0.6], classes=9, epsilon=1e-7, gamma=1.33, alpha=None, size_average=True):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
        self.weight = weight
        if (weight[0]+weight[1])!=1.0:
            self.weight[0] = weight[0]/(weight[0]+weight[1])
            self.weight[1] = weight[1]/(weight[0]+weight[1])
        self.focal = FocalLoss(gamma=gamma, alpha=alpha, size_average=size_average)
    
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        
        fcl_loss = self.focal(y_pred, y_true)
        
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        f1_loss = 1 - f1.mean()
        
        return self.weight[0]*f1_loss + self.weight[1]*fcl_loss

class ArcFaceClassifier(nn.Module):
    """
    ArcFaceClassifier
        n_classes = the number of classes to be classified.
        feature_d = dimension of features.
        epsilon = scalar to stabilize division operation.
    """
    def __init__(self, n_classes, feature_d, epsilon=1e-6):
        super(ArcFaceClassifier, self).__init__()
        self.n_classes = n_classes
        self.feature_d = feature_d
        self.epsilon = epsilon
        self.linear = nn.Linear(feature_d, n_classes, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        
    def get_norm(self, inputs):
        with torch.no_grad():
            ##### SHAPE #####
            # B : batch size
            # F : dimension of features
            # C : the number of classes to be classified
            inputs_norm = torch.norm(inputs, dim=1) # (B, F) -> (B)
            weight_norm = torch.norm(self.linear.weight, dim=1) # (C, F) -> (C)
        return inputs_norm, weight_norm

    def forward(self, inputs):
        # --------------------------- cos(theta) & phi(theta) ------------------------
        cosine = self.linear(inputs)
        inputs_norm, weight_norm = self.get_norm(inputs)
        cosine = torch.einsum("ij,i->ij", cosine, 1/(inputs_norm+self.epsilon))
        cosine = torch.einsum("ij,j->ij", cosine, 1/(weight_norm+self.epsilon))
        
        return cosine



class ArcFaceLoss(nn.Module):
    """
    ArcFaceLoss:
        s = radius of hyper sphere. default is 45.0.
        m = penalty marin angle. default is 0.1.
        crit = criterion to be used. default is "ce". ["focal", "ce", "f1focal"]
        reduction = reduction to be applied on loss. default is mean.
        gamma = gamma for focal loss. default is 1.33.
        n_classes = the number of classes to be classified. default is 9.
        
    """
    def __init__(self, s=45.0, m=0.1, crit="ce", reduction="mean", gamma=1.33, n_classes=9):
        super().__init__()
        
        self.reduction = reduction
        self.crit_type = crit
        
        if crit == "focal":
            self.crit = FocalLoss(
                gamma=gamma,
                size_average= True if reduction=="mean" else False
            )
        elif crit == "ce":
            self.crit = nn.CrossEntropyLoss(
                size_average= True if reduction=="mean" else False
            )
        elif crit == "f1focal":
            self.crit = F1FocalLoss(
                gamma=gamma, classes=n_classes,
                size_average= True if reduction=="mean" else False
            )
        else:
            assert False, "crit is one of [bce, focal, f1focal]."

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, logits, labels):
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        s = self.s

        output = output * s
        loss = self.crit(output, labels)
        
        return loss
