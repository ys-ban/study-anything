from torch import nn
from torch.nn import functional as F
import torch
from timm import models

class ResNetBased(nn.Module):
    def __init__(self, drop_p = 0.5, num_classes = 3, resnet_type = 50):
        super(ResNetBased, self).__init__()
        self.drop_p = drop_p
        if resnet_type not in [18, 34, 50, 101, 152]:
            raise("Please check the resnet type.\nIt should be one of 18, 34, 50, 101 or 152.")
        self.resnet_type = "ResNet"+str(resnet_type)
        if resnet_type==18:
            self.backbone = models.resnet18(True)
        elif resnet_type==34:
            self.backbone = models.resnet34(True)
        elif resnet_type==50:
            self.backbone = models.resnet50(True)
        elif resnet_type==101:
            self.backbone = models.resnet101(True)
        else:
            self.backbone = models.resnet152(True)
        self.added_fc = nn.Linear(1000, self.num_classes)
    
    def forward(self, x):
        net = self.backbone(x)
        net = F.dropout(net, self.drop_p, self.training)
        return self.added_fc(net)


class Inceptionv3Based(nn.Module):
    def __init__(self, drop_p = 0.5, num_classes = 3):
        super(Inceptionv3Based, self).__init__()
        self.drop_p = 0.5
        self.backbone = models.inception_v3(True)
        self.fc = nn.Linear(1000, num_classes)
    
    def forward(self, x):
        if self.training:
            net, aux = self.backbone(x)
        else:
            net = self.backbone(x)
        # print(f"{net.shape}")
        net = F.dropout(net, self.drop_p, self.training)
        return self.fc(net)

# nasnet = timm.models.nasnet.NASNetALarge()
# effi_b3 = timm.models.efficientnet_b3()
# effi_b4 = timm.models.efficientnet_b4()

class EffiB3Based(torch.nn.Module):
    def __init__(self, num_classes = 3, drop_p = 0.5):
        super(EffiB3Based, self).__init__()
        self.backbone = models.efficientnet_b3(True)
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.fc_last = torch.nn.Linear(1000, self.num_classes)
    
    def forward(self, x):
        net = self.backbone(x)
        net = F.dropout(net, self.drop_p, self.training)
        return self.fc_last(net)

class EffiB4Based(torch.nn.Module):
    def __init__(self, num_classes = 3, drop_p = 0.5):
        super(EffiB4Based, self).__init__()
        self.backbone = models.efficientnet_b4(True)
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.fc_last = torch.nn.Linear(1000, self.num_classes)
    
    def forward(self, x):
        net = self.backbone(x)
        net = F.dropout(net, self.drop_p, self.training)
        return self.fc_last(net)

class NasNetBased(torch.nn.Module):
    def __init__(self, num_classes = 3, drop_p = 0.5):
        super(NasNetBased, self).__init__()
        self.backbone = models.nasnetalarge(True)
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.fc_last = torch.nn.Linear(1000, self.num_classes)
    
    def forward(self, x):
        net = self.backbone(x)
        net = F.dropout(net, self.drop_p, self.training)
        return self.fc_last(net)

