import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary as summary_
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

# 3x3 convolution with padding
def conv3x3(in_planes, out_planes, stride=1,groups=1,dilation=1):   
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

# 1x1 convolution 
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, down:bool = False, starting:bool=False) -> None:
        super(Bottleneck,self).__init__()
        if starting:
            down = False
        self.block = bottleneck_block(in_dim,mid_dim,out_dim,down=down)
        self.relu = nn.ReLU(inplace=True)
        if down:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2, padding=0) # size 줄어듬
        else:
            conn_layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0) # size 줄어들지 않음

        self.changedim = nn.Sequential(
            conn_layer,
            nn.BatchNorm2d(out_dim)
        )
    def forward(self, x):
        identity = self.changedim(x)
        x = self.block(x)
        x += identity
        x = self.relu(x)
        return x

def bottleneck_block(in_dim,mid_dim,out_dim,down=False):
        layers =[]
        if down:
            layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=2, padding=0))
        else:
            layers.append(nn.Conv2d(in_dim, mid_dim, kernel_size=1, stride=1, padding=0))
        layers.extend([
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_dim, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
        ])
        return nn.Sequential(*layers)

def make_layer(id_dim, mid_dim, out_dim, repeats, starting=False):
    layers = []
    layers.append(Bottleneck(id_dim,mid_dim,out_dim,down=True,starting=starting))
    for _ in range(1,repeats):
        layers.append(Bottleneck(out_dim, mid_dim, out_dim, down=False))
    return nn.Sequential(*layers)

class M5(nn.Module):   # => resnet50 custom
    def __init__(self):
        super(M5, self).__init__()
        # conv1 : 7*7,64,2
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        base_dim = 64
        self.layer2 = make_layer(base_dim, base_dim, base_dim*4, repeats=3, starting=True)
        self.layer3 = make_layer(base_dim*4, base_dim*2, base_dim*8, repeats=4)
        self.layer4 = make_layer(base_dim*8, base_dim*4, base_dim*16,repeats=6)
        self.layer5 = make_layer(base_dim*16, base_dim*8, base_dim*32,repeats=3)
        self.fc = nn.Linear(1384448,2)      #692224 * 2
        self.avgpool = nn.AvgPool2d((7,7),stride=1)
    
    def convLayer(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        output = self.avgpool(out)
        return output

    def forward(self, imgL, imgR):
        imgL = self.convLayer(imgL)
        imgR = self.convLayer(imgR)
        imgL = imgL.view(imgL.size(0),-1)
        imgR = imgR.view(imgR.size(0),-1)

        img = torch.concat((imgL,imgR),1)
        img = self.fc(img)
        return img

def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model
def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

def getModel(mName):
    ModelList = {'M5':M5()}
    if mName=='resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,2)
        return model
    elif mName=='efficientnet-b2':
        return EfficientNet.from_pretrained('efficientnet-b2', num_classes=2)
    elif mName =='efficientnet-v2':
        return models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    else:
        return ModelList[mName]

'''
summary_(model=M5().cuda(),input_size=[(3,512,512),(3,512,512)],batch_size=4)
'''
