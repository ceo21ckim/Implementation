import torch 
import torch.utils.model_zoo as model_zoo
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transfrom
import torchvision.datasets as dsets
import torchvision.models.resnet as resnet 


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, stride = stride)

# basic ###################################################################
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # x.shape = 3x64x64
        identity = x 

        # identity = 3x64x64
        out = self.conv1(x) # 3x3 stride = stride 

        # out.shape = 3x32x32
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out.shape = 3x32x32
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) # downsample : stride가 1이 아니라 다른 값이 들어 갔을 경우 !

        # out.shape = 3x32x32
        # identity = 3x64x64
        # shape가 맞지 않아서 덧셈이 되지 않는 경우 downsample을 통해서 identity의 shape을 맞추어준다.
        out += identity
        out = self.relu(out)

        return out



# Bottleneck ################################################################
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)

        # ResNet Bottleneck 구조에서는 마지막 layer에 64 에서 256로 증가하기때문에 expansion을 통해 증폭시켜준다.
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x) # 1x1 stride = 1
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out) # 3x3 stride = stride
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out) # 1x1 stride = 1
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x) 


        out += identity
        out = self.relu(out)

        return out


