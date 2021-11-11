import torch
import torch.nn as nn 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim 
from torch.utils.data import DataLoader
import torchvision.models.resnet as resnet
resnet.resnet50()


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
    expansion = 4
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



# ResNet 50 ###############################################################################
class ResNet(nn.Module):
    # model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) #reset 50!
    def __init__(self, block, layers, num_classes=1000, zero_init_residual = False):
        super(ResNet, self).__init__()

        self.inplanes = 64 # remember

        # inputs = 3x224x224
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias=False)
        # outputs.shape = 3x112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace =True)

        # input.shape = 64x112x112
        # output.shape = 64x56x56
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mod='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # self.inplanes = 64
    # self.layer1 = self._make_layer(block, 64, 3)
    # self.layer2 = self._make_lyaer(Bottleneck, planes = 128, blocks = 4, stride = 2)
    # self.inplanes = 64
    def _make_layer(self, block, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion: # 64 != 256
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes*block.expandsion)
            )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion

            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


resnet.resnet50()
resnet.resnet18()

def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs) # 2*(2 + 2 + 2 + 2) + 1(conv1) + 1(fc) = 18
    return model

def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs) # 3*(3 + 4 + 6 + 3) + 1(conv1) + 1(fc) = 50
    return model

def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs) # 3 * (3 + 8 + 36 + 3) = 150 + 2= 152


