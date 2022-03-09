import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.models.vgg as vgg 
import torchvision.transforms as transforms
import torchvision.datasets as dsets 
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader

# 3 x 224 x 224 

cifar10 = dsets.CIFAR10(root = 'cifar10/', train=True, transform = transforms.ToTensor(), download=True)

__all__ = [
    'VGG', 'vgg11', 'vgg19', 'vgg19_bn'
]

model_urls = {
    'vgg11' : 'https://download.pytorch.org/models/vgg11-bb30ac9.pth', 
    'vgg19' : 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn' : 'https://download.pytorch.org/models/vgg19_bn_c79401a0.pth'
}

class VGG(nn.Module):
    def __init__(self, feature, num_classes = 1000, init_weights = True):
        super(VGG, self).__init__()

        self.feature = feature # conv layer 
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential( 
            nn.Linear(512 * 7 * 7, 4096),  # 이미지가 다르면 수정해주어야 한다. 
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        ) # fc layer 

        if init_weights :
            self._initialize_weights()

    
    def forward(self,x):
        x = self.feature(x) # conv 
        x = self.avgpool(x) # avgpooling 
        x = x.view(x.size(0), -1) # flatten 
        x = self.classifier(x) #fc layer 
        return x 

    def _initialize_weights(self): # weight initialization 
        for m in self.modules(): # return conv layer 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') # no use xavier, he 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) 

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)




def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size =3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]

            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v

    return nn.Sequential(*layers)

# 64 / 64


cfg = {
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # layer = 8, fc = 3 : vgg11
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # layer = 10 , fc = 3 : vgg13
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # layer = 13, fc = 3 : vgg16
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # layer = 16, fc = 3 : vgg19
    'custom' : [64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 'M', 512, 512, 512, 'M']
}

feature = make_layers(cfg['custom'], batch_norm=True)

CNN = VGG(make_layers(cfg['custom'], batch_norm=True), num_classes=10, init_weights=True)
CNN



