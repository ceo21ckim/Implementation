import torch
import torch.nn as nn 
from torch import Tensor
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torchvision import datasets as dsets
from torchvision import transforms
import torchvision.models.resnet as resnet

def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion= 1

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups= 1,
        base_width = 64,
        dilation = 1,
        norm_layer= None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
  
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride = 1,
        downsample = None,
        groups = 1,
        base_width = 64,
        dilation = 1,
        norm_layer = None ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes= 10,
        zero_init_residual = False,
        groups= 1,
        width_per_group= 64,
        replace_stride_with_dilation = None,
        norm_layer = None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(128 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ) :
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        
        x = self.conv1(x)
        
        x = self.bn1(x)

        x = self.relu(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(777)
if device == 'cuda': torch.cuda.manual_seed(7777)


# transform normalize 할 때 이전에는 0.5 0.5 0.5 로 보내버렸지만 지금은 다르게 한다.
# 정확한 normalize 기준이 아니다
transform = transforms.Compose(
    transforms.ToTensor()
)

trainset = dsets.CIFAR10(root = 'cifar10/', train = True, download=True, transform = transforms)
print(trainset.data.shape)

train_data_mean = trainset.data.mean(axis = (0, 1, 2)) # 각 축의 데이터의 mean, std를 구한다. 
train_data_std = trainset.data.std(axis = (0, 1, 2))

print(train_data_mean)
print(train_data_std)

train_data_mean = train_data_mean / 255
train_data_std = train_data_std / 255

print(train_data_mean)
print(train_data_std)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding = 4), # padding을 4로 해주고 32개를 랜덤하게 가져오게한다.
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])



trainset = dsets.CIFAR10(root = 'cifar10/', train = True, download=True, transform=transform_train)
train_loader = DataLoader(dataset=trainset, batch_size=128, shuffle=True, drop_last=True)

testset = dsets.CIFAR10(root = 'cifar10/', train = False, download=True, transform = transform_test)
test_loader = DataLoader(dataset=testset, batch_size=128, shuffle=True, drop_last=True)

calsses = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


conv1x1 = resnet.conv1x1
Bottleneck = resnet.Bottleneck
BasicBlock = resnet.BasicBlock
resnet50 = ResNet(resnet.Bottleneck, [3, 4, 6, 3], 10, True).to(device)
sample = torch.Tensor(1,3, 32, 32).to(device)
resnet50(sample)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(resnet50.parameters(), lr = 0.005)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size =10, gamma=0.5)


epochs = 40
##################################### test
resnet50.train()
for epoch in range(1, epochs+1):
    avg_loss = 0
    total_batchsize = len(train_loader)
    
    for i,data in enumerate(train_loader):
        img, label = data
        img = img.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        y_pred = resnet50(img)
        
        loss = criterion(y_pred, label)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        if i % 30 == 0 :
            print('epoch {}, step {}'.format(epoch, i))
    avg_loss /= float(total_batchsize)
    print('epoch {0} / {1}, loss : {2:.6f}'.format(epoch, epochs, avg_loss ))
  
print('Finished')

######################################### test
resnet50.eval()
accuracy = 0
with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        
        y_pred = resnet50(img)
        
        correct_prediction = torch.argmax(y_pred, 1) == label
        
        accuracy += correct_prediction.sum().item()
    accuracy /= float(len(test_loader))
    print('acc : {}'.format(accuracy))