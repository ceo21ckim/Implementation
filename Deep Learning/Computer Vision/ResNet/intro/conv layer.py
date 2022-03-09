import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets


# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# example 3 by 3 filter를 만들려면 ? conv = nn.Conv2d(1, 1, 3)

nn.Conv2d(1, 1, 3)

# input type : torch.Tensor
# input shape : ( N x C x H x W ) / (batch_size, channel, height, width)

# 1)
# input image size : 227 x 227
# filter size : 11 x 11
# stride : 4
# padding : 0 
# output image size : ?

# output size = (input size - filter size + (2*padding) ) / stride +1
# 소수점이 나오는 경우 버림한 후 계산한다. 
# 64, 128
# 60 / 124
# 30 / 62 

# 26 / 58
# 13 / 29 

# 다른 사이즈일 경우 input size = (32 x 64) 로 각각 따로 계산한다. 

conv = nn.Conv2d(1, 1, 11, stride = 4, padding = 0)
inputs = torch.Tensor(1, 1, 227, 227)
inputs.shape 
out = conv(inputs)
out.shape

conv = nn.Conv2d(3, 3, 7, stride = 2, padding = 0)
inputs = torch.Tensor(10, 3, 64, 64)

inputs.shape 
out = conv(inputs)
out.shape # batch_size = 10, channels = 3, size = 29 x 29

conv = nn.Conv2d(1, 1, 5, stride = 1, padding = 2)
inputs = torch.Tensor(1, 1, 32, 32)
out = conv(inputs)
out.shape 


conv = nn.Conv2d(1, 1, 5, stride = 1, padding = 0)
inputs = torch.Tensor(1, 1, 32, 64)
out = conv(inputs)
out.shape 


conv = nn.Conv2d(1, 1, 3, stride = 1, padding = 1)
inputs = torch.Tensor(1, 1, 64, 32)
out = conv(inputs)
out.shape


# max pooling, average pooling

# torch.nn.MaxPool2d(kernel_size, stride = None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# kernel_size = filter_size 

input =torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 5, 5) # input channel : 1, output channel : 5, filter size : 5
pool = nn.MaxPool2d(2) # filter 2
out = conv1(input)
out2 = pool(out)
out.size()
out2.size()


