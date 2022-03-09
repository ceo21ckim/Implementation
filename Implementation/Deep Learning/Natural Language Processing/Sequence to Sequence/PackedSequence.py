import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np 
import pandas as pd 
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, Dataset

# sequence의 길이가 다를 경우 pad token을 넣어주거나 packing을 해준다.
# packing작업이 pytorch에서 제대로 작동되려면 길이 기준 내림차순으로 정렬이 되어야한다.

# pack_sequence
# pad_sequence 
# pad_packed_sequence
# pack_padded_sequence

