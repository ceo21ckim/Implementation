import torch 
import random
import numpy as np 
import os
import glob
from tqdm import tqdm
from PIL import Image
import os.path as osp
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 
import torchvision.datasets as dsets
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset


# setting seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1234)
np,random.seed(1234)
random.seed(1234)

class ImageTransform():
    
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train' : transforms.Compose([
                transforms.RandomResizedCrop(
                resize, scale = (0.5, 1.0)), # 데이터 확장
                transforms.RandomHorizontalFlip(), # 데이터 확장
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val' : transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize), 
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    
    def __call__(self, img, phase = 'train'):
        return self.data_transform[phase](img)


# 개미, 벌 분류 : train 243장, val 153장
def make_datapath_list(phase = 'train'):
    rootpath = './data/hymenoptera_data/'
    target_path = osp.join(rootpath+phase+'/**/*.jpg')
    path_list = []
    
    for path in glob.glob(target_path): # 파일 경로를 가지고 온다.
        path_list.append(path)
    
    return path_list 


train_list = make_datapath_list(phase = 'train')
val_list = make_datapath_list(phase = 'val')



class HymenopteraDataset(Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]
            
        if label == 'ants':
            label = 0
            
        elif label == 'bees':
            label = 1
            
        return img_transformed, label
        

size = 2242
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_dataset = HymenopteraDataset(file_list = train_list, transform=ImageTransform(size, mean, std), phase='train')
val_dataset = HymenopteraDataset(file_list = val_list, transform=ImageTransform(size, mean, std), phase = 'val')

batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True, drop_last=True)


net = models.vgg16(pretrained = True)
net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias = True)

criterion = nn.CrossEntropyLoss().to(device)

# 파인튜닝으로 학습할 파라미터를 저장한다.
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_param_names_1 = ['features']
update_param_names_2 = ['classifier.0.weight', 
                       'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias']
update_param_names_3 = ['classifier.6.weight', 'classifier.6.bias']

# 파라미터를 각 리스트에 저장
for name, param in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print('params_to_update_1에 저장: ', name)
        
    elif name in update_param_names_2 :
        param.requires_grad = True
        params_to_update_2.append(param)
        print('params_to_update_2에 저장: ', name)
        
    elif name in update_param_names_3 :
        param.requires_grad = True
        params_to_update_3.append(param)
        print('params_to_update_3에 저장: ', name)
    
    else:
        param.requires_grad = False
        print('gradient 계산 없음. 학습하지 않음: ', name)


# 각 파라미터에 최적화를 하기 위해 learning rate를 조절해준다. 
optimizer = optim.SGD([
    {'params' : params_to_update_1, 'lr' : 1e-4,
     'params' : params_to_update_2, 'lr' : 5e-4,
     'params' : params_to_update_3, 'lr' : 1e-3
    }
])


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr = 0.004)

dataloaders_dict = {'train' : train_dataloader, 'val' : val_dataloader}

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print('사용 장치: ', device)
    
    net.to(device)
    
    # 네트워크가 어느 정도 고정되면 고속화시킨다.
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} / {num_epochs}')
        print('---------------------')
        
        for phase in ['train', 'val']:
            if phase  == 'train':
                net.train()
            else:
                net.eval()
                
            epoch_loss = 0.0
            epoch_corrects = 0
            
            if (epoch == 0 ) and (phase == 'train'):
                continue
            
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    y_pred = torch.argmax(outputs, 1)
                    correct_num = (y_pred == labels).sum()
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                    epoch_loss += loss.item()
                    epoch_corrects += correct_num
            
            epoch_loss /= len(train_dataloader)
            epoch_acc = epoch_corrects / len(train_dataloader.dataset)
            
            print(f'{phase} Loss : {epoch_loss:.4f}, acc : {epoch_acc:.4f}')


num_epochs = 10
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)

# save network parameter 
save_path = './weights_fine_tuning.pth'
torch.save(net.state_dict(), save_path)


# load pytorch network parameter
load_path = './weights_fine_tuning.pth'
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)


# GPU 상에 저장된 weight를 CPU에 로드할 경우
load_weights = torch.load(load_path, map_location = {'cuda:0' : 'cpu'})
net.load_state_dict(load_weights)

