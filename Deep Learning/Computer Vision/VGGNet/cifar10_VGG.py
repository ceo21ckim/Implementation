import torch
import torch.nn as nn
import torchvision.datasets as dsets 
import torchvision.transforms as transform
import torch.optim as optim 
from torch.utils.data import DataLoader
# import visdom 


trans = transform.Compose(
    [transform.ToTensor(),
     transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


train_set = dsets.CIFAR10(root = 'cifar/', train = True, transform = trans, download = True)

train_loader = DataLoader(dataset=train_set, batch_size = 512, shuffle=True, drop_last=True)

test_set = dsets.CIFAR10(root = 'cifar/', train = False, transform = trans, download = True)

test_loader = DataLoader(dataset=test_set, batch_size = 512, shuffle=True, drop_last=True)



class VGG(nn.Module):
    def __init__(self, feature, num_classes = 1000, init_weights = True):
        super(VGG, self).__init__()

        self.feature = feature
        self.classifier = nn.Sequential( 
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        if init_weights :
            self._initialize_weights()

    
    def forward(self,x):
        x = self.feature(x) 
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

# cifar image size = 32 32 
cfg = [64, 64, 'M', 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512, 'M']


feature = make_layers(cfg, batch_norm=True)

model = VGG(feature, num_classes=10, init_weights=True).to('cuda')

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

lr = 0.01
epochs = 3
total_batchsize = len(train_loader)


# loss function 
criterion = nn.CrossEntropyLoss().to('cuda')

# optimizer 
optimizer = optim.Adam(model.parameters(), lr = lr)

# lr_che !
# lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.9)


model.train()

for epoch in range(epochs+1):
    avg_loss = 0
#     lr_sche.step()
    
    for img, label in train_loader:
        img = img.to('cuda')
        label = label.to('cuda')
        
        optimizer.zero_grad()
        y_pred = model(img)
        
        loss = criterion(y_pred, label)
        
        loss.backward()
        optimizer.step()
        
        avg_loss += loss
    avg_loss /= total_batchsize 
    
    print('epoch : {0} / {1}, loss : {2:.6f}'.format(epoch+1, epochs, avg_loss))
print(' finished train! ')

model.eval()

with torch.no_grad():
    accuracy = 0 
    total_batchsize = len(test_loader)
    
    for img, label in test_loader:
        img = img.to('cuda')
        label = label.to('cuda')
        
        y_pred = model(img)
        
        correct_prediction = torch.argmax(y_pred, 1) == label
        
        accuracy += correct_prediction.sum().item()
    accuracy /= float(len(test_set))
    print( 'accuracy : {}%'.format(accuracy*100))
