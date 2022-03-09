import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dsets 
import torchvision.transforms as transforms
import torch.nn.init 


# 순서 

# 1) 라이브러리를 불러오기 (torch, torchvision, matplotlib, etc.)
# 2) gpu 사용 설정하고 random value를 위한 seed설정! (torch.manual)
# 3) 학습에 사용되는 parameter 설정(learning_rate, training_epochs, batch_size, etc)
# 4) 데이터셋을 가져오고 loader만들기 ( torch.utils.data )
# 5) 학습 모델 만들기 class CNN(nn.Module):
# 6) Loss function (criterion)을 선택하고 optimizer 선택
# 7) 모델 학습 및 loss check
# 8) 학습된 모델의 성능 check

inputs = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 32, 3, 1, 1)
relu = nn.ReLU()
maxpool = nn.MaxPool2d(2, 2)

conv2 = nn.Conv2d(32, 64, 3, 1, 1)

out = conv1(inputs)
out2 = relu(out)
out3 = maxpool(out2)

out4 = conv2(out3)
out5 = relu(out4)
out6 = maxpool(out5)

view = out6.view(out6.size(0), -1)


linear = nn.Linear(3136, 10)
linear(view).shape


import visdom 
vis = visdom.Visdom()
vis.close(env='main')


# make loss plt
loss_plt = vis.line(Y= torch.zeros(1), opts = dict(title = 'loss_tracker', legend = ['train_loss'], showlegend = True))

def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
    Y = loss_value, 
    win = loss_plot, 
    update = 'append'
    )


##################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
if device == 'cuda': torch.cuda.manual_seed(42)

learning_rate = 0.001
training_epochs = 15
batch_size = 100 

# load dataset
mnist_train = dsets.MNIST(root='MNIST_data/', train = True, transform=transforms.ToTensor(), download=False)
mnist_test = dsets.MNIST(root='MNIST_data/', train = False, transform=transforms.ToTensor(), download=False)

train_data_loader = DataLoader(dataset = mnist_train, batch_size = batch_size, shuffle = True, drop_last=True)
test_data_loader = DataLoader(dataset = mnist_test, batch_size = batch_size, shuffle = True, drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1), 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(7 * 7 * 64, 10, bias = True)

        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)

        out = self.fc(out)

        return out

model = CNN().to(device)

# loss function
criterion = nn.CrossEntropyLoss().to(device)

# optimizer 
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

model.train()
for epoch in range(1, training_epochs+1):
    avg_loss = 0
    total_batchsize = len(train_data_loader)

    for x, y in train_data_loader:
        
        x = x.to(device)
        y = y.to(device)

        # H(x)
        y_pred = model(x)

        # loss
        loss = criterion(y_pred, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss
    
    avg_loss /= total_batchsize

    print(f'epoch : {epoch} / {training_epochs}, loss : {avg_loss:.6f}')
    loss_tracker(loss_plt, torch.Tensor([avg_loss]), torch.Tensor([epoch]))


model.eval()
with torch.no_grad():
    # 원래라면 배치 사이즈가 들어가야 하는데 배치사이즈 대신 한 번에 넣어주려고 이렇게 함.
    # mnist의 channel은 1이고 사이즈는 28 x 28 이기 때문에 이렇게 해줌!
    x_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.targets.to(device)

    prediction = model(x_test)
    correct_pred = torch.argmax(prediction, 1) == y_test 
    accuracy = correct_pred.sum() / len(y_test)

    print( f'Accuracy : {accuracy.item()}')



class deep_CNN(nn.Module):
    def __init__(self):
        super(deep_CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64,kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128,kernel_size= 3, stride = 1,  padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3*3*128, 625)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(625, 10, bias = True)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)



    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

deep_model = deep_CNN()

deep_model(torch.Tensor(1, 1, 28, 28))

