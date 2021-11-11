import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torch.optim as optim


# import asyncio
# asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# import visdom
import visdom 
vis = visdom.Visdom()
# visdom server가 끊기면 안된다. 

# Text
vis.text('Hello, world!', env = 'main')


# Image 
a = torch.randn(3, 200, 200)
vis.image(a)

# images
vis.images(torch.Tensor(3, 3, 28, 28))

# example (using MNIST and CIFAR-10)

MNIST = dsets.MNIST(root = 'MNIST_data/', train = True, transform=transforms.ToTensor(), download=False)

data = MNIST.__getitem__(0)
print(data[0].shape)
vis.image(data[0], env='main')

data_loader = DataLoader(dataset = MNIST, batch_size = 32, shuffle = False)

for num, value in enumerate(data_loader):
    value = value[0]
    print(value.shape)
    vis.images(value)
    if num == 3:
        break

vis.close(env='main') # 모든 창이 꺼진다.

# Line Plot
# x 값을 지정하지 않으면 x축의 기본은 0~1값을 가진다. 
y_data =torch.rand(5)
plt = vis.line(Y=y_data)

x_data = torch.Tensor([1, 2, 3, 4, 5])
plt = vis.line(Y = y_data, X = x_data)

# Line update

y_append = torch.rand(1)
x_append = torch.Tensor([6])

vis.line(Y=y_append, X=x_append, win=plt, update = 'append')

# multiple Line on single windows
num= torch.Tensor(list(range(10)))
num = num.view(-1, 1)
num = torch.cat((num, num), dim = 1)

plt = vis.line(Y=torch.randn(10,2), X = num)


# Line info
plt = vis.line(Y= y_data, X = x_data, opts = dict(title = 'Test', showlegend =True))
plt = vis.line(Y = y_data, X = x_data, opts = dict(title = 'Test', legend = ['1번'], showlegend = True))
plt = vis.line(Y=torch.randn(10, 2), X = num, opts = dict(title = 'Test', legend = ['1번', '2번'], showlegend = True))



# make function for update line
def loss_tracker(loss_plot, loss_value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
    Y = loss_value, 
    win = loss_plot, 
    update = 'append'
    )

plt = vis.line(Y=torch.Tensor(1).zero_())

for i in range(500):
    loss = torch.randn(1) + i 
    loss_tracker(plt, loss, torch.Tensor([i]))

import visdom 
vis = visdom.Visdom()
vis.close(env='main')


# make loss plt
loss_plt = vis.line(Y= torch.zeros(1), opts = dict(title = 'loss_tracker', legend = ['train_loss'], showlegend = True))
