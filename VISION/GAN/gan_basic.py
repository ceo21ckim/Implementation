import torch 
import torch.nn as nn 

D = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid()
)

G = nn.Sequential(
    nn.Linear(100, 128),
    nn.ReLU(),
    nn.Linear(128, 784),
    # 이미지를 넣을 때 -1~1 사이로 normalization을 하기 때문에 Tanh를 써주는데, 
    # 굳이 Tanh를 하지 않더라도 Generator 가 -1~1사이로 맞춰가려고 학습을 한다. 
    nn.Tanh()
)

criterion = nn.BCELoss()

d_optimizer = torch.optim.Adam(D.parameters(), lr = 0.01)
g_optimizer = torch.optim.Adam(G.parameters(), lr = 0.01)

# Assume x be real images of shape (batch_size, 784)
# Assume z be real images of shape (batch_shze, 100)

while True:
    # Train D
    loss = criterion(D(x), 1) + criterion(D(G(z)), 0)
    loss.backward()
    d_optimizer.step()


    # train G
    # Generator 가 생성한 이미지가 진짜와 가까워지기 위해서 1로 해준다. 
    loss = criterion(D(G(z)), 1)
    loss.backward()

    g_optimizer.step()
    # Generator를 학습할 때 backward를 하지만 Discriminator의 weight들은 수정하면 안되기 때문에 g_optimizer만 step()을 해준다.