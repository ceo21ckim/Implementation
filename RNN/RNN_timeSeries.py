import FinanceDataReader as fdr
from sklearn.preprocessing import MinMaxScaler
import torch 
import torch.optim as optim
import torch.nn as nn
import numpy as np 

# Many-to-One 
# 이전까지의 데이터를 입력 받아서 다음 종가를 예측하는 모델.
# 이론적인 부분을 공부하고 싶으면 아래의 링크로.
# https://ok-lab.tistory.com/



# Random seed to make results deterministic and reproducible
torch.manual_seed(42)

seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

# reverse order
df = fdr.DataReader('GOOG').loc[:,['Open', 'High', 'Low', 'Volume', 'Close']]
df = df[::-1]

train_size = int(len(df) * 0.7)
train_set = df[:train_size].copy()

# seq_length를 빼는 이유는..뭘까..
test_set = df[train_size-seq_length:].copy()


# scaling
minmax = MinMaxScaler()

feature_col = df.columns.tolist()

train_set = minmax.fit_transform(train_set[feature_col])
test_set = minmax.fit_transform(test_set[feature_col])

train_set.shape

trainX, trainY = [], []
testX, testY = [], []

for i, data in enumerate(train_set):
    if (i+1) % 8 == 0:
        trainY.append(data[-1])
    else:
        trainX.append(data)

for i, data in enumerate(test_set):
    if (i+1) % 8 == 0:
        testY.append(data[-1])
    else:
        testX.append(data)

trainX_tensor = torch.Tensor(trainX)
trainY_tensor = torch.Tensor(trainY)

testX_tensor = torch.Tensor(testX)
testY_tensor = torch.Tensor(testY)


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers = layers, batch_first = True)

        self.fc = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, x):
        x, _status = self.rnn(x)
        out = self.fc(x[:, -1])
        return out 

net = Net(data_dim, hidden_dim, output_dim, 1)



# loss & optimizer setting
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = learning_rate)


for i in range(iterations):

    optimizer.zero_grad()

    outputs = net(trainX_tensor)

    loss = criterion(outputs, trainY_tensor)
    loss.backward()
    optimizer.step()

    print(i, loss.item())
