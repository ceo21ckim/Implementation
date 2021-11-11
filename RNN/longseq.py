 # 긴 문장의 seq가 있으면 한 번에 받을 수 없기 때문에 특정 길이만큼 잘라서 넣어야 한다.
 
from typing import Sequence
import torch 
import torch.nn as nn 
import numpy as np
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers = layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias = True)

    def forward(self, x):
        out, _status = self.rnn(x)
        out = self.fc(out)

        return out

sentence = 'if you want to build a ship, do not drum up people together to collect' 

char_set = list(set(sentence))

char_dic = dict({})
for i, c in enumerate(char_set):
    char_dic[c] = i
    
dic_size = len(char_dic)
hidden_size = 5
x_data, y_data = [], []
sequence_length = 5
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i+1:i + sequence_length + 1]
    
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])


net = Net(dic_size, hidden_size, 2)

# loss & optimizer setting
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.1)


x_one_hot = [np.eye(dic_size)[x] for x in x_data]
x_one_hot[0]
x = torch.Tensor(x_one_hot)
y = torch.LongTensor(y_data)

for i in range(100):
    optimizer.zero_grad()
    output = net(x)

    loss = criterion(output.view(-1, dic_size), y.view(-1))
    loss.backward()
    optimizer.step()

    result = output.argmax(dim=2)
    predict_str = ''
    for j, result in enumerate(result):
        print(i, j, ''.join([char_set[t] for t in result]), loss.item())
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]
