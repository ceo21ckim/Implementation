import numpy as np
from numpy.testing._private.utils import requires_memory
import pandas as pd

import FinanceDataReader as fdr

fdr.__version__

KRX = fdr.StockListing('KRX')

import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'NanumGothic'

start = datetime.datetime(2000,1,1)
end = datetime.date.today()
samsung = fdr.DataReader('005930', start = start)
samsung

samsung['Close'].plot()
plt.show()

del samsung['Volume']
del samsung['Change']

x = samsung.iloc[:,:-1]
y = samsung[['Close']]

from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
rb, mm, sd = RobustScaler(), MinMaxScaler(), StandardScaler()
x_rb, x_mm, x_sd = rb.fit_transform(x), mm.fit_transform(x), sd.fit_transform(x)
y_rb, y_mm, y_sd = rb.fit_transform(y), mm.fit_transform(y), sd.fit_transform(y)

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
x_train, x_test, y_train, y_test = train_test_split(x_rb, y_rb, test_size = 0.3, random_state = 42)
x_train_tensor = Variable(data = torch.Tensor(x_train))
x_test_tensor = Variable(data = torch.Tensor(x_test))

y_train_tensor = Variable(data = torch.Tensor(y_train))
y_test_tensor = Variable(data = torch.Tensor(y_test))

x_train_tensor_final = torch.reshape(x_train_tensor, (x_train_tensor.shape[0], 1, x_train_tensor.shape[1]))
x_test_tensor_final = torch.reshape(x_test_tensor, (x_test_tensor.shape[0], 1, x_test_tensor.shape[1]))


device = torch.device( "cpu")  # device


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU() 

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device) #internal state   
        # Propagate input through LSTM

        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
    
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
    
        return out 

num_epochs = 30000 #1000 epochs
learning_rate = 0.00001 #0.001 lr

input_size = 5 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, x_train_tensor_final.shape[1]).to(device)

loss_function = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate)  # adam optimize



for epoch in range(num_epochs):
    outputs = lstm1.forward(x_train_tensor_final.to(device)) #forward pass
    optimizer.zero_grad() #caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = loss_function(outputs, y_train_tensor.to(device))

    loss.backward() #calculates the loss of the loss function

    optimizer.step() #improve from loss, i.e backprop
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))



df_X_ss = sd.transform(samsung.drop(columns='Volume'))
df_y_mm = mm.transform(samsung.iloc[:, 5:6])

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))
train_predict = lstm1(df_X_ss.to(device))#forward pass
data_predict = train_predict.data.detach().cpu().numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=4500, c='r', linestyle='--') #size of the training set

plt.plot(dataY_plot, label='Actuall Data') #actual plot
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show()