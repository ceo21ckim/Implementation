'''
@Author : Dong Eon 
@Date : 2022-01-15 14:44:00
@LastEditors : Dong Eon 
@LastEditTime : 2022-01-17 05:01
@Email : ponben@naver.com
@Description :
'''

import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset 


class MF(nn.Module): # class 
    def __init__(self, num_users, num_items, latent_dim): # constructor
        super(MF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        # self.predict_layer = torch.ones(latent_dim, 1)
        
        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)
        
        self.user_bias = nn.Embedding(self.num_users, 1)
        self.user_bias.weight.data = torch.zeros(self.num_users, 1).float()
        
        self.item_bias = nn.Embedding(self.num_items, 1)
        self.item_bias.weight.data = torch.zeros(self.num_items, 1).float()
        
    def forward(self, user_indices, item_indices): # method
        user_vec = self.user_embedding(user_indices)
        item_vec = self.item_embedding(item_indices)
        
        dot = torch.mul(user_vec, item_vec).sum(dim = 1) # mul : element-wise multiplication ( 내적 )
        
        # dot = torch.matmul(user_vec * item_vec, predict_layer).veiw(-1)

        return dot


class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor 
        self.target_tensor = target_tensor 
        
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)

############################################## test ##############################################
data = {(0,0): 4, # (user_id, item_id) : rating
        (0,1): 5, 
        (0,2): 3,
        (0,3): 4, 
        (1,0): 5, 
        (1,1): 3,
        (1,2): 4, 
        (1,3): 1, 
        (2,0): 3,
        (2,1): 2, 
        (2,2): 5, 
        (2,3): 5,
        (3,0): 4, 
        (3,1): 2, 
        (3,2): 3,
        (3,3): 1
        }
Iu = {key:[0,1,2,3] for key in range(4)}



user_tensor = torch.LongTensor([key[0] for key in data.keys()])
item_tensor = torch.LongTensor([key[1] for key in data.keys()])

rating_tensor = torch.LongTensor( [value for value in data.values()])


num_user = user_tensor.unique().__len__()
num_item = item_tensor.unique().__len__()



latent_dim = 10
model = MF(num_user, num_item, latent_dim)


criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)

dataset = RateDataset(user_tensor, item_tensor, rating_tensor)
train_loader = DataLoader(dataset, batch_size = 4, shuffle = True, drop_last=False)


epochs = 30
for epoch in range(epochs):
    for bid, (user, item, rating) in enumerate(train_loader):
        rating = rating.float()
        
        predict = model(user, item)
        loss = criterion(predict, rating)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'epoch [{epoch+1}/30], Loss {loss.item():.4f}')




