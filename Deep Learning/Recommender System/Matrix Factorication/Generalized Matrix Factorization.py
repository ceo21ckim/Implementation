import torch.nn as nn 
import torch 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset

class GMF(nn.Module):
    def __init__(self, user_id, item_id, latent_dim):
        self.user_id = user_id
        self.item_id = item_id 
        self.latent_dim = latent_dim 
        
        self.user_embedding = nn.Embedding(self.user_id, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_id, self.latent_dim)
        
        self.predict_layer = nn.Linear(latent_dim, 1)
        
    def _init_weight_(self):
        nn.init.normal_(self.user_embedding.weight, std = 0.01)
        nn.init.normal_(self.item_embedding.weight, std = 0.01)
        nn.init.kaiming_normal_(self.predict_layer, a = 1, nonlinearity='sigmoid')
        
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        
        layer1 = torch.matmul(user_vec, item_vec)
        
        layer2 = self.predict_layer(layer1)
        
        return layer2
    
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

epochs = 30 

class RateDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, target_tensor):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor 
        self.target_tensor = target_tensor 
        
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]
    
    def __len__(self):
        return self.user_tensor.size(0)


user_tensor = torch.Tensor([key[0] for key in data.keys()])
item_tensor = torch.Tensor([key[1] for key in data.keys()])

rating_tensor = torch.FloatTensor([value for value in data.values()])
