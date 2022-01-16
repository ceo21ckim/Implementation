import torch.nn as nn 
import torch 
import torch.optim as optim 


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