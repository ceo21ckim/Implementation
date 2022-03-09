from re import M
from numpy import isin
import torch 
import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, num_layers):
        super(MLP, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        
        self.user_embedding = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.latent_dim)
        
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(self.latent_dim*2, self.latent_dim),
            nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim//2), 
            nn.ReLU(), 
            nn.Linear(self.latent_dim//2, self.latent_dim//4),
            nn.ReLU(),
            nn.Linear(self.latent_dim//4, self.latent_dim//8), 
            nn.ReLU()
        )
        
        self.predict_layer = nn.Linear(self.latent_dim//8, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        
        user_item_embedding = torch.cat([user_vec, item_vec], dim = -1)
        
        mlp_layer = self.mlp_layer(user_item_embedding)
        
        predict_layer = self.predict_layer(mlp_layer)
        rating = self.sigmoid(predict_layer)
        
        return rating
    
    def init_weight(self):
        nn.init.normal_(self.user_embedding.weight, std = 0.01)
        nn.init.normal_(self.item_embedding.weight, std = 0.01)
        
        for m in nn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.xavier_uniform_(self.predict_layer)
        
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.zero_()
                
        
        
