'''
@Author : KIM DONG EON 
@Date : 2022-01-17 15:38
@LastEditor : KIM DONG EON 
@LastEditTime : -
@email : ponben@naver.com
'''

import torch 
import torch.nn as nn 
import numpy as np 
import torch.optim as optim 

class GMF(nn.Module):
    def __init__(self, user_id, item_id, latent_dim):
        self.user_id = user_id 
        self.item_id = item_id 
        self.latent_dim = latent_dim 
        
        self.user_embedding = nn.Embedding(self.user_id, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_id, self.latent_dim)
        
        self.logistic = torch.nn.Sigmoid()
        
        self._init_weight_()
    
    def _init_weight_(self):
        torch.nn.init.normal_(self.user_embedding.weight, std = 0.01)
        torch.nn.init.normal_(self.item_embedding.weight, std = 0.01)
        torch.nn.init.kaiming_normal_(self.predict_layer, nonlinearity='sigmoid')
        
        for m in nn.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, user, item):
        user_vec = self.user_embedding(user)
        item_vec = self.item_embedding(item)
        
        output = torch.matmul(user_vec, item_vec)

        ratings = self.logistic(output)
        
        return ratings