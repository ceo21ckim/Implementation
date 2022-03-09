import torch 
import torch.nn as nn 

class Generalized_Matrix_Fatorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(Generalized_Matrix_Fatorization, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim
        
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        
        self.predict_layer = nn.Linear(latent_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, user_id, item_id):
        user_vec = self.user_embedding(user_id)
        item_vec = self.item_embedding(item_id)
        
        multiply_layer = torch.mul(user_vec, item_vec)
        predict = self.predict_layer(multiply_layer)
        output = self.sigmoid(predict)
        
        return output
    
    def init_weight(self):
        torch.nn.init.normal_(self.user_embedding.weight, std = 0.01)
        torch.nn.init.normal_(self.item_embedding.weight, std = 0.01)
        
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.zero_()