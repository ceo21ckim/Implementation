import torch
import torch.nn as nn 
import torch.optim as optim 


class NCF(nn.Module):
    def __init__(self, config):
        super(NCF, self).__init__()
        self.config = config 
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        
        self.latent_dim_mf = config['latent_dim_mf']
        
        self.latent_dim_mlp = config['latent_dim_mlp']
        
        
        self.user_embedding_mf = nn.Embedding(self.num_users, self.latent_dim_mf)
        self.item_embedding_mf = nn.Embedding(self.num_items, self.latent_dim_mf)
        
        self.user_embedding_mlp = nn.Embedding(self.num_users, self.latent_dim_mlp)
        self.item_embedding_mlp = nn.Embedding(self.num_items, self.latent_dim_mlp)
        
        self.mlp_layer = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(16, 8), 
            nn.ReLU(), 
            nn.Dropout(0.2),     
        )
        self.predict_layer = nn.Linear(40, 1)
        
    def _init_weight_(self):
        nn.init.normal_(self.user_embedding_mf.weight, std = 0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std = 0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std = 0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std = 0.01)
        nn.init.kaiming_normal_(self.predict_layer)
        
        for m in nn.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                

    def forward(self, user, item):
        user_vec_mf = self.user_embedding_mf(user)
        item_vec_mf = self.item_embedding_mf(item)
        
        user_vec_mlp = self.user_embedding_mlp(user)
        item_vec_mlp = self.item_embedding_mlp(item)
        
        user_item_concat = torch.cat([user_vec_mlp, item_vec_mlp], dim = -1) # concat latent vector
        layer_mf = torch.mul(user_vec_mf, item_vec_mf)
        
        layer_mlp = self.mlp_layer(user_item_concat)
        
        layer_concat = torch.cat([layer_mf, layer_mlp], dim = -1)
        output = self.predict_layer(layer_concat)
        
        return output

###################################### test ######################################

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

rating_tensor = torch.FloatTensor( [value for value in data.values()])


config = {
    'num_users' : 100, 
    'num_items' : 100, 
    'latent_dim_mf' : 32, 
    'latent_dim_mlp' : 32
}

ncf = NCF(config)

ncf


epochs = 100
criterion = nn.MSELoss()
optimizer = optim.SGD(ncf.parameters(), lr = 0.01, momentum = 0.9)

for epoch in range(epochs):
    total_loss = 0
    for idx, (user, item, rating) in enumerate(zip(user_tensor, item_tensor, rating_tensor)):
        
        prediction = ncf(user, item)
        loss = criterion(prediction, rating)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
    total_loss /= data.__len__()
    print(f'epoch : [{epoch+1}/{epochs}], Loss : {total_loss:.4f}')
    
    
