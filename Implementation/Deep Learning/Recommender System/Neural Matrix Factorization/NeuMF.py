import torch 
import torch.nn as nn 


class NCF(nn.Module):
    def __init__(self, num_users, num_items, config):
        super(NCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.layers = config.layers
        self.latent_dim = config.latent_dim


        # Matrix Factorication embedding
        self.user_embedding_mf = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding_mf = nn.Embedding(self.num_items, self.latent_dim)

        # Multi Layer Perceptron embedding
        self.user_embedding_mlp = nn.Embedding(self.num_users, self.latent_dim)
        self.item_embedding_mlp = nn.Embedding(self.num_items, self.latent_dim)

        self.fc_layer_mlp = nn.ModuleList()
        for (in_size, out_size) in zip(self.layers[:-1], self.layers[1:]):
            self.fc_layer_mlp.append(nn.Linear(in_size, out_size))
            self.fc_layer_mlp.append(nn.ReLU())
            # self.fc_layer_mlp.append(nn.Dropout(0.2))


        self.predict_layer = nn.Linear(self.latent_dim + self.latent_dim//4, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weight()

    def forward(self, user, item):
        user_vec_mf = self.user_embedding_mf(user)
        item_vec_mf = self.item_embedding_mf(item)

        user_vec_mlp = self.user_embedding_mlp(user)
        item_vec_mlp = self.item_embedding_mlp(item)

        predict_layer_mf = torch.mul(user_vec_mf, item_vec_mf)

        predict_layer_mlp = torch.cat([user_vec_mlp, item_vec_mlp], dim = -1)

        for idx, _ in enumerate(range(len(self.fc_layer_mlp))):
            predict_layer_mlp = self.fc_layer_mlp[idx](predict_layer_mlp)

        mf_mlp_concat = torch.cat([predict_layer_mlp, predict_layer_mf], dim = -1)

        pred = self.predict_layer(mf_mlp_concat)

        purchase = self.sigmoid(pred)

        return purchase.squeeze()
    
    def init_weight(self):
        nn.init.normal_(self.user_embedding_mf.weight, std = 0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std = 0.01)

        nn.init.normal_(self.user_embedding_mlp.weight, std = 0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std = 0.01)

        for m in self.fc_layer_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        
        nn.init.xavier_uniform_(self.predict_layer.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

        

