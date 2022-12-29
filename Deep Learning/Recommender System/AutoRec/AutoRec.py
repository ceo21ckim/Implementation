import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F 

import sys, os, glob 
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.pardir)
os.getcwd()

torch.cuda.set_device(0)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# torch norm
class I_AutoRec(nn.Module):
    def __init__(self, item_vector, latent_dim=64, hidden_unit=500, device='cuda:0'):
        super(self, I_AutoRec).__init__()

        # initialization
        self.item_vector = item_vector # m dimension
        self.latent_dim = latent_dim  # d dimension
        self.hidden_unit = hidden_unit # k dimension

        # embed
        self.item_embedding = nn.Embedding(self.item_vector, self.latent_dim) # dimension is m by d

        # connection 
        self.V = nn.Parameter(torch.randn(self.latent_dim, self.hidden_unit, device=device))
        self.W = nn.Parameter(torch.randn(self.hidden_unit, self.latent_dim, deivce=device))

        # bias
        self.mu_bias = nn.Parameter(torch.ones(self.hidden_unit, device=device))
        self.b_bias = nn.Parameter(torch.ones(self.latent_dim, device=device))

        # activate function
        self.identity = nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, item_id):
        items = self.item_embedding(item_id)

        step1 = torch.matmul(self.V, items) + self.mu_bias 
        step2 = self.sigmoid(step1)
        step3 = torch.matmul(self.W, step2) + self.b_bias 
        step4 = self.identity(step3)

        return step4


class NetflixDataset(Dataset):
    def __init__(self, data, n_user, n_item, user_based = False):
        self.data = data
        self.n_user = n_user 
        self.n_item = n_item 
        self.x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        users = self.data.users[idx]
        items = self.data.items[idx]
        ratings = self.data.ratings[idx]
        return (
            torch.Tensor(users), 
            torch.Tensor(items), 
            torch.FloatTensor(ratings)
        )


if __name__ == '__main__':
    batch_size = 128
    epoch = 100

    netflix = pd.read_csv('netflix_titles.csv')
    netflix = netflix.loc[:,[]]
    

    # trainset = NetflixDataset(trainset)
    train_loader = DataLoader()





    