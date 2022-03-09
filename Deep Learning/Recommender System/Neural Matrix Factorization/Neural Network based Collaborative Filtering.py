import torch 
import torch.nn as nn 

class NeuMF(nn.Module):
    def __init__(self, config):
        self.num_users = config.num_users
        self.num_items = config.num_items 
        self.latent_dim = config.latent_dim

        # self.