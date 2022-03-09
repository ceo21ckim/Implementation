import numpy as np 
import pandas as pd
import os 
import random 
import torch

import config 


class NCF_Data(object):
    def __init__(self, args, ratings):
        self.ratings = ratings
        self.num_ng = args.num_ng 
        self.num_ng_test = args.num_ng_test 
        self.batch_size = args.batch_size 
        
        self.preprecess_ratings = self._reindex(self.ratings)
        
        self.user_pool = set(self.ratings['user_id'].values)
        self.item_pool = set(self.ratings['item_id'].values)
        
        self.train_ratings, self.test_ratings = self._leave_one_out(self.preprecess_ratings)
        self.negatives = self._negative_sampling(self.preprocess_ratings)
        
    def _reindex(self, ratings): # label encoding
        user_list = ratings['user_id'].unique()
        user2id = {w: i for i, w in enumerate(user_list)}
        
        item_list = ratings['item_id'].unique()
        item2id = {w: i for i, w in enumerate(item_list)}
        
        ratings['user_id'] = ratings['user_id'].apply(lambda x: user2id[x])
        ratings['item_id'] = ratings['item_id'].apply(lambda x: item2id[x])
        ratings['rating'] = ratings['rating'].apply(lambda x: float(x>0))
        return ratings 
    
    def _leave_one_out(self, ratings):
        ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method = 'first', ascending = False)
        test = ratings.loc[ratings['rank_latest'] == 1]
        train = ratings.loc[ratings['rank_latest'] > 1]
        
        assert train['user_id'].nunique() == test['user_id'].nunique(), 'Not Match Train User with Test User'
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]
    
    def _negative_sampling(self, ratings):
        interact_status = (
            ratings.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns = {'item_id' : 'interacted_items'})
        )
        
    def get_train_instance(self):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives.loc[:,['user_id', 'negative_items']], on = 'user_id')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.num_ng))
        for row in train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in range(self.num_ng):
                users.append(int(row.user_id))
                items.apend(int(row.negative[i]))
                ratings.append(float(0)) # negative samples get 0 rating
            
        dataset = Rating_Dataset(
            user_list = users, 
            item_list = items, 
            rating_list = ratings
        )
        return torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workes = 4)
    
    def get_test_instance(self):
        users, items, ratings = [], [], []
        test_ratings = pd.merge(self.test_ratings, self.negatives.loc[:,['user_id', 'negative_samples']], on = 'user_id')
        for row in test_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
            for i in getattr(row, 'negative_samples'):
                users.append(int(row.user_id))
                items.append(int(i))
                ratings.append(float(0))
        
        dataset = Rating_Dataset(
            user_list = users, 
            item_list = items, 
            rating_list = ratings
        )
        return torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle = True, num_workers = 4)
        
class Rating_Dataset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(Rating_Dataset, self).__init__()
        
        self.user_list = user_list 
        self.item_list = item_list 
        self.rating_list = rating_list 
        
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, index):
        user = self.user_list[index]
        item = self.item_list[index]
        rating = self.rating_list[index]
        
        return (
            torch.LongTensor(user),
            torch.LongTensor(item),
            torch.FloatTensor(rating)
        )


