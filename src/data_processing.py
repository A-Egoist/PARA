# -*- coding: utf-8 -*- 
# @File : data_processing.py
# @Time : 2023/10/10 16:34:07
# @Author : Amonologue 
# @Software : Visual Studio Code
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from sklearn.model_selection import train_test_split
import pickle
import argparse


def load_data(dataset, train_data_path, test_data_path):
    print(f'Loading dataset {dataset}.')
    if dataset == 'amazon-music':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, skiprows=1, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, skiprows=1, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'ciao':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'douban-book':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['user', 'item', 'rating'], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['user', 'item', 'rating'], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'douban-movie':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'ml-1m':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'ml-10m':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'ml-100k':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['user', 'item', 'rating'], usecols=[0, 1, 2], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'netflix':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['user', 'item', 'rating'], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['user', 'item', 'rating'], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    elif dataset == 'yelp':
        train_data = pd.read_csv(train_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['user', 'item', 'rating'], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
        test_data = pd.read_csv(test_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['user', 'item', 'rating'], dtype={'user': np.int32, 'item': np.int32, 'rating': np.float32})
    
    save_path = '.' + train_data_path.split('.')[1] + '.pkl'
    if os.path.exists(save_path):
        # load processed data
        with open(save_path, 'rb') as f:
            num_users, num_items, train_data, test_data, item_information = pickle.load(f)
        return num_users, num_items, train_data, test_data, item_information
    else:
        num_users = max(train_data['user'].max(), test_data['user'].max()) + 1  # user id from 0 to max
        num_items = max(train_data['item'].max(), test_data['item'].max()) + 1  # item id from 0 to max

        item_information = pd.DataFrame(range(num_items), columns=['item'])  # columns = ['item', 'count', 'popularity', 'quality']
        
        # count
        def item_count(row):
            return len(train_data[train_data['item'] == row['item']])
        item_information['count'] = item_information.swifter.apply(item_count, axis=1)  # C_{i}
        # item_information['count'].fillna(0, inplace=True)  # Set the number of items that have not appeared to 0
        item_count_max = item_information['count'].max()  # C_{Max}
        # item_count_min = item_information['count'].min()  # C_{Min} TODO
        print('Count finished.')

        # calc item popularity
        # TODO
        item_information['popularity'] = item_information['count'].swifter.apply(lambda x: np.round(x / item_count_max, 4))  # $m_{i}=\frac{C_{i}}{C_{Max}}$
        print('Calculate item popularity finished.')

        # calc item quality
        # Definition 6
        item_quality_6 = calculate_item_quality_method_6(train_data.copy())
        item_information = item_information.merge(item_quality_6, on='item', how='left')
        print('Calculate item quality finished.')
        
        item_information['popularity'].fillna(0, inplace=True)  # fill the NA with alpha
        item_information['quality'].fillna(0, inplace=True)  # fill the NA with beta
        print('Data processing finished.')
        # train_data.dropna(axis=0, how='any', inplace=True)
        # test_data.dropna(axis=0, how='any', inplace=True)
        # return train_data(users, items), test_data(users, items), num_users(train_data + test_data), num_items(train_data + test_data), item_information(item, count, popularity, sum_rating, average_rating)
        
        # save processed data
        with open(save_path, 'wb') as f:
            pickle.dump((num_users, num_items, train_data, test_data, item_information), f)
        return num_users, num_items, train_data, test_data, item_information


def calculate_item_quality_method_6(train_data):
    # drop row with no rating
    train_data.drop(train_data[train_data['rating'] == -1.0].index, inplace=True)
    
    user_stddev = train_data.groupby('user')['rating'].std().reset_index()
    user_stddev.columns = ['user', 'rating_stddev']

    mu = user_stddev['rating_stddev'].median()
    sigma = user_stddev['rating_stddev'].std()

    def calculate_h_u(s_u, mu=mu, sigma=sigma):
        return np.exp(-((s_u - mu) ** 2) / (2 * sigma ** 2))

    # 计算用户评分习惯
    user_stddev['h_u'] = user_stddev['rating_stddev'].swifter.apply(calculate_h_u)
    train_data = train_data.merge(user_stddev, on='user', how='left')
    train_data['h_u'].fillna(0, inplace=True)

    train_data['weighted_rating'] = train_data['h_u'] * train_data['rating']
    item_quality = train_data.groupby('item').apply(lambda x: 0 if np.sum(x['h_u']) == 0 else np.sum(x['weighted_rating']) / np.sum(x['h_u'])).reset_index()
    item_quality.columns = ['item', 'quality']
    
    # Max-Min Normalization
    q_min = item_quality['quality'].min()
    q_max = item_quality['quality'].max()
    item_quality['quality_normalized'] = (item_quality['quality'] - q_min) / (q_max - q_min)

    # Sigmoid Normalization
    # item_quality['quality_normalized'] = 1 / (1 + np.exp(-item_quality['quality']))

    item_quality.drop(columns=['quality'], inplace=True)
    item_quality.rename(columns={'quality_normalized': 'quality'}, inplace=True)

    return item_quality


class BPRDataset(Dataset):
    def __init__(self, data, num_items, num_negatives=4, is_training=True):
        super().__init__()
        self.data = data  # DataFrame
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.is_training = is_training

    def negative_sample(self, dataset, extend_data_path):
        assert self.is_training, 'no need to sampling when testing'
        # TODO
        # how to run .exe file by python code?

        if dataset == 'ml-100k':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'ml-1m':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32, engine='python')
        elif dataset == 'ml-10m':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32, engine='python')
        elif dataset == 'douban-movie':
            # extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, skiprows=1, names=['userId', 'positiveItemId', 'negativeItemId'], usecols=[0, 1, 2], dtype=np.int32)
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'amazon-music':
            # extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, skiprows=1, names=['userId', 'positiveItemId', 'negativeItemId'], usecols=[0, 1, 2], dtype=np.int32)
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'ciao':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'netflix':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'douban-book':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'yelp':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        elif dataset == 'RunningExample':
            extend_data = pd.read_csv(extend_data_path, sep='\t', header=None, usecols=[0, 1, 2], names=['userId', 'positiveItemId', 'negativeItemId'], dtype=np.int32)
        self.data_fill = extend_data.values.tolist()

    def __getitem__(self, index):
        data = self.data_fill if self.is_training else self.data.values.to_list()
        user = data[index][0]
        item_i = data[index][1]
        item_j = data[index][2] if self.is_training else data[index][1]
        return user, item_i, item_j
    
    def __len__(self):
        # TODO
        # return 39
        return self.num_negatives * len(self.data) if self.is_training else len(self.data)


def get_graph(dataset, cv_index, num_users, num_items, train_data):
    # print(f'cwd: {os.getcwd()}')
    graph_index_path = './data/LightGCN_graph' + f'/{dataset}-{cv_index}-graph_index.npy'
    graph_data_path = './data/LightGCN_graph' + f'/{dataset}-{cv_index}-graph_data.npy'
    if os.path.exists(graph_data_path) and os.path.exists(graph_index_path):
        graph_index = np.load(graph_index_path)
        graph_data = np.load(graph_data_path)
        graph_index = torch.from_numpy(graph_index)
        graph_data = torch.from_numpy(graph_data)

        # graph_size = torch.Size([num_users + num_items, num_users + num_items])
        graph = torch.sparse.FloatTensor(graph_index, graph_data, torch.Size([num_users + num_items, num_users + num_items]))
        graph = graph.coalesce()
    else:
        trainUser = train_data['user'].values
        trainItem = train_data['item'].values
        user_dim = torch.LongTensor(trainUser)
        item_dim = torch.LongTensor(trainItem)

        # first subgraph
        first_sub = torch.stack([user_dim, item_dim + num_users])
        # second subgraph
        second_sub = torch.stack([item_dim + num_users, user_dim])
        index = torch.cat([first_sub, second_sub], dim=1)
        data = torch.ones(index.size(-1)).int()

        # DefaultCPUAllocator: not enough memory
        # graph = torch.sparse.IntTensor(index, data, torch.Size([num_users + num_items, num_users + num_items]))
        # dense = graph.to_dense()
        # D = torch.sum(dense, dim=1).float()
        # D[D == 0.] = 1.
        # D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
        # dense = dense / D_sqrt
        # dense = dense / D_sqrt.t()
        # index = dense.nonzero()
        # data = dense[dense >= 1e-9]
        # graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([num_users + num_items, num_users + num_items]))

        graph = torch.sparse.FloatTensor(index, data, torch.Size([num_users + num_items, num_users + num_items]))
        row_sum = torch.sparse.sum(graph, dim=1).to_dense()
        row_sum[row_sum == 0] = 1.
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        
        row, col = index
        data = data * d_inv_sqrt[row] * d_inv_sqrt[col]
        graph = torch.sparse.FloatTensor(index, data, torch.Size([num_users + num_items, num_users + num_items]))
        
        # np.save(graph_index_path, index.t().numpy())
        np.save(graph_index_path, index.numpy())
        np.save(graph_data_path, data.numpy())
        graph = graph.coalesce()
    return graph


def amazon_music_5_fold_split():
    data_path = './data/Amazon/music/ratings_Digital_Music.csv'
    save_path = './data/Amazon/music/amazon_music.csv'
    train_data_path = './data/Amazon/music/amazon_music{}.train'
    test_data_path = './data/Amazon/music/amazon_music{}.test'
    with open(data_path, 'r', encoding='utf-8') as f:
        user_dict, item_dict = {}, {}
        lines = f.readlines()
        u_index, i_index = 1, 1
        for line in lines:
            temp = line.strip().split(',')
            u_id = temp[0]
            i_id = temp[1]
            if u_id not in user_dict.keys():
                user_dict[u_id] = u_index
                u_index += 1
            if i_id not in item_dict.keys():
                item_dict[i_id] = i_index
                i_index += 1
    data = pd.read_csv(data_path, sep=',', header=None, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3])
    print('loaded')
    # data.replace({'user': user_dict, 'item': item_dict}, inplace=True)
    data['user'] = data['user'].map(user_dict)
    print('user replaced')
    data['item'] = data['item'].map(item_dict)
    print('item replaced')
    data = data.astype(np.int32)
    data = data.sort_values(by='timestamp')
    data.to_csv(save_path, sep='\t', header=False, index=False)

    # 5-core filtering
    # user_counts = data['user'].value_counts()
    # item_counts = data['item'].value_counts()
    # data = data[(data['user'].isin(user_counts[user_counts >= 5].index)) & (data['item'].isin(item_counts[item_counts >= 5].index))]

    k_fold = 5  # k fold
    k_fold_count = data.shape[0] // k_fold
    for fold in range(k_fold):
        index_begin = fold * k_fold_count
        index_end = (fold + 1) * k_fold_count
        test_data = data[index_begin:index_end]
        train_data = pd.concat([
            data[:index_begin],
            data[index_end:]
        ])
        train_data.to_csv(train_data_path.format(fold + 1), sep='\t', header=False, index=False)
        test_data.to_csv(test_data_path.format(fold + 1), sep='\t', header=False, index=False)


def ciao_5_fold_split():
    data_path = './data/Ciao/movie-ratings.txt'
    train_data_path = './data/Ciao/movie-ratings{}.train'
    test_data_path = './data/Ciao/movie-ratings{}.test'
    data = pd.read_csv(data_path, sep=',', header=None, names=['user', 'item', 'genre', 'review', 'rating', 'timestamp'], usecols=[0, 1, 4, 5], dtype={'user': np.int32, 'item': np.int32, 'rating': np.int32, 'timestamp': np.str_})
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp')
    
    # 5-core filtering
    # user_counts = data['user'].value_counts()
    # item_counts = data['item'].value_counts()
    # data = data[(data['user'].isin(user_counts[user_counts >= 5].index)) & (data['item'].isin(item_counts[item_counts >= 5].index))]

    k_fold = 5  # k fold
    k_fold_count = data.shape[0] // k_fold
    for fold in range(k_fold):
        index_begin = fold * k_fold_count
        index_end = (fold + 1) * k_fold_count
        test_data = data[index_begin:index_end]
        train_data = pd.concat([
            data[:index_begin],
            data[index_end:]
        ])
        train_data.to_csv(train_data_path.format(fold + 1), sep='\t', header=False, index=False)
        test_data.to_csv(test_data_path.format(fold + 1), sep='\t', header=False, index=False)


def douban_book_5_fold_split():
    data_path = './data/Douban/book/douban_book.tsv'
    train_save_path = './data/Douban/book/douban_book{}.train'
    test_save_path = './data/Douban/book/douban_book{}.test'
    data = pd.read_csv(data_path, sep='\t', header=None, skiprows=1, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32)
    # data.loc[data['rating']==-1, 'rating'] = 3
    print(f'data.shape[0]={data.shape[0]}')
    data.drop(data[data['rating']==-1].index, inplace=True)
    print(f'data.shape[0]={data.shape[0]}')
    data = data.sort_values(by='timestamp')

    # 10-core filtering
    # user_counts = data['user'].value_counts()
    # item_counts = data['item'].value_counts()
    # data = data[(data['user'].isin(user_counts[user_counts >= 10].index)) & (data['item'].isin(item_counts[item_counts >= 10].index))]

    k_fold = 5  # k fold
    k_fold_count = data.shape[0] // k_fold
    for fold in range(k_fold):
        index_begin = fold * k_fold_count
        index_end = (fold + 1) * k_fold_count
        test_data = data[index_begin:index_end]
        train_data = pd.concat([
            data[:index_begin],
            data[index_end:]
        ])
        train_data.to_csv(train_save_path.format(fold + 1), sep='\t', header=False, index=False)
        test_data.to_csv(test_save_path.format(fold + 1), sep='\t', header=False, index=False)
    

def douban_movie_5_fold_split():
    data_path = './data/Douban/movie/douban_movie.tsv'
    train_data_path = './data/Douban/movie/douban_movie{}.train'
    test_data_path = './data/Douban/movie/douban_movie{}.test'
    data = pd.read_csv(data_path, sep='\t', header=None, skiprows=1, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32)
    # data.loc[data['rating']==-1, 'rating'] = 3
    print(f'data.shape[0]={data.shape[0]}')
    data.drop(data[data['rating']==-1].index, inplace=True)
    print(f'data.shape[0]={data.shape[0]}')
    data = data.sort_values(by='timestamp')

    # 10-core filtering
    # user_counts = data['user'].value_counts()
    # item_counts = data['item'].value_counts()
    # data = data[(data['user'].isin(user_counts[user_counts >= 10].index)) & (data['item'].isin(item_counts[item_counts >= 10].index))]

    k_fold = 5  # k fold
    k_fold_count = data.shape[0] // k_fold
    for fold in range(k_fold):
        index_begin = fold * k_fold_count
        index_end = (fold + 1) * k_fold_count
        test_data = data[index_begin:index_end]
        train_data = pd.concat([
            data[:index_begin],
            data[index_end:]
        ])
        train_data.to_csv(train_data_path.format(fold + 1), sep='\t', header=False, index=False)
        test_data.to_csv(test_data_path.format(fold + 1), sep='\t', header=False, index=False)


def ml_1m_5_fold_split():
    data_path = './data/ml-1m/ratings.dat'
    train_data_path = './data/ml-1m/ratings{}.train'
    test_data_path = './data/ml-1m/ratings{}.test'
    data = pd.read_csv(data_path, sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32, engine='python')
    # data.loc[data['rating']==-1, 'rating'] = 3
    print(f'data.shape[0]={data.shape[0]}')
    data = data.sort_values(by='timestamp')

    k_fold = 5  # k fold
    k_fold_count = data.shape[0] // k_fold
    for fold in range(k_fold):
        index_begin = fold * k_fold_count
        index_end = (fold + 1) * k_fold_count
        test_data = data[index_begin:index_end]
        train_data = pd.concat([
            data[:index_begin],
            data[index_end:]
        ])
        train_data.to_csv(train_data_path.format(fold + 1), sep='\t', header=False, index=False)
        test_data.to_csv(test_data_path.format(fold + 1), sep='\t', header=False, index=False)


def ml_10m_5_fold_split():
    data_path = './data/ml-10m/ratings.dat'
    train_data_path = './data/ml-10m/ratings{}.train'
    test_data_path = './data/ml-10m/ratings{}.test'
    data = pd.read_csv(data_path, sep='::', header=None, names=['user', 'item', 'rating', 'timestamp'], usecols=[0, 1, 2, 3], dtype=np.int32, engine='python')
    # data.loc[data['rating']==-1, 'rating'] = 3
    print(f'data.shape[0]={data.shape[0]}')
    data = data.sort_values(by='timestamp')

    k_fold = 5  # k fold
    k_fold_count = data.shape[0] // k_fold
    for fold in range(k_fold):
        index_begin = fold * k_fold_count
        index_end = (fold + 1) * k_fold_count
        test_data = data[index_begin:index_end]
        train_data = pd.concat([
            data[:index_begin],
            data[index_end:]
        ])
        train_data.to_csv(train_data_path.format(fold + 1), sep='\t', header=False, index=False)
        test_data.to_csv(test_data_path.format(fold + 1), sep='\t', header=False, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--dataset', type=str, default='ciao', help="['amazon-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']")
    args = parser.parse_args()

    # 5 fold cross validation
    if args.dataset == 'amazon-music':
        amazon_music_5_fold_split()
    elif args.dataset == 'ciao':
        ciao_5_fold_split()
    elif args.dataset == 'douban-book':
        douban_book_5_fold_split()
    elif args.dataset == 'douban-movie':
        douban_movie_5_fold_split()
    elif args.dataset == 'ml-1m':
        ml_1m_5_fold_split()
    elif args.dataset == 'ml-10m':
        ml_10m_5_fold_split()
