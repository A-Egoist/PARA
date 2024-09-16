# -*- coding: utf-8 -*- 
# @File : utils.py
# @Time : 2023/10/19 14:59:29
# @Author : Amonologue 
# @Software : Visual Studio Code
import numpy as np
from loguru import logger
import torch
import random
import os
import yaml


def set_seed(seed=2000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_data_path(dataset, cv_index):
    datasets = {'amazon-music': [['./data/Amazon/music/amazon_music1.train', './data/Amazon/music/amazon_music1.extend', './data/Amazon/music/amazon_music1.test'],
                                 ['./data/Amazon/music/amazon_music2.train', './data/Amazon/music/amazon_music2.extend', './data/Amazon/music/amazon_music2.test'],
                                 ['./data/Amazon/music/amazon_music3.train', './data/Amazon/music/amazon_music3.extend', './data/Amazon/music/amazon_music3.test'],
                                 ['./data/Amazon/music/amazon_music4.train', './data/Amazon/music/amazon_music4.extend', './data/Amazon/music/amazon_music4.test'],
                                 ['./data/Amazon/music/amazon_music5.train', './data/Amazon/music/amazon_music5.extend', './data/Amazon/music/amazon_music5.test']],
                'ciao': [['./data/Ciao/movie-ratings1.train', './data/Ciao/movie-ratings1.extend', './data/Ciao/movie-ratings1.test'],
                         ['./data/Ciao/movie-ratings2.train', './data/Ciao/movie-ratings2.extend', './data/Ciao/movie-ratings2.test'],
                         ['./data/Ciao/movie-ratings3.train', './data/Ciao/movie-ratings3.extend', './data/Ciao/movie-ratings3.test'],
                         ['./data/Ciao/movie-ratings4.train', './data/Ciao/movie-ratings4.extend', './data/Ciao/movie-ratings4.test'],
                         ['./data/Ciao/movie-ratings5.train', './data/Ciao/movie-ratings5.extend', './data/Ciao/movie-ratings5.test']],
                'douban-book': [['./data/Douban/book/douban_book1.train', './data/Douban/book/douban_book1.extend', './data/Douban/book/douban_book1.test'],
                                ['./data/Douban/book/douban_book2.train', './data/Douban/book/douban_book2.extend', './data/Douban/book/douban_book2.test'],
                                ['./data/Douban/book/douban_book3.train', './data/Douban/book/douban_book3.extend', './data/Douban/book/douban_book3.test'],
                                ['./data/Douban/book/douban_book4.train', './data/Douban/book/douban_book4.extend', './data/Douban/book/douban_book4.test'],
                                ['./data/Douban/book/douban_book5.train', './data/Douban/book/douban_book5.extend', './data/Douban/book/douban_book5.test']],
                'douban-movie': [['./data/Douban/movie/douban_movie1.train', './data/Douban/movie/douban_movie1.extend', './data/Douban/movie/douban_movie1.test'],
                                 ['./data/Douban/movie/douban_movie2.train', './data/Douban/movie/douban_movie2.extend', './data/Douban/movie/douban_movie2.test'],
                                 ['./data/Douban/movie/douban_movie3.train', './data/Douban/movie/douban_movie3.extend', './data/Douban/movie/douban_movie3.test'],
                                 ['./data/Douban/movie/douban_movie4.train', './data/Douban/movie/douban_movie4.extend', './data/Douban/movie/douban_movie4.test'],
                                 ['./data/Douban/movie/douban_movie5.train', './data/Douban/movie/douban_movie5.extend', './data/Douban/movie/douban_movie5.test']],
                'ml-1m': [['./data/ml-1m/ratings1.train', './data/ml-1m/ratings1.extend', './data/ml-1m/ratings1.test'],
                           ['./data/ml-1m/ratings2.train', './data/ml-1m/ratings2.extend', './data/ml-1m/ratings2.test'],
                           ['./data/ml-1m/ratings3.train', './data/ml-1m/ratings3.extend', './data/ml-1m/ratings3.test'],
                           ['./data/ml-1m/ratings4.train', './data/ml-1m/ratings4.extend', './data/ml-1m/ratings4.test'],
                           ['./data/ml-1m/ratings5.train', './data/ml-1m/ratings5.extend', './data/ml-1m/ratings5.test']],
                'ml-10m': [['./data/ml-10m/ratings1.train', './data/ml-10m/ratings1.extend', './data/ml-10m/ratings1.test'],
                           ['./data/ml-10m/ratings2.train', './data/ml-10m/ratings2.extend', './data/ml-10m/ratings2.test'],
                           ['./data/ml-10m/ratings3.train', './data/ml-10m/ratings3.extend', './data/ml-10m/ratings3.test'],
                           ['./data/ml-10m/ratings4.train', './data/ml-10m/ratings4.extend', './data/ml-10m/ratings4.test'],
                           ['./data/ml-10m/ratings5.train', './data/ml-10m/ratings5.extend', './data/ml-10m/ratings5.test']],
                'ml-100k': [['./data/ml-100k/u1.train', './data/ml-100k/u1.extend', './data/ml-100k/u1.test'],
                            ['./data/ml-100k/u2.train', './data/ml-100k/u2.extend', './data/ml-100k/u2.test'],
                            ['./data/ml-100k/u3.train', './data/ml-100k/u3.extend', './data/ml-100k/u3.test'],
                            ['./data/ml-100k/u4.train', './data/ml-100k/u4.extend', './data/ml-100k/u4.test'],
                            ['./data/ml-100k/u5.train', './data/ml-100k/u5.extend', './data/ml-100k/u5.test']],
                'netflix': [['./data/netflix/netflix1.train', './data/netflix/netflix1.extend', './data/netflix/netflix1.test'],
                            ['./data/netflix/netflix2.train', './data/netflix/netflix2.extend', './data/netflix/netflix2.test'],
                            ['./data/netflix/netflix3.train', './data/netflix/netflix3.extend', './data/netflix/netflix3.test'],
                            ['./data/netflix/netflix4.train', './data/netflix/netflix4.extend', './data/netflix/netflix4.test'],
                            ['./data/netflix/netflix5.train', './data/netflix/netflix5.extend', './data/netflix/netflix5.test']],
                'yelp': [['./data/Yelp/yelp1.train', './data/Yelp/yelp1.extend', './data/Yelp/yelp1.test'],
                         ['./data/Yelp/yelp2.train', './data/Yelp/yelp2.extend', './data/Yelp/yelp2.test'],
                         ['./data/Yelp/yelp3.train', './data/Yelp/yelp3.extend', './data/Yelp/yelp3.test'],
                         ['./data/Yelp/yelp4.train', './data/Yelp/yelp4.extend', './data/Yelp/yelp4.test'],
                         ['./data/Yelp/yelp5.train', './data/Yelp/yelp5.extend', './data/Yelp/yelp5.test']],}
    # train, extend, test
    return datasets[dataset][cv_index - 1][0], datasets[dataset][cv_index - 1][1], datasets[dataset][cv_index - 1][2]


def evaluate_output(backbone, dataset, method, top_k, result):
    # print(f'backbone: {backbone}, device: {device}, dataset: {dataset}, method: {method}')
    result = {metric: np.array(values) for metric, values in result.items()}
    result_mean = {metric: np.round(np.mean(values), 4) for metric, values in result.items()}
    result_std = {metric: np.round(np.std(values), 4) for metric, values in result.items()}
    logger.info(f"{backbone}-{method}'s performance ({dataset}@{top_k}): "
          f"Recall={result_mean['Recall']} ± {result_std['Recall']}, "
          f"Precision={result_mean['Precision']} ± {result_std['Precision']}, "
          f"F1_score={result_mean['F1_score']} ± {result_std['F1_score']}, "
          f"HR={result_mean['HR']} ± {result_std['HR']}, "
          f"MRR={result_mean['MRR']} ± {result_std['MRR']}, "
          f"MAP={result_mean['MAP']} ± {result_std['MAP']}, "
          f"NDCG={result_mean['NDCG']} ± {result_std['NDCG']}, "
          f"Novelty={result_mean['Novelty']} ± {result_std['Novelty']}, "
          f"Gini={result_mean['Gini']} ± {result_std['Gini']}, "
          f"ARP_t={result_mean['ARP_t']} ± {result_std['ARP_t']}, "
          f"ARP_r={result_mean['ARP_r']} ± {result_std['ARP_r']}, "
          f"ARQ_t={result_mean['ARQ_t']} ± {result_std['ARQ_t']}, "
          f"ARQ_r={result_mean['ARQ_r']} ± {result_std['ARQ_r']}, "
          f"NG_score={result_mean['NG_score']} ± {result_std['NG_score']}, "
          f"Overall_score={result_mean['Overall_score']} ± {result_std['Overall_score']}.")
    

def get_optimizer(backbone, method, model, lr, method_config):
    if backbone == 'MF':
        if method == 'TIDE':
            params = [
                {'params': model.user_embedding.weight, 'lr': lr}, 
                {'params': model.item_embedding.weight, 'lr': lr}, 
                {'params': model.q, 'lr': method_config['lr_q']}, 
                {'params': model.b, 'lr': method_config['lr_b']}, 
            ]
        else: 
            params = [
                {'params': model.user_embedding.weight, 'lr': lr}, 
                {'params': model.item_embedding.weight, 'lr': lr}
            ]
    elif backbone == 'LightGCN':
        if method == 'TIDE':
            params = [
                {'params': model.user_embedding_0.weight, 'lr': lr}, 
                {'params': model.item_embedding_0.weight, 'lr': lr}, 
                {'params': model.q, 'lr': method_config['lr_q']}, 
                {'params': model.b, 'lr': method_config['lr_b']}, 
            ]
        else: 
            params = [
                {'params': model.user_embedding_0.weight, 'lr': lr}, 
                {'params': model.item_embedding_0.weight, 'lr': lr}
            ]
    # optimizer = torch.optim.Adam(params)
    optimizer = torch.optim.SGD(params)
    return optimizer


def trisecting(method, alpha, beta, item_information):
    # item_information['popularity'].fillna(alpha, inplace=True)  # fill the NA with alpha
    # item_information['quality'].fillna(beta, inplace=True)  # fill the NA with beta
    def classification(row):
        if row['popularity'] < alpha and row['quality'] > beta:
            return 1  # positive
        elif row['popularity'] > alpha and row['quality'] < beta:
            return 3  # negative
        else:
            return 2  # neutral
    item_information['group'] = item_information.apply(classification, axis=1)
    return item_information


def acting(item_information):
    def function(row):
        if row['group'] == 1:
            # promote
            return (row['popularity'] ** 0.25) * row['quality']
        elif row['group'] == 2:
            # maintain
            return (row['popularity']) * row['quality']
        elif row['group'] == 3:
            # suppress
            return (row['popularity'] ** 4) * row['quality']
    item_information['coefficient'] = item_information.apply(function, axis=1)
    return item_information


def get_config(backbone, dataset, method):
    # args, parameters = myparser.parser()
    with open(os.path.join(os.getcwd(), 'config.yaml'), 'r', encoding='utf-8') as f:
        parameters = yaml.safe_load(f)
    # global config
    train_batch_size = parameters['batch_size']
    test_batch_size = parameters['test_batch_size']
    embedding_dim = parameters['embedding_dim']
    num_workers = parameters['num_workers']
    # dataset config
    n_layers = parameters[f'n_layers-{dataset}']
    lamb = parameters[f'lamb-{dataset}']
    
    # method config
    if method == 'base':
        num_epoch = parameters[f'{backbone}-base-num_epoch']
        lr = parameters[f'{backbone}-{method}-{dataset}-lr']
        method_config = None
    elif method == 'IPS':
        num_epoch = parameters[f'{backbone}-IPS-num_epoch']
        lr = parameters[f'{backbone}-{method}-lr']
        method_config = {'ips_lambda': parameters['IPS_lambda']}
    elif method == 'DICE':
        num_epoch = parameters[f'{backbone}-DICE-num_epoch']
        lr = parameters['DICE_lr']
        method_config = {'DICE_alpha': parameters['DICE_alpha'],
                         'DICE_beta': parameters['DICE_beta']}
    elif method == 'PD' or method == 'PDA':
        num_epoch = parameters[f'{backbone}-PDA-num_epoch']
        lr = parameters['PD_lr']
        method_config = {'gamma': parameters[f'PD_gamma-{dataset}']}
    elif method == 'TIDE': 
        num_epoch = parameters[f'{backbone}-{method}-num_epoch']
        lr = parameters[f'{backbone}-TIDE-{dataset}-lr']
        method_config = {'q': parameters[f'TIDE_q-{dataset}'],
                         'b': parameters[f'TIDE_b-{dataset}'],
                         'lr_q': parameters[f'TIDE_lr_q-{dataset}'],
                         'lr_b': parameters[f'TIDE_lr_b-{dataset}']}
    elif method == 'PARA':
        num_epoch = parameters[f'{backbone}-PARA-num_epoch']
        lr = parameters[f'{backbone}-PARA-{dataset}-lr']
        method_config = {'alpha': parameters[f'PARA_alpha-{dataset}'],
                         'beta': parameters[f'PARA_beta-{dataset}']}
    return num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config


