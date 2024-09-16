# -*- coding: utf-8 -*- 
# @File : ItemCF.py
# @Time : 2024/08/18 23:10:30
# @Author : Amonologue 
# @Software : Visual Studio Code
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
import os
import yaml
from .utils import get_data_path, evaluate_output, set_seed
from .metrics import *
import data_processing


def get_config(dataset, method):
    with open(os.path.join(os.getcwd(), 'config.yaml'), 'r', encoding='utf-8') as f:
        parameters = yaml.safe_load(f)
    # global config
    test_batch_size = parameters['test_batch_size']
    # dataset config
    lamb = parameters[f'lamb-{dataset}']
    
    # method config
    if method == 'ItemCF':
        method_config = None
    elif method == 'ItemCF-PARA':
        method_config = {'alpha': parameters[f'TWDP_alpha-{dataset}'],
                         'beta': parameters[f'TWDP_beta-{dataset}']}
    return test_batch_size, lamb, method_config


def run(device, top_k, dataset, method):
    log_file = logger.add(f'./Code/log/Result - eval.log', encoding='utf-8')
    logger.info(f'device: {device}, dataset: {dataset}, method: {method}')

    cv_Recall = []
    cv_Precision = []
    cv_F1_score = []
    cv_HR = []
    cv_RR = []
    cv_AP = []
    cv_NDCG = []
    cv_Novelty = []
    cv_Gini = []
    cv_ARP_t = []  # Ground Truth
    cv_ARP_r = []  # Recommend
    cv_ARQ_t = []  # Ground Truth
    cv_ARQ_r = []  # Recommend
    cv_NG_score = []
    cv_Overall_score = []
    
    for cv_index in range(1, 6):
        test_batch_size, lamb, method_config = get_config(dataset, method)
        train_data_path, extend_data_path, test_data_path = get_data_path(dataset, cv_index)
        num_users, num_items, train_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, test_data_path)

        if method == 'ItemCF-PARA':
            item_information = trisecting(method_config['alpha'], method_config['beta'], item_information)
            item_information = acting(item_information)
            item_adjustment_coefficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)

        user_indices = train_data['user'].values
        item_indices = train_data['item'].values
        rating_indices = train_data['rating'].values

        data_indices = np.array([user_indices, item_indices], dtype=np.int64)
        data_values = np.array(rating_indices, dtype=np.float32)

        user_item_sparse_matrix = torch.sparse_coo_tensor(torch.tensor(data_indices, dtype=torch.int64), torch.tensor(data_values, dtype=torch.float32), size=(num_users, num_items), device=device)

        item_similarity = torch.sparse.mm(user_item_sparse_matrix.T, user_item_sparse_matrix)

        # evaluate
        logger.info('Evaluation started')
        Recall = []
        Precision = []
        HR = []
        RR = []
        AP = []
        NDCG = []
        Novelty = []
        ARP_t = []  # Ground Truth
        ARP_r = []  # Recommend
        ARQ_t = []  # Ground Truth
        ARQ_r = []  # Recommend
        r_item_count = defaultdict(int)

        test_users = test_data['user'].unique()
        test_batch_size = 2048
        num_user_batchs = len(test_users) // test_batch_size + 1
        train_user_items = train_data.groupby('user')['item'].apply(list).to_dict()  # the set composed of $\mathbb{I}_{u}$ where $u\in\mathbb{U}$
        items_remove = []  # 2d
        for user in test_users:
            if user in train_user_items.keys():
                items_remove.append(train_user_items[user])
            else:
                items_remove.append([])  # ensure len(items_removie) == len(test_users)

        for batch_id in tqdm(range(num_user_batchs)):
            user_batch = test_users[batch_id * test_batch_size: (batch_id + 1) * test_batch_size]  # get a batch of users
            items_remove_batch = items_remove[batch_id * test_batch_size: (batch_id + 1) * test_batch_size]
            user_ids = torch.from_numpy(user_batch).long().to(device)
            if method == 'ItemCF':
                prediction_batch = predict_ItemCF(user_ids, user_item_sparse_matrix, item_similarity).cpu()
            elif method == 'ItemCF-PARA':
                prediction_batch = predict_ItemCF_PARA(user_ids, user_item_sparse_matrix, item_similarity, item_adjustment_coefficient).cpu()

            for index in range(len(items_remove_batch)):
                prediction_batch[index][np.array(items_remove_batch[index])] = -1

            _, top_k_indices_sorted = torch.topk(prediction_batch, k=top_k, dim=1)
            top_k_indices_sorted = top_k_indices_sorted.numpy()

            # get true list
            ground_truth = []
            for user in user_batch:
                ground_truth.append(test_data.loc[test_data['user'] == user, 'item'].values.reshape(-1))

            # compute performances
            for t, r in zip(ground_truth, top_k_indices_sorted):
                # t: y_true
                # r: y_pred
                Recall.append(get_Recall(t, r))
                Precision.append(get_Precision(t, r, top_k))
                HR.append(get_HR(t, r))
                RR.append(get_RR(t, r))
                AP.append(get_AP(t, r))
                NDCG.append(get_NDCG(t, r, top_k))
                Novelty.append(get_Novelty(r, item_information['popularity'].values, top_k))
                # Gini
                for item in r:
                    r_item_count[item] += 1  # count the number of times item i appears
                ARP_t.append(get_ARP(t, item_information['popularity'].values))
                ARP_r.append(get_ARP(r, item_information['popularity'].values))
                ARQ_t.append(get_ARQ(t, item_information['quality'].values))
                ARQ_r.append(get_ARQ(r, item_information['quality'].values))
                # coverage = coverage.union(r)
            
        # Compute F1-score
        recall = np.mean(Recall)
        precision = np.mean(Precision)
        F1_score = 0
        if recall + precision > 1e-6:
            F1_score = 2 * recall * precision / (recall + precision)

        # Compute Gini Coefficient
        r_item_counts = list(r_item_count.values())
        r_item_counts.sort()
        cumulative_counts = np.cumsum(r_item_counts)
        cumulative_proportion = cumulative_counts / cumulative_counts[-1]
        item_proportion = np.arange(1, len(r_item_counts) + 1) / len(r_item_counts)
        area_under_lorenz_curve = np.trapz(cumulative_proportion, item_proportion)
        Gini = 1 - 2 * area_under_lorenz_curve

        # Compute NG-score and Overall-score
        ndcg = np.mean(NDCG)
        gini = Gini
        NG_score = 0
        if ndcg + 1 - gini > 1e-6:
            NG_score = 2 * ndcg * (1 - gini) / (ndcg + 1 - gini)
        Overall_score = (ndcg + 1 - gini) / 2
        
        cv_Recall.append(np.mean(Recall))
        cv_Precision.append(np.mean(Precision))
        cv_F1_score.append(np.mean(F1_score))
        cv_HR.append(np.mean(HR))
        cv_RR.append(np.mean(RR))
        cv_AP.append(np.mean(AP))
        cv_NDCG.append(np.mean(NDCG))
        cv_Novelty.append(np.mean(Novelty))
        cv_Gini.append(Gini)
        cv_ARP_t.append(np.mean(ARP_t))
        cv_ARP_r.append(np.mean(ARP_r))
        cv_ARQ_t.append(np.mean(ARQ_t))
        cv_ARQ_r.append(np.mean(ARQ_r))
        cv_NG_score.append(NG_score)
        cv_Overall_score.append(Overall_score)
    logger.info('Evaluation completed')
    logger.info('5-fold cross validation result:')
    result = {'Recall': cv_Recall, 'Precision': cv_Precision, 'F1_score': cv_F1_score, 'HR': cv_HR, 'MRR': cv_RR, 'MAP': cv_AP, 'NDCG': cv_NDCG, 'Novelty': cv_Novelty, 'Gini': cv_Gini, 'ARP_t': cv_ARP_t, 'ARP_r': cv_ARP_r, 'ARQ_t': cv_ARQ_t, 'ARQ_r': cv_ARQ_r, 'NG_score': cv_NG_score, 'Overall_score': cv_Overall_score}
    evaluate_output('', dataset, method, top_k, result)
    
    logger.remove(log_file)
    with open(f'./log/eval.log', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write('\n')


def predict_ItemCF(user_ids, user_item_tensor, item_similarity):
    # user_ratings = user_item_tensor[user_ids]
    user_ratings = torch.index_select(user_item_tensor, 0, user_ids)

    # prediction = torch.matmul(user_ratings, item_similarity) / torch.sum(item_similarity, dim=1)
    prediction = torch.sparse.mm(user_ratings, item_similarity)  # sparse
    prediction = prediction.to_dense()
    return prediction


def trisecting(alpha, beta, item_information):
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


def predict_ItemCF_PARA(user_ids, user_item_tensor, item_similarity, item_adjustment_coefficient):
    user_ratings = torch.index_select(user_item_tensor, 0, user_ids)

    prediction = torch.sparse.mm(user_ratings, item_similarity)  # sparse
    prediction = prediction.to_dense()
    
    # get adjusted prediction
    prediction = torch.nn.functional.softplus(prediction) * torch.sigmoid(item_adjustment_coefficient)
    return prediction


if __name__ == '__main__':
    device = torch.device('cpu')
    print(f'device: {device}')
    set_seed.set_seed()
    datasets = ['amazon-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']
    for dataset in datasets:
        run(device, 10, dataset, 'ItemCF')
