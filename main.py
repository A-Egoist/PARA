# -*- coding: utf-8 -*- 
# @File : main.py
# @Time : 2024/09/16 12:18:12
# @Author : Amonologue 
# @Software : Visual Studio Code
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from loguru import logger
import argparse
from collections import defaultdict
from tqdm import tqdm

from src.utils import *
from src.metrics import *
from src import data_processing
from src.models import *


def run(device, backbone, method, dataset, mode):
    # e.g. ('MF', 'cuda:0', 'amazon-music', 'TWPA')
    # mode: {'train', 'eval', 'both'}
    # print(f'backbone: {backbone}, device: {device}, dataset: {dataset}, method: {method}, mode: {mode}')
    log_file = logger.add(f'./logs/{mode}.log', encoding='utf-8')
    logger.info(f'backbone: {backbone}, device: {device}, dataset: {dataset}, method: {method}')
    if mode == 'eval' or mode == 'both':
        top_k = 10
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
        logger.info(f'Fold [{cv_index}/5]')
        # config
        num_epoch, train_batch_size, test_batch_size, lr, num_workers, embedding_dim, n_layers, lamb, method_config = get_config(backbone, dataset, method)

        # data processing 
        train_data_path, extend_data_path, test_data_path = get_data_path(dataset, cv_index)
        num_users, num_items, train_data, test_data, item_information = data_processing.load_data(dataset, train_data_path, test_data_path)
        if backbone == 'LightGCN':
            graph = data_processing.get_graph(dataset, cv_index, num_users, num_items, train_data).to(device)
        train_dataset = data_processing.BPRDataset(train_data[['user', 'item']], num_items, 4, True)
        train_loader = DataLoader(train_dataset, train_batch_size, shuffle=True)

        if method == 'PARA':
            item_information = trisecting(method, method_config['alpha'], method_config['beta'], item_information)
            item_information = acting(item_information)
            item_adjustment_coeeficient = torch.tensor(item_information['coefficient'].values, dtype=torch.float32, device=device)
        else:
            # item_information['popularity'].fillna(0, inplace=True)
            item_popularity = torch.tensor(item_information['popularity'].values, dtype=torch.float32, device=device)

        if backbone == 'MF':
            if method == 'base':
                model = Base(backbone, device, num_users, num_items, embedding_dim, n_layers)
            elif method == 'IPS':
                model = IPS(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config)
            elif method == 'DICE':
                model = DICE(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config)
            elif method == 'PD':
                model = PD(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config)
            elif method == 'PDA':
                model = PDA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config)
            elif method == 'TIDE':
                model = TIDE(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config)
            elif method == 'PARA':
                model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config)
        elif backbone == 'LightGCN':
            if method == 'base':
                model = Base(backbone, device, num_users, num_items, embedding_dim, n_layers, graph)
            elif method == 'IPS':
                model = IPS(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph)
            elif method == 'DICE':
                model = DICE(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph)
            elif method == 'PD':
                model = PD(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph)
            elif method == 'PDA':
                model = PDA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph)
            elif method == 'TIDE':
                model = TIDE(backbone, device, num_users, num_items, embedding_dim, n_layers, item_popularity, method_config, graph)
            elif method == 'PARA':
                model = PARA(backbone, device, num_users, num_items, embedding_dim, n_layers, item_adjustment_coeeficient, method_config, graph)
        # if pretrained:
        #     model.load_state_dict(torch.load(f'./save_model/LightGCN-{method}-{dataset}-{cv_index}.pt'))
        if mode == 'eval':
                model.load_state_dict(torch.load(f'./save_model/{backbone}-{method}-{dataset}-{cv_index}.pt'))
        model.to(device)

        if mode == 'train' or mode == 'both':
            # optimizer
            # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            optimizer = get_optimizer(backbone, method, model, lr, method_config)

            # train
            model.train()
            logger.info('Training started')
            # print('loading extend data.')
            train_loader.dataset.negative_sample(dataset, extend_data_path)  # generate negative samples
            loss_min = 0x3f3f3f3f
            for epoch in range(num_epoch):
                model.train()
                loss_sum = torch.tensor([0], dtype=torch.float32).to(device)
                for user_ids, item_i_ids, item_j_ids in tqdm(train_loader):
                    user_ids = user_ids.to(device)
                    item_i_ids = item_i_ids.to(device)
                    item_j_ids = item_j_ids.to(device)
                    if method == 'base':
                        prediction_i, prediction_j, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                        loss = -1 * (prediction_i - prediction_j).sigmoid().log().sum() + lamb * reg_loss
                    elif method == 'IPS':
                        prediction_i, prediction_j, ips_c, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                        loss = -1 * (ips_c * (prediction_i - prediction_j).sigmoid().log()).sum() + lamb * reg_loss
                    elif method == 'DICE':
                        loss_click, loss_interest, loss_popularity_1, loss_popularity_2, loss_discrepancy, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                        loss = loss_click + method_config['DICE_alpha'] * (loss_interest + loss_popularity_1 + loss_popularity_2) + method_config['DICE_beta'] * loss_discrepancy + lamb * reg_loss
                    elif method == 'PD' or method == 'PDA' or method == 'TIDE':
                        prediction_i, prediction_j, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                        loss = torch.mean(torch.nn.functional.softplus(prediction_j - prediction_i)) + lamb * reg_loss
                    elif method == 'PARA':
                        prediction_i, prediction_j, reg_loss = model(user_ids, item_i_ids, item_j_ids)
                        loss = -1 * (prediction_i - prediction_j).sigmoid().log().sum() + lamb * reg_loss
                    # print(f'lamb * reg_loss = {lamb * reg_loss.item()}')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.item()
                
                logger.info(f'{backbone}-{method} on {dataset}: Epoch [{epoch + 1}/{num_epoch}], Loss={np.round(loss_sum.item(), 4)}')
                if loss_sum.item() < loss_min:
                    loss_min = loss_sum
                    path = f'./save_model/{backbone}-{method}-{dataset}-{cv_index}.pt'
                    torch.save(model.state_dict(), path)
            logger.info('Training completed')

        if mode == 'eval' or mode == 'both':
            # evaluate
            model.eval()
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
            test_users = test_data['user'].unique()  # the set composed of all users in test data, ndarray
            num_user_batchs = len(test_users) // test_batch_size + 1
            train_user_items = train_data.groupby('user')['item'].apply(list).to_dict()  # the set composed of $\mathbb{I}_{u}$ where $u\in\mathbb{U}$
            items_remove = []  # 2d
            for user in test_users:
                if user in train_user_items.keys():
                    items_remove.append(train_user_items[user])
                else:
                    items_remove.append([])  # ensure len(items_removie) == len(test_users)
            with torch.no_grad():
                for batch_id in tqdm(range(num_user_batchs)):
                    user_batch = test_users[batch_id * test_batch_size: (batch_id + 1) * test_batch_size]  # get a batch of users
                    items_remove_batch = items_remove[batch_id * test_batch_size: (batch_id + 1) * test_batch_size]
                    user_ids = torch.from_numpy(user_batch).long().to(device)

                    # get prediction value
                    prediction_batch = model.predict(user_ids).detach().cpu()
                    
                    # remove items that user have interacted in train data
                    for index in range(len(items_remove_batch)):
                        prediction_batch[index][np.array(items_remove_batch[index])] = -1

                    # get top-k recommendation lists
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

            # Output the results of each fold
            # result_fold = {'Recall': np.mean(Recall), 'Precision': np.mean(Precision), 'F1_score': np.mean(F1_score), 'HR': np.mean(HR), 'MRR': np.mean(RR), 'MAP': np.mean(AP), 'NDCG': np.mean(NDCG), 
            #                'Novelty': np.mean(Novelty), 'Gini': Gini, 'ARP_t': np.mean(ARP_t), 'ARP_r': np.mean(ARP_r), 'ARQ_t': np.mean(ARQ_t), 'ARQ_r': np.mean(ARQ_r), 
            #                'NG_score': NG_score, 'Overall_score': Overall_score}
            # evaluate_output(backbone, dataset, method, top_k, result_fold)
            
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

    if mode == 'eval' or mode == 'both':
        # output
        logger.info('5-fold cross validation result:')
        result = {'Recall': cv_Recall, 'Precision': cv_Precision, 'F1_score': cv_F1_score, 'HR': cv_HR, 'MRR': cv_RR, 'MAP': cv_AP, 'NDCG': cv_NDCG, 'Novelty': cv_Novelty, 'Gini': cv_Gini, 
                'ARP_t': cv_ARP_t, 'ARP_r': cv_ARP_r, 'ARQ_t': cv_ARQ_t, 'ARQ_r': cv_ARQ_r, 'NG_score': cv_NG_score, 'Overall_score': cv_Overall_score}
        evaluate_output(backbone, dataset, method, top_k, result)
    
    logger.remove(log_file)
    with open(f'./logs/{mode}.log', 'a', encoding='utf-8') as f:
        f.write('\n')
        f.write('\n')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    set_seed()

    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--backbone', type=str, default='MF', help="['MF', 'LightGCN', 'ItemCF']")
    parser.add_argument('--method', type=str, default='PARA', help="['base', 'IPS', 'DICE', 'PDA', 'TIDE', 'PARA']")
    parser.add_argument('--dataset', type=str, default='ciao', help="['amazon-music', 'ciao', 'douban-book', 'douban-movie', 'ml-1m', 'ml-10m']")
    parser.add_argument('--mode', type=str, default='eval', help="['train', 'eval', 'both']")
    args = parser.parse_args()

    run(device, args.backbone, args.method, args.dataset, args.mode)
