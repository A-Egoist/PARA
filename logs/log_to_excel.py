# -*- coding: utf-8 -*- 
# @File : log_to_excel.py
# @Time : 2024/02/01 14:32:30
# @Author : Amonologue 
# @Software : Visual Studio Code
import argparse
import os
import re
import numpy as np
import pandas as pd


def log_to_excel(log_path, start_line, end_line, excel_path):
    # Load data
    with open(log_path, 'r', encoding='utf-8') as file:
        logs = file.readlines()
    logs = logs[start_line - 1: end_line]
    results = []
    results_mean = []
    results_std = []
    for line in logs:
        if "evaluate_output" in line:
            match = re.search(r' - (\w+)-(.+?)\'s performance \((.+?@\d+)\): Recall=([\d\.]+) ± ([\d\.]+), Precision=([\d\.]+) ± ([\d\.]+), F1_score=([\d\.]+) ± ([\d\.]+), HR=([\d\.]+) ± ([\d\.]+), MRR=([\d\.]+) ± ([\d\.]+), MAP=([\d\.]+) ± ([\d\.]+), NDCG=([\d\.]+) ± ([\d\.]+), Novelty=([\d\.]+) ± ([\d\.]+), Gini=([\d\.]+) ± ([\d\.]+), ARP_t=([\d\.]+) ± ([\d\.]+), ARP_r=([\d\.]+) ± ([\d\.]+), ARQ_t=([\d\.]+) ± ([\d\.]+), ARQ_r=([\d\.]+) ± ([\d\.]+), NG_score=([\d\.]+) ± ([\d\.]+), Overall_score=([\d\.]+) ± ([\d\.]+).', line)
            if match:
                results.append({
                    'dataset': match.group(3),
                    'backbone': match.group(1),
                    'method': match.group(2),
                    'Recall': f"{float(match.group(4)):.4f} ± {float(match.group(5)):.4f}",
                    'Precision': f"{float(match.group(6)):.4f} ± {float(match.group(7)):.4f}",
                    'F1_score': f"{float(match.group(8)):.4f} ± {float(match.group(9)):.4f}",
                    'HR': f"{float(match.group(10)):.4f} ± {float(match.group(11)):.4f}",
                    'MRR': f"{float(match.group(12)):.4f} ± {float(match.group(13)):.4f}",
                    'MAP': f"{float(match.group(14)):.4f} ± {float(match.group(15)):.4f}",
                    'NDCG': f"{float(match.group(16)):.4f} ± {float(match.group(17)):.4f}",
                    'Novelty': f"{float(match.group(18)):.4f} ± {float(match.group(19)):.4f}",
                    'Gini': f"{float(match.group(20)):.4f} ± {float(match.group(21)):.4f}",
                    'ARP_t': f"{float(match.group(22)):.4f} ± {float(match.group(23)):.4f}",
                    'ARP_r': f"{float(match.group(24)):.4f} ± {float(match.group(25)):.4f}",
                    'ARQ_t': f"{float(match.group(26)):.4f} ± {float(match.group(27)):.4f}",
                    'ARQ_r': f"{float(match.group(28)):.4f} ± {float(match.group(29)):.4f}",
                    'NG_score': f"{float(match.group(30)):.4f} ± {float(match.group(31)):.4f}",
                    'Overall_score': f"{float(match.group(32)):.4f} ± {float(match.group(33)):.4f}"
                })
                results_mean.append({
                    'dataset': match.group(3),
                    'backbone': match.group(1),
                    'method': match.group(2),
                    'Recall': f"{float(match.group(4)):.4f}",
                    'Precision': f"{float(match.group(6)):.4f}",
                    'F1_score': f"{float(match.group(8)):.4f}",
                    'HR': f"{float(match.group(10)):.4f}",
                    'MRR': f"{float(match.group(12)):.4f}",
                    'MAP': f"{float(match.group(14)):.4f}",
                    'NDCG': f"{float(match.group(16)):.4f}",
                    'Novelty': f"{float(match.group(18)):.4f}",
                    'Gini': f"{float(match.group(20)):.4f}",
                    'ARP_t': f"{float(match.group(22)):.4f}",
                    'ARP_r': f"{float(match.group(24)):.4f}",
                    'ARQ_t': f"{float(match.group(26)):.4f}",
                    'ARQ_r': f"{float(match.group(28)):.4f}",
                    'NG_score': f"{float(match.group(30)):.4f}",
                    'Overall_score': f"{float(match.group(32)):.4f}",
                })
                results_std.append({
                    'dataset': match.group(3),
                    'backbone': match.group(1),
                    'method': match.group(2),
                    'Recall': f"{float(match.group(5)):.4f}",
                    'Precision': f"{float(match.group(7)):.4f}",
                    'F1_score': f"{float(match.group(9)):.4f}",
                    'HR': f"{float(match.group(11)):.4f}",
                    'MRR': f"{float(match.group(13)):.4f}",
                    'MAP': f"{float(match.group(15)):.4f}",
                    'NDCG': f"{float(match.group(17)):.4f}",
                    'Novelty': f"{float(match.group(19)):.4f}",
                    'Gini': f"{float(match.group(21)):.4f}",
                    'ARP_t': f"{float(match.group(23)):.4f}",
                    'ARP_r': f"{float(match.group(25)):.4f}",
                    'ARQ_t': f"{float(match.group(27)):.4f}",
                    'ARQ_r': f"{float(match.group(29)):.4f}",
                    'NG_score': f"{float(match.group(31)):.4f}",
                    'Overall_score': f"{float(match.group(33)):.4f}"
                })
    df = pd.DataFrame(results)
    df_mean = pd.DataFrame(results_mean)
    df_std = pd.DataFrame(results_std)
    df_rank = pd.DataFrame(columns=df.columns)
    metrics = ['Recall', 'Precision', 'F1_score', 'HR', 'MRR', 'MAP', 'NDCG', 'Novelty', 'Gini', 'ARP_t', 'ARP_r', 'ARQ_t', 'ARQ_r', 'NG_score', 'Overall_score']
    for metric in metrics:
        df_mean[metric] = df_mean[metric].astype(float)
        df_std[metric] = df_std[metric].astype(float)
    
    df_temp = pd.DataFrame(columns=df.columns)
    df_temp[['dataset', 'backbone', 'method']] = df[['dataset', 'backbone', 'method']]
    for metric in metrics:
        if metric == 'Gini' or metric == 'ARP_t' or metric == 'ARP_r':
            df_temp[metric] = (1 - np.round(df_mean[metric], 4)) * 10000 + (1 - np.round(df_std[metric], 4))
        else:
            df_temp[metric] = np.round(df_mean[metric], 4) * 10000 + (1 - np.round(df_std[metric], 4))
    
    df_rank[['dataset', 'backbone', 'method']] = df[['dataset', 'backbone', 'method']]
    for metric in metrics:
        df_rank[metric] = df_temp.groupby('dataset')[metric].rank(ascending=False, method='min')

    for metric in metrics:
        df[metric] = df.apply(lambda row: f"{row[metric]} ({int(df_rank[(df_rank['dataset'] == row['dataset']) & (df_rank['backbone'] == row['backbone']) & (df_rank['method'] == row['method'])][metric].values[0])})", axis=1)
    
    df.to_excel(excel_path, index=False)
    df_mean.to_excel('.' + excel_path.split('.')[1] + '-mean' + '.' + 'xlsx', index=False)
    df_std.to_excel('.' + excel_path.split('.')[1] + '-std' + '.' + 'xlsx', index=False)
    df_rank.to_excel('.' + excel_path.split('.')[1] + '-rank' + '.' + 'xlsx', index=False)
    print("Convert successfully.")

    
if __name__ == '__main__':
    # TODO
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument('--input', type=str, default='./log/eval.log', help="Path to the source log file")
    parser.add_argument('--output', type=str, default='./output/eval.xlsx', help="Path to the target Excel file")
    parser.add_argument('--sl', type=int, default=1, help="Start line number in the log file (inclusive)")
    parser.add_argument('--el', type=int, default=1, help="End line number in the log file (inclusive)")
    args = parser.parse_args()

    # 检查行号范围是否有效
    if args.sl > args.el:
        raise ValueError(f"Start line ({args.sl}) must lower than end line ({args.el}).")

    log_to_excel(args.input, args.sl, args.el, args.output)
