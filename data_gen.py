# 生成train/val/test data及graph data
from collections import defaultdict
import pandas as pd
from gensim.corpora.dictionary import Dictionary
import torch
import random
from tqdm import tqdm
import os
import sys
from config import DataConfig

def split_dataset(data_cfg):
    # 将数据分成训练集、测试集和验证集
    df = pd.read_pickle(data_cfg.raw_data_root + 'data.pkl')
    total_num = len(df)
    train_df = df[:int(total_num * 0.8)].reset_index()
    val_df = df[int(total_num * 0.8):int(total_num * 0.9)].reset_index()
    test_df = df[int(total_num * 0.9):].reset_index()
    return train_df, val_df, test_df

def train_val_test_data_gen(data_cfg, data_type, df):
    assert data_type in ['train', 'val', 'test']
    drug_dict = Dictionary.load(data_cfg.dataset_root + 'dict/drug_dict.dict')
    diag_dict = Dictionary.load(data_cfg.dataset_root + 'dict/diag_dict.dict')
    proc_dict = Dictionary.load(data_cfg.dataset_root + 'dict/proc_dict.dict')
    drug_pad_id = drug_dict.token2id[data_cfg.PAD_DRUG]
    # diag_pad_id = diag_dict.token2id[data_cfg.PAD_DIAG]
    # proc_pad_id = proc_dict.token2id[data_cfg.PAD_PROC]
    diag_num = len(diag_dict)
    proc_num = len(proc_dict)
    diag_data = []
    proc_data = []
    drug_data = []
    len_data = []
    for i in tqdm(range(len(df))):
        # temp_diag_data = torch.ones(data_cfg.max_diag_num) * diag_pad_id
        # temp_proc_data = torch.ones(data_cfg.max_proc_num) * proc_pad_id
        temp_diag_data = torch.zeros(diag_num)
        temp_proc_data = torch.zeros(proc_num)
        temp_drug_data = torch.ones(data_cfg.max_drug_num) * drug_pad_id
        diag = df.at[i, 'ICD9_CODE']
        proc = df.at[i, 'PRO_CODE']
        drug = df.at[i, 'DRUG']
        drug.append(data_cfg.EOS)  # 增加EOS字符
        drug_len = df.at[i, 'DRUG_Len']
        len_data.append(int(drug_len))
        for j in range(len(diag)):
            temp_diag_data[diag_dict.token2id[diag[j]]] = 1
        for j in range(len(proc)):
            temp_proc_data[proc_dict.token2id[proc[j]]] = 1
        for j in range(len(drug)):
            temp_drug_data[j] = drug_dict.token2id[drug[j]]
        diag_data.append(temp_diag_data)
        proc_data.append(temp_proc_data)
        drug_data.append(temp_drug_data)
    diag_data = torch.stack(diag_data)
    proc_data = torch.stack(proc_data)
    drug_data = torch.stack(drug_data).long()
    len_data = torch.tensor(len_data).long()
    if not os.path.exists(data_cfg.dataset_root + data_type):
        os.makedirs(data_cfg.dataset_root + data_type)
    torch.save(diag_data, data_cfg.dataset_root + '{}/diag.pt'.format(data_type))
    torch.save(proc_data, data_cfg.dataset_root + '{}/proc.pt'.format(data_type))
    torch.save(drug_data, data_cfg.dataset_root + '{}/drug.pt'.format(data_type))
    torch.save(len_data, data_cfg.dataset_root + '{}/len.pt'.format(data_type))


def one_hot_drug_gen(data_cfg, data_type, df):
    assert data_type in ['train', 'val', 'test']
    drug_dict = Dictionary.load(data_cfg.dataset_root + 'dict/drug_dict.dict')
    drug_num = len(drug_dict)
    drug_data = []
    for i in tqdm(range(len(df))):
        temp_drug_data = torch.zeros(drug_num)
        drug = df.at[i, 'DRUG']
        for j in range(len(drug)):
            temp_drug_data[drug_dict.token2id[drug[j]]] = 1
        drug_data.append(temp_drug_data)
    drug_data = torch.stack(drug_data).long()
    if not os.path.exists(data_cfg.dataset_root + data_type):
        os.makedirs(data_cfg.dataset_root + data_type)
    torch.save(drug_data, data_cfg.dataset_root + '{}/drug_onehot.pt'.format(data_type))
    
def graph_data_gen(data_cfg):
    # 读取相互作用数据
    df_drugbank = pd.read_excel(data_cfg.raw_data_root + 'DrugBank重合_翻译.xlsx')
    antagonism_set = set()
    synergism_set = set()
    for i in range(len(df_drugbank)):
        drug1 = df_drugbank.at[i, 'drug1']
        drug2 = df_drugbank.at[i, 'drug2']
        type = df_drugbank.at[i, 'Type']
        if type == 'Antagonism':
            antagonism_set.add(drug1 + drug2)
            antagonism_set.add(drug2 + drug1)
        elif type == 'Synergism':
            synergism_set.add(drug1 + drug2)
            synergism_set.add(drug2 + drug1)
    # 构造DPG中所用的相互作用图数据
    drug_dict = Dictionary.load(data_cfg.dataset_root + 'dict/drug_dict.dict')
    # print(len(drug_dict))
    # print(drug_dict[1])
    row = []
    col = []
    edge_type = []
    for i in range(len(drug_dict)):
        for j in range(len(drug_dict)):
            drug_i = drug_dict[i]
            drug_j = drug_dict[j]
            if (drug_i + drug_j) in antagonism_set:
                row.append(i)
                col.append(j)
                edge_type.append(0)
            elif (drug_i + drug_j) in synergism_set:
                row.append(i)
                col.append(j)
                edge_type.append(1)
    DPG_edge_index = torch.tensor([row, col]).long()
    DPG_edge_type = torch.tensor(edge_type).long()
    if not os.path.exists(data_cfg.dataset_root + 'graph'):
        os.makedirs(data_cfg.dataset_root + 'graph')
    torch.save(DPG_edge_index, data_cfg.dataset_root + 'graph/DPG_edge_index.pt')
    torch.save(DPG_edge_type, data_cfg.dataset_root + 'graph/DPG_edge_type.pt')
    # 构造GAMENet中所用的负面相互作用图和药品共现图
    row = []
    col = []
    for i in range(len(drug_dict)):
        for j in range(len(drug_dict)):
            drug_i = drug_dict[i]
            drug_j = drug_dict[j]
            if (drug_i + drug_j) in antagonism_set:
                row.append(i)
                col.append(j)
    negative_edge_index = torch.tensor([row, col]).long()
    torch.save(negative_edge_index, data_cfg.dataset_root + 'graph/negative_edge_index.pt')
    
    row = []
    col = []
    edge_weight = []
    df = pd.read_pickle(data_cfg.raw_data_root + 'data.pkl')
    count = torch.zeros((len(drug_dict), len(drug_dict)))
    for i in tqdm(range(len(df))):
        drug = df.at[i, 'DRUG']
        for j in range(len(drug)):
            for k in range(len(drug)):
                if k == j:
                    continue
                drug_j = drug[j]
                drug_k = drug[k]
                count[drug_dict.token2id[drug_j], drug_dict.token2id[drug_k]] += 1
                count[drug_dict.token2id[drug_k], drug_dict.token2id[drug_j]] += 1
    for i in range(len(drug_dict)):
        for j in range(len(drug_dict)):
            row.append(i)
            col.append(j)       
            edge_weight.append(int(count[i, j]))
    count_edge_index = torch.tensor([row, col]).long()
    count_edge_weight = torch.tensor(edge_weight).long()
    torch.save(count_edge_index, data_cfg.dataset_root + 'graph/count_edge_index.pt')
    torch.save(count_edge_weight, data_cfg.dataset_root + 'graph/count_edge_weight.pt')


if __name__ == '__main__':
    data_cfg = DataConfig()
    train_df, val_df, test_df = split_dataset(data_cfg)
    # train_val_test_data_gen(data_cfg, 'train', train_df)
    one_hot_drug_gen(data_cfg, 'train', train_df)
    # train_val_test_data_gen(data_cfg, 'val', val_df)
    # train_val_test_data_gen(data_cfg, 'test', test_df)
    # graph_data_gen(data_cfg)