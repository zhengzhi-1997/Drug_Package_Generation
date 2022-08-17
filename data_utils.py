import torch
import torch.utils.data as Data
import pandas as pd

def read_train_val_test_data(data_cfg, model_cfg):
    train_diag_data = torch.load(data_cfg.dataset_root + 'train/diag.pt', map_location=model_cfg.device)
    train_proc_data = torch.load(data_cfg.dataset_root + 'train/proc.pt', map_location=model_cfg.device)
    train_drug_data = torch.load(data_cfg.dataset_root + 'train/drug.pt', map_location=model_cfg.device)
    train_len_data = torch.load(data_cfg.dataset_root + 'train/len.pt', map_location=model_cfg.device)
    val_diag_data = torch.load(data_cfg.dataset_root + 'val/diag.pt', map_location=model_cfg.device)
    val_proc_data = torch.load(data_cfg.dataset_root + 'val/proc.pt', map_location=model_cfg.device)
    val_drug_data = torch.load(data_cfg.dataset_root + 'val/drug.pt', map_location=model_cfg.device)
    val_len_data = torch.load(data_cfg.dataset_root + 'val/len.pt', map_location=model_cfg.device)
    test_diag_data = torch.load(data_cfg.dataset_root + 'test/diag.pt', map_location=model_cfg.device)
    test_proc_data = torch.load(data_cfg.dataset_root + 'test/proc.pt', map_location=model_cfg.device)
    test_drug_data = torch.load(data_cfg.dataset_root + 'test/drug.pt', map_location=model_cfg.device)
    test_len_data = torch.load(data_cfg.dataset_root + 'test/len.pt', map_location=model_cfg.device)
    train_dataset = Data.TensorDataset(train_diag_data, train_proc_data, train_drug_data, train_len_data)
    val_dataset = Data.TensorDataset(val_diag_data, val_proc_data, val_drug_data, val_len_data)
    test_dataset = Data.TensorDataset(test_diag_data, test_proc_data, test_drug_data, test_len_data)
    return train_dataset, val_dataset, test_dataset

def read_one_hot_drugs(data_cfg, model_cfg):
    return torch.load(data_cfg.dataset_root + 'train/drug_onehot.pt', map_location=model_cfg.device)

def read_DPG_graph(data_cfg, model_cfg):
    DPG_edge_index = torch.load(data_cfg.dataset_root + 'graph/DPG_edge_index.pt', map_location=model_cfg.device)
    DPG_edge_type = torch.load(data_cfg.dataset_root + 'graph/DPG_edge_type.pt', map_location=model_cfg.device)
    return DPG_edge_index, DPG_edge_type

def read_GAME_graph(data_cfg, model_cfg):
    negative_edge_index = torch.load(data_cfg.dataset_root + 'graph/negative_edge_index.pt', map_location=model_cfg.device)
    count_edge_index = torch.load(data_cfg.dataset_root + 'graph/count_edge_index.pt', map_location=model_cfg.device)
    count_edge_weight = torch.load(data_cfg.dataset_root + 'graph/count_edge_weight.pt', map_location=model_cfg.device)
    return negative_edge_index, count_edge_index, count_edge_weight

def read_raw_drugs(data_cfg):
    df = pd.read_pickle(data_cfg.raw_data_root + 'data.pkl')
    total_num = len(df)
    train_df = df[:int(total_num * 0.8)].reset_index()
    val_df = df[int(total_num * 0.8):int(total_num * 0.9)].reset_index()
    test_df = df[int(total_num * 0.9):].reset_index()
    return train_df['DRUG'].tolist(), val_df['DRUG'].tolist(), test_df['DRUG'].tolist()

def read_raw_diags(data_cfg):
    df = pd.read_pickle(data_cfg.raw_data_root + 'data.pkl')
    total_num = len(df)
    train_df = df[:int(total_num * 0.8)].reset_index()
    val_df = df[int(total_num * 0.8):int(total_num * 0.9)].reset_index()
    test_df = df[int(total_num * 0.9):].reset_index()
    return train_df['ICD9_CODE'].tolist(), val_df['ICD9_CODE'].tolist(), test_df['ICD9_CODE'].tolist()

def read_raw_procs(data_cfg):
    df = pd.read_pickle(data_cfg.raw_data_root + 'data.pkl')
    total_num = len(df)
    train_df = df[:int(total_num * 0.8)].reset_index()
    val_df = df[int(total_num * 0.8):int(total_num * 0.9)].reset_index()
    test_df = df[int(total_num * 0.9):].reset_index()
    return train_df['PRO_CODE'].tolist(), val_df['PRO_CODE'].tolist(), test_df['PRO_CODE'].tolist()