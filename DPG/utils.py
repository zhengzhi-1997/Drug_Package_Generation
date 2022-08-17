import collections
import copy
from heapq import heappush, heappop
import os
import io
import math
import sys
from typing import Generator
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import codecs
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import train_utils
import data_utils
from config import DataConfig, ModelConfig
from reward import get_F_reward

def MLE_loss(generator, diags, procs, drugs, len_list, data_cfg, model_cfg, drug_embs):
    loss = nn.CrossEntropyLoss(reduction='none')
    batch_size = diags.size(0)
    patient_embs = generator.get_patient_embs(diags, procs)
    drug_num = model_cfg.drug_num
    I = torch.zeros(batch_size, drug_num).to(model_cfg.device)
    col_index = torch.arange(drug_num).to(model_cfg.device)
    state = patient_embs.unsqueeze(0)
    input = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.BOS]] * batch_size).to(model_cfg.device)
    mask, num_not_pad_tokens = torch.ones(batch_size,).to(model_cfg.device), 0
    l = torch.tensor([0.0]).to(model_cfg.device)
    previous_drugs = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.PAD_DRUG]] * batch_size).reshape(batch_size, -1).to(model_cfg.device)
    for d in drugs.permute(1, 0):  # d is 1-D vector
        output, state = generator(input, state, drug_embs, previous_drugs, patient_embs, I)
        previous_drugs = torch.cat([previous_drugs, input.reshape(batch_size, -1)], dim=1)
        l = l + (mask * loss(output, d)).sum()
        input = d
        num_not_pad_tokens += mask.sum().cpu().item()
        mask = mask * (d != model_cfg.drug_dict.token2id[data_cfg.EOS]).float()
        I_mask = d[:, None] == col_index[None, :]
        # I.masked_fill_(I_mask, float('-inf'))
        I.masked_fill_(I_mask, -100000)
    
    # exit(0)
    return l / num_not_pad_tokens


def RL_loss(generator, diags, procs, drugs, len_list, data_cfg, model_cfg, drug_embs):
    batch_masks = []
    batch_logps = []
    batch_acts = []
    batch_size = diags.size(0)
    patient_embs = generator.get_patient_embs(diags, procs)
    drug_num = model_cfg.drug_num
    state = patient_embs.unsqueeze(0)
    input = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.BOS]] * batch_size).to(model_cfg.device)
    mask = torch.ones(batch_size).to(model_cfg.device)
    gen_len_list = torch.zeros(batch_size).to(model_cfg.device)  # 生成药品包的真实药品数量列表
    time_step = 0
    I = torch.zeros(batch_size, drug_num).to(model_cfg.device)
    col_index = torch.arange(drug_num).to(model_cfg.device)
    previous_drugs = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.PAD_DRUG]] * batch_size).reshape(batch_size, -1).to(model_cfg.device)
    # 使用蒙塔卡罗采样出一个序列
    while time_step < data_cfg.max_drug_num:
        time_step += 1
        output, state = generator(input, state, drug_embs, previous_drugs, patient_embs, I)  # output:[batch_size * drug_num]
        previous_drugs = torch.cat([previous_drugs, input.reshape(batch_size, -1)], dim=1)
        c = Categorical(logits=output)
        act = c.sample()  # act:[batch_size]
        input = act
        batch_acts.append(act)
        logp = c.log_prob(act)  # logp:[batch_size]
        batch_logps.append(logp)
        batch_masks.append(mask)
        mask = mask * (act != model_cfg.drug_dict.token2id[data_cfg.EOS]).float()
        gen_len_list += mask
        I_mask = act[:, None] == col_index[None, :]
        I.masked_fill_(I_mask, float('-inf'))
        # I.masked_fill_(I_mask, -100000)

    # 使用gready_search得到baseline序列
    I = torch.zeros(batch_size, drug_num).to(model_cfg.device)
    baseline_acts = []
    patient_embs = generator.get_patient_embs(diags, procs)
    state = patient_embs.unsqueeze(0)
    input = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.BOS]] * batch_size).to(model_cfg.device)
    mask = torch.ones(batch_size).to(model_cfg.device)
    baseline_gen_len_list = torch.zeros(batch_size).to(model_cfg.device)  # 生成药品包的真实药品数量列表
    time_step = 0
    previous_drugs = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.PAD_DRUG]] * batch_size).reshape(batch_size, -1).to(model_cfg.device)
    while time_step < data_cfg.max_drug_num:
        time_step += 1
        output, state = generator(input, state, drug_embs, previous_drugs, patient_embs, I)  # output:[batch_size * drug_num]
        previous_drugs = torch.cat([previous_drugs, input.reshape(batch_size, -1)], dim=1)
        act = output.argmax(dim=1)  # act:[batch_size]
        input = act
        baseline_acts.append(act)
        mask = mask * (act != model_cfg.drug_dict.token2id[data_cfg.EOS]).float()
        baseline_gen_len_list += mask
        I_mask = act[:, None] == col_index[None, :]
        I.masked_fill_(I_mask, float('-inf'))
        # I.masked_fill_(I_mask, -100000)

    gen_drugs = torch.stack(batch_acts).t()
    baseline_drugs = torch.stack(baseline_acts).t()

    gen_reward = torch.tensor(get_F_reward(drugs.int().cpu().numpy(), len_list.int().cpu().numpy(),
    gen_drugs.int().cpu().numpy(), gen_len_list.int().cpu().numpy())).to(model_cfg.device)

    baseline_reward = torch.tensor(get_F_reward(drugs.int().cpu().numpy(), len_list.int().cpu().numpy(),
    baseline_drugs.int().cpu().numpy(), baseline_gen_len_list.int().cpu().numpy())).to(model_cfg.device)

    batch_logps = torch.stack(batch_logps)
    batch_masks = torch.stack(batch_masks)

    reward = (baseline_reward - gen_reward).reshape(-1, 1)
    return (reward * batch_logps.t()).masked_select(batch_masks.bool().t()).mean()

def gready_search(generator, diags, procs, data_cfg, model_cfg, drug_embs):
    drug_num = model_cfg.drug_num
    patient_embs = generator.get_patient_embs(diags.unsqueeze(0), procs.unsqueeze(0))
    state = patient_embs.unsqueeze(0)
    input = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.BOS]]).to(model_cfg.device)
    output_drugs = []
    I = torch.zeros(1, drug_num).to(model_cfg.device)
    previous_drugs = torch.tensor([model_cfg.drug_dict.token2id[data_cfg.PAD_DRUG]] * 1).reshape(1, -1).to(model_cfg.device)
    for _ in range(data_cfg.max_drug_num):
        output, state = generator(input, state, drug_embs, previous_drugs, patient_embs, I)  # output:[1, drug_num]
        previous_drugs = torch.cat([previous_drugs, input.reshape(1, -1)], dim=1)
        pred = output.argmax(dim=1)  # prde:[1]
        I[0, int(pred.item())] = float('-inf')
        pred_drug = model_cfg.drug_dict[int(pred.item())]
        if pred_drug == data_cfg.EOS:
            break
        else:
            input = pred
            output_drugs.append(pred_drug)
    return output_drugs

def evaluate_gready(generator, test_diags, test_procs, raw_test_drugs, data_cfg, model_cfg, drug_embs):
    generator.eval()
    result = {}
    with torch.no_grad():
        P = 0.0
        R = 0.0
        F = 0.0
        total_num = len(raw_test_drugs)
        for i in range(total_num):
            output_drugs = set(gready_search(generator, test_diags[i], test_procs[i], data_cfg, model_cfg, drug_embs))
            # print(output_drugs)
            ground_truth = set(raw_test_drugs[i])
            true_positive = len(output_drugs & ground_truth)
            if len(output_drugs) == 0 or len(ground_truth) == 0:
                temp_P = 0
                temp_R = 0
            else:
                temp_P = true_positive / len(output_drugs)
                temp_R = true_positive / len(ground_truth)
            P += temp_P
            R += temp_R
            if (temp_P + temp_R) == 0:
                F += 0
            else:
                F += 2 * temp_P * temp_R / (temp_P + temp_R)
    generator.train()
    result['P'] = P / total_num
    result['R'] = R / total_num
    result['F'] = F / total_num
    return result

def save(model, data_cfg, result_val, save_model, epoch, name, alpha, beta, reinforce, lr):
    # 保存结果和模型
    root = data_cfg.result_root + 'DPG/results/'
    model_root = data_cfg.result_root + 'DPG/results/models/'
    result_root = data_cfg.result_root + 'DPG/results/results/val/'
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    if reinforce == 'no':
        model_path = model_root + name + '_{}.pkl'.format(alpha)
        val_result_path = result_root + name + '_{}.txt'.format(alpha)
    else:
        model_path = model_root + name + '_{}_{}_{}.pkl'.format(alpha, beta, lr)
        val_result_path = result_root + name + '_{}_{}_{}.txt'.format(alpha, beta, lr)
    if save_model == True:
        best_model = copy.deepcopy(model)
        torch.save(best_model, model_path)
    with codecs.open(val_result_path, 'a') as f:
        f.write('epoch {}\n'.format(epoch))
        for key in result_val:
            f.write('{}  {}\n'.format(key, result_val[key]))

def performance_test(args, data_cfg, model_cfg, name, alpha, beta, reinforce, lr):
    result_root = data_cfg.result_root + 'DPG/results/results/test/'
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    model_root = data_cfg.result_root + 'DPG/results/models/'
    if reinforce == 'no':
        model_path = model_root + name + '_{}.pkl'.format(alpha)
    else:
        model_path = model_root + name + '_{}_{}_{}.pkl'.format(alpha, beta, lr)
    generator =  torch.load(model_path, map_location=model_cfg.device)
    generator.eval()
    train_dataset, val_dataset, test_dataset = data_utils.read_train_val_test_data(data_cfg, model_cfg)
    test_diag_data, test_proc_data, test_drug_data, test_len_data = test_dataset.tensors
    raw_train_drugs, raw_val_drugs, raw_test_drugs = data_utils.read_raw_drugs(data_cfg)
    DPG_edge_index, DPG_edge_type = data_utils.read_DPG_graph(data_cfg, model_cfg)
    drug_embs, edge_embs = generator.get_drug_edge_embs(DPG_edge_index)
    result_test = evaluate_gready(generator, test_diag_data, test_proc_data, raw_test_drugs, data_cfg, model_cfg, drug_embs)
    if reinforce == 'no':
        test_result_path = result_root + name + '_{}.txt'.format(alpha)
    else:
        test_result_path = result_root + name + '_{}_{}_{}.txt'.format(alpha, beta, lr)
    with codecs.open(test_result_path, 'a') as f:
        for key in result_test:
            f.write('{}  {}\n'.format(key, result_test[key]))


def save_simple(model, data_cfg, result_val, save_model, epoch, name, lr):
    # 保存结果和模型
    root = data_cfg.result_root + 'DPG/results/'
    model_root = data_cfg.result_root + 'DPG/results/models/'
    result_root = data_cfg.result_root + 'DPG/results/results/val/'
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    model_path = model_root + name + '_{}.pkl'.format(lr)
    val_result_path = result_root + name + '_{}.txt'.format(lr)
    if save_model == True:
        best_model = copy.deepcopy(model)
        torch.save(best_model, model_path)
    with codecs.open(val_result_path, 'a') as f:
        f.write('epoch {}\n'.format(epoch))
        for key in result_val:
            f.write('{}  {}\n'.format(key, result_val[key]))

def performance_test_simple(args, data_cfg, model_cfg, name, lr):
    result_root = data_cfg.result_root + 'DPG/results/results/test/'
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    model_root = data_cfg.result_root + 'DPG/results/models/'
    model_path = model_root + name + '_{}.pkl'.format(lr)
    generator =  torch.load(model_path, map_location=model_cfg.device)
    generator.eval()
    train_dataset, val_dataset, test_dataset = data_utils.read_train_val_test_data(data_cfg, model_cfg)
    test_diag_data, test_proc_data, test_drug_data, test_len_data = test_dataset.tensors
    raw_train_drugs, raw_val_drugs, raw_test_drugs = data_utils.read_raw_drugs(data_cfg)
    DPG_edge_index, DPG_edge_type = data_utils.read_DPG_graph(data_cfg, model_cfg)
    drug_embs, edge_embs = generator.get_drug_edge_embs(DPG_edge_index)
    result_test = evaluate_gready(generator, test_diag_data, test_proc_data, raw_test_drugs, data_cfg, model_cfg, drug_embs)
    test_result_path = result_root + name + '_{}.txt'.format(lr)
    with codecs.open(test_result_path, 'a') as f:
        for key in result_test:
            f.write('{}  {}\n'.format(key, result_test[key]))

