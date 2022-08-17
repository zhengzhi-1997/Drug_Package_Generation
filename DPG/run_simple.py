import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
import argparse
from tqdm import tqdm
import codecs
import copy
import utils
from models import DPG_simple
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import train_utils
import data_utils
from config import DataConfig, ModelConfig

def train(args, data_cfg, model_cfg, generator, optimizer, train_dataloader, DPG_edge_index, DPG_edge_type):
    generator.train()
    for diags, procs, drugs, len_list in train_dataloader:
        drug_embs, edge_embs = generator.get_drug_edge_embs(DPG_edge_index)
        optimizer.zero_grad()
        if args.reinforce == 'no':
            loss = utils.MLE_loss(generator, diags, procs, drugs, len_list, data_cfg, model_cfg, drug_embs)
        else:
            loss = utils.RL_loss(generator, diags, procs, drugs, len_list, data_cfg, model_cfg, drug_embs)
        loss.backward()
        optimizer.step()


def run(args, data_cfg, model_cfg):
    train_utils.set_seed(42)
    train_dataset, val_dataset, test_dataset = data_utils.read_train_val_test_data(data_cfg, model_cfg)
    val_diag_data, val_proc_data, val_drug_data, val_len_data = val_dataset.tensors
    raw_train_drugs, raw_val_drugs, raw_test_drugs = data_utils.read_raw_drugs(data_cfg)
    DPG_edge_index, DPG_edge_type = data_utils.read_DPG_graph(data_cfg, model_cfg)
    # 构建dataloader
    train_dataloader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 设置模型和优化器
    if args.reinforce == 'no':
        generator = DPG_simple(data_cfg, model_cfg).to(model_cfg.device)
    else:
        # 读取预训练的模型
        model_root = data_cfg.result_root + 'DPG/results/models/'
        model_path = model_root + 'MLE_{}.pkl'.format(0.001)
        generator =  torch.load(model_path, map_location=model_cfg.device) 
    optimizer = torch.optim.Adam(params=generator.parameters(), lr=args.lr)
    best_F = 0
    for epoch in tqdm(range(args.num_epochs)):
        if epoch % args.eval_num == 0:
            drug_embs, edge_embs = generator.get_drug_edge_embs(DPG_edge_index)
            result_val = utils.evaluate_gready(generator, val_diag_data, val_proc_data, raw_val_drugs, data_cfg, model_cfg, drug_embs)
            if args.save == 'yes':
                if args.reinforce == 'no':
                    utils.save_simple(generator, data_cfg, result_val, result_val['F'] > best_F, epoch, 'MLE', args.lr)
                else:
                    utils.save_simple(generator, data_cfg, result_val, result_val['F'] > best_F, epoch, 'RL', args.lr)
            best_F = max(best_F, result_val['F'])
        train(args, data_cfg, model_cfg, generator, optimizer, train_dataloader, DPG_edge_index, DPG_edge_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='the number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_num', type=int, default=5, help='evaluate every eval_num epochs')
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
    parser.add_argument("--save", type=str, default="yes", help="write result to file or not")
    parser.add_argument("--train", type=str, default="no", help="train the generator or not")
    parser.add_argument("--test", type=str, default="yes", help="test the generator or not")
    parser.add_argument("--reinforce", type=str, default="yes", help="use reinforcement learning or not")
    args = parser.parse_args()

    assert args.save in ['yes', 'no']
    assert args.train in ['yes', 'no']
    assert args.test in ['yes', 'no']
    assert args.reinforce in ['yes', 'no']
    data_cfg = DataConfig()
    model_cfg = ModelConfig(data_cfg)
    model_cfg.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    train_utils.set_seed(42)
    if args.train == 'yes':
        run(args, data_cfg, model_cfg)
    if args.test == 'yes':
        if args.reinforce == 'no':
            utils.performance_test_simple(args, data_cfg, model_cfg, 'MLE', args.lr)
        else:
            utils.performance_test_simple(args, data_cfg, model_cfg, 'RL', args.lr)