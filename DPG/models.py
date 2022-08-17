import torch
import torch.nn as nn
import sys
import os
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from PatientEncoder import PatientEncoder


class Edge_GCN(MessagePassing):
    def __init__(self, x_dim):
        super(Edge_GCN, self).__init__(aggr='mean')
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * x_dim, x_dim),
            nn.ReLU(),
            nn.Linear(x_dim, x_dim)
        )
    def forward(self, x, edge_index, edge_attr):
        receive_message = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.update_mlp(torch.cat([x, receive_message], dim=1))
    def message(self, x_j, edge_attr):
        return edge_attr

class DPG(nn.Module):
    # 同时使用EdgeGNN进行药品表征，并考虑生成过程中的药品关系，使用mask对边向量进行更新，并使用求和聚合，主要模型DPG！
    def __init__(self, data_cfg, model_cfg):
        super(DPG, self).__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.patient_encoder = PatientEncoder(data_cfg, model_cfg)
        self.drug_emb_layer = nn.Embedding(self.model_cfg.drug_num, self.model_cfg.drug_emb_num)
        self.emb_num = self.model_cfg.emb_num
        self.GRU = nn.GRU(self.emb_num * 2, self.emb_num)
        self.out = nn.Linear(self.model_cfg.emb_num, self.model_cfg.drug_num)
        self.edge_emb_mlp_1 = nn.Sequential(
            nn.Linear(2 * self.emb_num, self.emb_num),
            nn.ReLU(),
            nn.Linear(self.emb_num, self.emb_num)
        )
        self.edge_emb_mlp_2 = nn.Sequential(
            nn.Linear(2 * self.emb_num, self.emb_num),
            nn.ReLU(),
            nn.Linear(self.emb_num, self.emb_num)
        )

        self.edge_emb_mlp_3 = nn.Sequential(
            nn.Linear(2 * self.emb_num, self.emb_num),
            nn.ReLU(),
            nn.Linear(self.emb_num, self.emb_num),
        )
        self.gnn_1 = Edge_GCN(self.emb_num)
        self.gnn_2 = Edge_GCN(self.emb_num)
        self.edge_predict_net = nn.Linear(self.emb_num, 3)
        self.mask_layer = nn.Sequential(
            nn.Linear(self.emb_num, self.emb_num),
            nn.Tanh(),
            nn.Linear(self.emb_num, self.emb_num),
            nn.Sigmoid()
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(self.emb_num, self.emb_num),
            nn.Tanh(),
            nn.Linear(self.emb_num, self.emb_num),
        )
    
    def get_patient_embs(self, diags, procs):
        return self.patient_encoder(diags, procs)

    def get_drug_edge_embs(self, edge_index):
        initial_drug_embs = self.drug_emb_layer.weight
        row, col = edge_index
        row_drugs_1 = initial_drug_embs[row]
        col_drugs_1 = initial_drug_embs[col]
        edge_attr_1 = self.edge_emb_mlp_1(torch.cat([row_drugs_1, col_drugs_1], dim=1))
        x = F.relu(self.gnn_1(initial_drug_embs, edge_index, edge_attr_1))
        row_drugs_2 = x[row]
        col_drugs_2 = x[col]
        edge_attr_2 = self.edge_emb_mlp_2(torch.cat([row_drugs_2, col_drugs_2], dim=1))
        drug_embs = self.gnn_2(x, edge_index, edge_attr_2)
        row_drugs_3 = drug_embs[row]
        col_drugs_3 = drug_embs[col]
        edge_embs = self.edge_emb_mlp_3(torch.cat([row_drugs_3, col_drugs_3], dim=1))
        return drug_embs, edge_embs
    
    def get_edge_predict(self, edge_embs):
        return self.edge_predict_net(edge_embs)

    def forward(self, cur_drugs, state, drug_embs, previous_drugs, patient_embs, I):
        # previous_drugs [batch_size * gen_num]
        # 引入相互作用向量
        gen_num = previous_drugs.size(1)
        drug_input = drug_embs[cur_drugs]  # [batch*size * emb_num]
        previous_drugs_embs = drug_embs[previous_drugs]  # [batch_size * gen_num * emb_num]
        repeat_drug_input = drug_input.repeat(1, previous_drugs.size(1)).reshape(previous_drugs_embs.size())
        # 计算新引入的相互作用向量
        relation_embs = self.edge_emb_mlp_3(torch.cat([repeat_drug_input, previous_drugs_embs], dim=2))  # [batch_size * gen_num * emb_num]
        repeated_patient_embeddings = patient_embs.repeat(1, gen_num).reshape(relation_embs.size())
        repeated_mask = self.mask_layer(repeated_patient_embeddings)
        relation_input = self.relation_mlp((repeated_mask * relation_embs).sum(dim=1))
        input = torch.cat([drug_input, relation_input], dim=1)
        output, state = self.GRU(input.unsqueeze(0), state)
        output = self.out(output).squeeze(dim=0)
        output = output + I
        return output, state 


class DPG_simple(nn.Module):
    def __init__(self, data_cfg, model_cfg):
        super(DPG_simple, self).__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.patient_encoder = PatientEncoder(data_cfg, model_cfg)
        self.drug_emb_layer = nn.Embedding(self.model_cfg.drug_num, self.model_cfg.drug_emb_num)
        self.emb_num = self.model_cfg.emb_num
        self.GRU = nn.GRU(self.emb_num * 2, self.emb_num)
        self.out = nn.Linear(self.model_cfg.emb_num, self.model_cfg.drug_num)
        self.edge_emb_mlp_3 = nn.Sequential(
            nn.Linear(2 * self.emb_num, self.emb_num),
            nn.ReLU(),
            nn.Linear(self.emb_num, self.emb_num)
        )
        self.mask_layer = nn.Sequential(
            nn.Linear(self.emb_num, self.emb_num),
            nn.Tanh(),
            nn.Linear(self.emb_num, self.emb_num),
            nn.Sigmoid()
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(self.emb_num, self.emb_num),
            nn.Tanh(),
            nn.Linear(self.emb_num, self.emb_num),
        )
    
    def get_patient_embs(self, diags, procs):
        return self.patient_encoder(diags, procs)

    def get_drug_edge_embs(self, edge_index):
        drug_embs = self.drug_emb_layer.weight
        return drug_embs, None
    

    def forward(self, cur_drugs, state, drug_embs, previous_drugs, patient_embs, I):
        # previous_drugs [batch_size * gen_num]
        # 引入相互作用向量
        gen_num = previous_drugs.size(1)
        drug_input = drug_embs[cur_drugs]  # [batch*size * emb_num]
        previous_drugs_embs = drug_embs[previous_drugs]  # [batch_size * gen_num * emb_num]
        repeat_drug_input = drug_input.repeat(1, previous_drugs.size(1)).reshape(previous_drugs_embs.size())
        # 计算新引入的相互作用向量
        relation_embs = self.edge_emb_mlp_3(torch.cat([repeat_drug_input, previous_drugs_embs], dim=2))  # [batch_size * gen_num * emb_num]
        repeated_patient_embeddings = patient_embs.repeat(1, gen_num).reshape(relation_embs.size())
        repeated_mask = self.mask_layer(repeated_patient_embeddings)
        relation_input = self.relation_mlp((repeated_mask * relation_embs).sum(dim=1))
        input = torch.cat([drug_input, relation_input], dim=1)
        output, state = self.GRU(input.unsqueeze(0), state)
        output = self.out(output).squeeze(dim=0)
        output = output + I
        return output, state 

