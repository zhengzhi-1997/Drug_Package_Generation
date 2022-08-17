import torch
import torch.nn as nn

# class PatientEncoder(nn.Module):
#     # get patient embedding
#     def __init__(self, data_cfg, model_cfg):
#         super(PatientEncoder, self).__init__()
#         self.data_cfg = data_cfg
#         self.model_cfg = model_cfg
#         self.diag_char_emb_layer = nn.Embedding(self.model_cfg.diag_num, self.model_cfg.char_emb_num)
#         self.proc_char_emb_layer = nn.Embedding(self.model_cfg.proc_num, self.model_cfg.char_emb_num)
#         self.diag_GRU = nn.GRU(self.model_cfg.char_emb_num, self.model_cfg.diag_emb_num)
#         self.proc_GRU = nn.GRU(self.model_cfg.char_emb_num, self.model_cfg.proc_emb_num)

#     def forward(self, diags, procs):
#         diag_char_embs = self.diag_char_emb_layer(diags).permute(1, 0, 2)
#         diag_output, diag_state = self.diag_GRU(diag_char_embs, None)
#         diag_embs = diag_state[0]
#         proc_char_embs = self.proc_char_emb_layer(procs).permute(1, 0, 2)
#         proc_output, proc_state = self.proc_GRU(proc_char_embs, None)
#         proc_embs = proc_state[0]
#         patient_embs = torch.cat([diag_embs, proc_embs], dim=1)
#         return patient_embs

class PatientEncoder(nn.Module):
    # get patient embedding
    def __init__(self, data_cfg, model_cfg):
        super(PatientEncoder, self).__init__()
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        input_num = model_cfg.diag_num + model_cfg.proc_num
        self.MLP = nn.Sequential(
            nn.Linear(input_num, 2 * model_cfg.emb_num),
            nn.ReLU(),
            nn.Linear(2 * model_cfg.emb_num, model_cfg.emb_num)
        )

    def forward(self, diags, procs):
        return self.MLP(torch.cat([diags, procs], dim=1))