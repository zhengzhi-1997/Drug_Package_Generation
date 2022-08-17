from gensim.corpora.dictionary import Dictionary
import os
class DataConfig():
    def __init__(self):
        self.BOS = '<bos>'
        self.EOS = '<eos>'
        self.PAD_DIAG = '<pad_diag>'
        self.PAD_PROC = '<pad_proc>'
        self.PAD_DRUG = '<pad_drug>'
        self.raw_data_root = os.path.expanduser('~') + '/data/MIMIC/'  # 原始csv文件保存位置
        self.dataset_root = os.path.expanduser('~') + '/data/MIMIC/dataset/'  # 生成的数据集保存位置
        self.result_root = os.path.expanduser('~') + '/code/DPRTrans/MIMIC/'
        self.max_diag_num = 40  # 最大诊断编码数量
        self.max_proc_num = 25  # 最大处理过程编码数量
        self.max_drug_num = 31  # 最大药品包长度。最大药品数量30，增加EOS后为31.

class ModelConfig():
    def __init__(self, data_cfg):
        self.drug_dict = Dictionary.load(data_cfg.dataset_root + 'dict/drug_dict.dict')
        self.diag_dict = Dictionary.load(data_cfg.dataset_root + 'dict/diag_dict.dict')
        self.proc_dict = Dictionary.load(data_cfg.dataset_root + 'dict/proc_dict.dict')
        self.drug_num = len(self.drug_dict)
        self.diag_num = len(self.diag_dict)
        self.proc_num = len(self.proc_dict)
        self.char_emb_num = 64
        self.diag_emb_num = 32
        self.proc_emb_num = 32
        self.drug_emb_num = 64
        self.emb_num = 64

