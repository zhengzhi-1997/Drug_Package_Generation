# 生成用户id、item id、字id、type id
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from config import DataConfig
import os


def id_gen(data_cfg):
    # 生成诊断、处理、药品编码id
    df = pd.read_pickle(data_cfg.raw_data_root + 'data.pkl')
    drug_dict = Dictionary(df['DRUG'].tolist())
    diag_dict = Dictionary(df['ICD9_CODE'].tolist())
    proc_dict = Dictionary(df['PRO_CODE'].tolist())
    # 增加诊断、处理、药品编码padding符
    drug_dict.add_documents([[data_cfg.PAD_DRUG, data_cfg.EOS, data_cfg.BOS]])  
    # diag_dict.add_documents([[data_cfg.PAD_DIAG]]) 
    # proc_dict.add_documents([[data_cfg.PAD_PROC]]) 
    if not os.path.exists(data_cfg.dataset_root + 'dict'):
        os.makedirs(data_cfg.dataset_root + 'dict')
    drug_dict.save(data_cfg.dataset_root + 'dict' + '/drug_dict.dict')
    diag_dict.save(data_cfg.dataset_root + 'dict' + '/diag_dict.dict')
    proc_dict.save(data_cfg.dataset_root + 'dict' + '/proc_dict.dict')

if __name__ == '__main__':
    data_cfg = DataConfig()
    id_gen(data_cfg)