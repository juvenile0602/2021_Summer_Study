import torch
import Bert
import os
from transformers import BertForSequenceClassification, AdamW,BertTokenizer,BertModel
 # 测试
if __name__=="__main__":
    params_dir='./Model/bert_base_model_beta.pkl'

    path='./Data/bert_base_uncased/'
    model=BertForSequenceClassification.from_pretrained(path)
    model.load_state_dict(torch.load(params_dir))
    Bert.test_model(model)
