# -*- coding: utf-8 -*-
class configurations(object):
    def __init__(self):
        self.batch_size = 30
        self.emb_dim = 256
        self.hid_dim = 512
        self.n_layers = 3
        self.dropout = 0.5
        self.learning_rate = 0.00005
        self.max_output_len = 50              # 最后输出句子的最大长度
        self.num_steps = 12000                # 总训练次数
        self.store_steps = 300                 # 训练多少次后存储模型
        self.summary_steps = 300               # 训练多少次后需要检验是否有overfitting
        self.load_model = False               # 是否需要载入模型
        self.store_model_path = "Model"     # 存储模型的位置
        self.load_model_path = None           # 载入模型的位置 e.g. "./ckpt/model_{step}"
        self.data_path = "Data"     # 数据存放的位置
        self.attention = False              # 是否使用Attention Mechanism
        
    def set_load_model_path(self, load_model_path):
        self.load_model_path = load_model_path
