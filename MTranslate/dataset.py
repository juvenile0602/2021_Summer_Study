# 定义Dataset
# Data (出自manythings 的 cmn-eng):
# 訓練資料：18000句
# 檢驗資料： 500句
# 測試資料： 2636句

# 数据预处理：
# 英文
# 用subword-nmt套件，将word转为subword
# 建立字典，取出标签中出现频率高于定值的subword
# 中文
# 用jieba将中文句子分词
# 建立字典，取出标签中出现频率高于定值的词
# 特殊字符
# < PAD > ：无意义，將句子拓展到相同长度
# < BOS > ：Begin of sentence
# < EOS > ：End of sentence
# < UNK > ：单字没有出现在字典里的字

# 将字典里每个subword(词)用一个整数表示，分为英文和中文的词典，方便之后转为one-hot向量

# 处理后的数据:
# 字典：
# int2word_*.json 将整数转为文字，如int2word_en.json
# word2int_*.json 将文字转为整数，如word2int_en.json

# 训练数据：
# 不同语言的句子用TAB('\t')分开
# 字与字之间用空格分开

# 在将答案传出去之前，在答案开头加入< BOS >，在答案结尾加入< EOS >符号

import re
import json
import os
from LabelTransform import LabelTransform
import numpy as np
import torch
import torch.utils.data as data


class EN2CNDataset(data.Dataset):
    # root为数据根目录
    # max_output_len为输出句子的最大长度
    # set_name为载入数据的名称
    def __init__(self, root, max_output_len, set_name):
        self.root = root
        self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
        self.word2int_en, self.int2word_en = self.get_dictionary('en')

        # 载入数据
        self.data = []
        with open(os.path.join(self.root, f'{set_name}.txt'), "r", encoding='UTF-8') as f:
            for line in f:
                self.data.append(line)
        print(f'{set_name} dataset size: {len(self.data)}')

        self.cn_vocab_size = len(self.word2int_cn)  # 中文词表大小
        self.en_vocab_size = len(self.word2int_en)  # 英文词表大小
        # 创建一个LabelTransform的实例
        # 用<PAD>对应的整数作填充
        self.transform = LabelTransform(
            max_output_len, self.word2int_en['<PAD>'])

    # 载入字典
    def get_dictionary(self, language):
        with open(os.path.join(self.root, f'word2int_{language}.json'), "r", encoding='UTF-8') as f:
            word2int = json.load(f)
        with open(os.path.join(self.root, f'int2word_{language}.json'), "r", encoding='UTF-8') as f:
            int2word = json.load(f)
        return word2int, int2word  # 返回的是两个dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, Index):
        # 先将中英文分开
        sentences = self.data[Index]
        sentences = re.split('[\t\n]', sentences)
        sentences = list(filter(None, sentences))
        # print(sentences)
        assert len(sentences) == 2

        # 预备特殊字符
        BOS = self.word2int_en['<BOS>']
        EOS = self.word2int_en['<EOS>']
        UNK = self.word2int_en['<UNK>']

        # 在开头添加<BOS>,在结尾添加<EOS>,不在字典中的subword(词)用<UNK>取代
        # 初始化代表英文与中文的index序列
        en, cn = [BOS], [BOS]
        # 将英文句子拆解为subword并转为整数
        # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
        sentence = re.split(' ', sentences[0])
        sentence = list(filter(None, sentence))
        # dict.get(key, default=None)
        # key -- 字典中要查找的键。
        # default -- 如果指定键的值不存在时，返回该默认值。
        for word in sentence:
            en.append(self.word2int_en.get(word, UNK))
        en.append(EOS)

        # 将中文句子拆解为单词并转为整数
        sentence = re.split(' ', sentences[1])
        sentence = list(filter(None, sentence))
        for word in sentence:
            cn.append(self.word2int_cn.get(word, UNK))
        cn.append(EOS)

        en, cn = np.asarray(en), np.asarray(cn)

        # 用<PAD>将句子补到相同长度
        en, cn = self.transform(en), self.transform(cn)
        en, cn = torch.LongTensor(en), torch.LongTensor(cn)

        return en, cn
