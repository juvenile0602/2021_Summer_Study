import torch
import torch.nn as nn
import random
from torch.nn import functional as F

# Encoder
# seq2seq模型的编码器为RNN。
# 对于每个输入，Encoder会输出一个向量和一个隐藏状态（hidden state）,并将隐藏状态用于下一个输入
# Encoder会逐步读取输入序列，并输出单个向量（最终隐藏状态）

# 参数：
# en_vocab_size:  英文词典的大小，也就是英文的subword的个数
# emb_dim:   embedding的维度，将one-hot的单词向量压缩到指定的维度
#            可以使用预先训练好的word embedding,如Glove和word2vector
#            设置self.embeddings.weight.requires_grad = False
# hid_dim:   RNN输出和隐藏状态的维度
# n_layers:  RNN要叠多少层
# dropout:   决定有大多的几率将某个节点变为0，主要是防止overfitting，一般在训练时使用，测试时不使用

# Encoder的输入和输出：
# 输入：
# 英文的整数序列，例如：1,28,29,205,2
# 输出：
# outputs:最上层RNN全部的输出，可以用Attention再进行处理
# hidden:每层最后的隐藏状态，将传导到Decoder进行解码

# nn.GRU的参数
# input_size – The number of expected features in the input x
# hidden_size – The number of features in the hidden state h
# num_layers – Number of recurrent layers.
# E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU,
# with the second GRU taking in outputs of the first GRU and computing the final results.
# batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).


class Encoder(nn.Module):
    def __init__(self, en_vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        # nn.Embedding进行默认随机赋值
        # 参数1:num_embeddings (int) – size of the dictionary of embeddings
        # 参数2:embedding_dim (int) – the size of each embedding vector
        self.embedding = nn.Embedding(en_vocab_size, emb_dim)
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers,
                          dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        # input = [batch size, sequence len]
        # 每个元素为一个int，取值在[1, en_vocab_size]
        # 注意nn.embedding的输入只能是编号！这里的注释有迷惑性！
        embedding = self.embedding(input)
        outputs, hidden = self.rnn(self.dropout(embedding))
        # outputs = [batch size, sequence len, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        # outputs 是最上层RNN的输出
        return outputs, hidden


# Decoder
# Decoder是另一个RNN，在最简单的seq2seq decoder中，仅使用Encoder每一层最后的隐藏状态进行解码
# 而这最后的隐藏状态有时被称为"content vector"，因为可以想象它对整个前文序列进行编码
# 此"content vector"用作Decoder的初始隐藏状态
# 而Encoder的输出通常用于注意力机制计算

# 参数：
# en_vocab_size:  英文词典的大小，也就是英文的subword的个数
# emb_dim:     embedding的维度，将one-hot的单词向量压缩到指定的维度，可以使用预先训练好的word embedding,如Glove和word2vector
# hid_dim:     RNN输出和隐藏状态的维度
# output_dim:  最终输出的维度，一般是将hid_dim转到one-hot vector的单词向量
# n_layers:    RNN要叠多少层
# isatt:       决定是否使用注意力机制

# Decoder的输入和输出:
# 输入：
# 前一次解码出来的单词的整数表示
# 输出：
# hidden: 根据输入和前一次的隐藏状态，现在的隐藏状态更新的结果
# output: 每个字有多少概率是这次解码的结果

class Decoder(nn.Module):

    def __init__(self, cn_vocab_size, emb_dim, hid_dim, n_layers, dropout, isatt):
        super().__init__()
        self.cn_vocab_size = cn_vocab_size
        # 因为Encoder采用双向GRU
        self.hid_dim = hid_dim * 2
        self.n_layers = n_layers
        self.embedding = nn.Embedding(cn_vocab_size, emb_dim)
        self.isatt = isatt
        self.attention = Attention(hid_dim)
        # 如果使用 Attention Mechanism 會使得輸入維度變化，請在這裡修改
        if isatt:
            # e.g. Attention 接在输入后面会使维度变化，所以输入维度改为
            self.input_dim = emb_dim + hid_dim * 2
        else:
            self.input_dim = emb_dim
        # 这里提前知不知道翻译结果，不能双向注意力流
        self.rnn = nn.GRU(self.input_dim, self.hid_dim,
                          self.n_layers, dropout=dropout, batch_first=True)
        self.embedding2vocab1 = nn.Linear(self.hid_dim, self.hid_dim * 2)
        self.embedding2vocab2 = nn.Linear(self.hid_dim * 2, self.hid_dim * 4)
        self.embedding2vocab3 = nn.Linear(self.hid_dim * 4, self.cn_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size, vocab size]
        # hidden = [batch size, n layers * directions, hid dim]
        # Encoder的输出：outputs = [batch size, sequence len, hid dim * directions]
        # Decoder 只会是单向，所以 directions=1
        input = input.unsqueeze(1)  # [batch size,1,vocab size ]
        # [batch_size,1, emb_dim]
        embedded = self.dropout(self.embedding(input))
        if self.isatt:
            # encoder_outputs:最上层RNN全部的输出，可以用Attention再进行处理
            attn = self.attention(encoder_outputs, hidden)
        # TODO: 在這裡決定如何使用 Attention，e.g. 相加 或是 接在後面， 請注意維度變化
        output, hidden = self.rnn(embedded, hidden)
        # output = [batch size, 1, hid dim]
        # hidden = [num_layers, batch size, hid dim]

        # 将RNN的输出转为每个词的输出概率
        # 相当于通过连接一个前馈神经网络，实现词表大小的多分类器
        output = self.embedding2vocab1(output.squeeze(1))
        output = self.embedding2vocab2(output)
        prediction = self.embedding2vocab3(output)
        # prediction = [batch size, vocab size]
        return prediction, hidden


# Attention
# 当输入过长，或是单独靠“content vector”无法取得整个输入的意思时
# 用注意力机制来提供Decoder更多的信息
# 主要是根据现在Decoder的hidden state，去计算在Encoder outputs中，与其有较高的关系
# 根据关系的数值来决定该传给Decoder哪些额外信息
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.hid_dim = hid_dim

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs = [batch size, sequence len, hid dim * directions]
        # decoder_hidden = [num_layers, batch size, hid dim]
        # 一般來說是取最後一層的 hidden state 來做 attention
        ########
        # TODO #
        ########
        attention = None

        return attention


# 由Encoder和Decoder组成
# 接受输入并传给Encoder
# 将Encoder的输出传给Decoder
# 不断地将Decoder的输出传回Decoder，进行解码
# 当解码完成后，将Decoder的输出传回
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, input, target, teacher_forcing_ratio):
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        # teacher_forcing_ratio 有多少几率使用正确数据来训练
        batch_size = target.shape[0]  # 和config中相同，为60
        target_len = target.shape[1]  # 和config中相同，为50
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, target_len,
                              vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)

        # Encoder最后的隐藏层(hidden state)用来初始化Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因为Encoder是双向的RNN，所以需要将同一层两个方向的hidden state接在一起
        # .view()用来做reshape
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        # 取切片降一维，拼成[num_layers, batch size  ,hid dim*2]
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取得 <BOS> token作为第一个输入
        input = target[:, 0]
        preds = []
        for t in range(1, target_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # output:[batch size, vocab size]
            outputs[:, t] = output
            # 决定是否用正确答案来做训练
            teacher_force = random.random() <= teacher_forcing_ratio
            # 取出概率最大的单词（batch_size大小）
            top1 = output.argmax(1)
            # 如果是 teacher force,则用正确的数据输入，反之用自己预测的数据做输入
            if teacher_force and t < target_len:
                input = target[:, t]
            else:
                input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)  # [batch_szie, target_len-1]
        return outputs, preds

    def inference(self, input, target):
        ########
        # TODO #
        ########
        # 在這裡實施 Beam Search
        # 此函式的 batch size = 1  
        # input  = [batch size, input len, vocab size]
        # target = [batch size, target len, vocab size]
        batch_size = input.shape[0]
        input_len = input.shape[1]        # 取得最大字數
        vocab_size = self.decoder.cn_vocab_size

        # 準備一個儲存空間來儲存輸出
        outputs = torch.zeros(batch_size, input_len, vocab_size).to(self.device)
        # 將輸入放入 Encoder
        encoder_outputs, hidden = self.encoder(input)
        # Encoder 最後的隱藏層(hidden state) 用來初始化 Decoder
        # encoder_outputs 主要是使用在 Attention
        # 因為 Encoder 是雙向的RNN，所以需要將同一層兩個方向的 hidden state 接在一起
        # hidden =  [num_layers * directions, batch size  , hid dim]  --> [num_layers, directions, batch size  , hid dim]
        hidden = hidden.view(self.encoder.n_layers, 2, batch_size, -1)
        hidden = torch.cat((hidden[:, -2, :, :], hidden[:, -1, :, :]), dim=2)
        # 取的 <BOS> token
        input = target[:, 0]
        preds = []
        for t in range(1, input_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            # 將預測結果存起來
            outputs[:, t] = output
            # 取出機率最大的單詞
            top1 = output.argmax(1)
            input = top1
            preds.append(top1.unsqueeze(1))
        preds = torch.cat(preds, 1)
        return outputs, preds
