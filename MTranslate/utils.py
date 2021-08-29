# -*- coding: utf-8 -*-
# 此程序存放了一些基本操作的函数
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from model import *

# 储存模型，step表示训练多少词存储一次模型
def save_model(model, optimizer, store_model_path, step):
    torch.save(model.state_dict(), f'{store_model_path}/model_{step}.ckpt')
    return

# 载入模型
def load_model(model, load_model_path):
    print(f'Load model from {load_model_path}')
    model.load_state_dict(torch.load(f'{load_model_path}.ckpt'))
    return model

# 构建模型
def build_model(config, en_vocab_size, cn_vocab_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(en_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout)
    decoder = Decoder(cn_vocab_size, config.emb_dim, config.hid_dim, config.n_layers, config.dropout, config.attention)
    model = Seq2Seq(encoder, decoder, device)
    print(model)
    # 构建 optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print(optimizer)
    if config.load_model_path:
      model = load_model(model, config.load_model_path)
    model = model.to(device)
      
    return model, optimizer

# 将一连串的数字还原回句子
def tokens2sentence(outputs, int2word):
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)

    return sentences

# 计算BLEU score
def computebleu(sentences, targets):
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score

# 迭代dataloader
# dataloader本质是一个可迭代对象，使用iter()访问，不能使用next()访问；
# 用iter(dataloader)返回的是一个迭代器，然后可以使用next访问；
# 一般我们实现一个datasets对象，传入到dataloader中；然后内部使用yeild返回每一次batch的数据；
def infinite_iter(data_loader):
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)
            
# 绘图函数
# 图形化训练过程
def plt_line(data, x, y, title):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.show()   
    
