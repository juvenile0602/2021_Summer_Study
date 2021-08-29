# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.utils.data as data

# 导入自己编写的模块
# 设定工作目录
os.chdir(r'D:\Study\SummerStudy\MTranslate') #修改成你自己的工作路径
# 读取自己写的python程序中的函数
from Config import configurations
from train_process import *
from test import *
from dataset import EN2CNDataset
from model import *
from utils import *



def testf(model, dataloader, loss_function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []
    for sources, targets in dataloader:
        sources, targets = sources.to(device), targets.to(device)
        batch_size = sources.size(0)
        outputs, preds = model.inference(sources, targets)
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)

        loss = loss_function(outputs, targets)
        loss_sum += loss.item()
        
        # 将预测结果转为文字
        targets = targets.view(sources.size(0), -1)
        preds = tokens2sentence(preds, dataloader.dataset.int2word_cn)
        sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
        targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
        for source, pred, target in zip(sources, preds, targets):
            result.append((source, pred, target))
        # 計算 Bleu Score
        bleu_score += computebleu(preds, targets)

        n += batch_size

    return loss_sum / len(dataloader), bleu_score / n, result


if __name__ == '__main__':
     # 在执行Test 之前，请先到config中设定所要载入的模型位置
    config = configurations()
    config.set_load_model_path("./Model/model_11400")
    print('config:\n', vars(config))
    # 准备测试数据
    test_dataset = EN2CNDataset(
        config.data_path, config.max_output_len, 'testing')
    test_loader = data.DataLoader(test_dataset, batch_size=1)
    # 构建模型
    model, optimizer = build_model(
        config, test_dataset.en_vocab_size, test_dataset.cn_vocab_size)
    print("Finish build model")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    # 測試模型
    test_loss, bleu_score, result = testf(model, test_loader, loss_function)
    # 儲存結果
    with open(f'./OutPut/test_output.txt', 'w') as f:
        for line in result:
            print(line, file=f)


    print(f'test loss: {test_loss}, bleu_score: {bleu_score}')