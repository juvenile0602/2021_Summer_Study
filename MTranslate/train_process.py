# -*- coding: utf-8 -*-
# 此程序执行模型的训练
# 导入所需模组
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
from torchvision import datasets, transforms
import numpy as np

# 设定工作目录
os.chdir(r'D:\Study\SummerStudy\MTranslate') #修改成你自己的工作路径
# 读取自己写的python程序中的函数
from Config import configurations
from schedule_sampling import schedule_sampling
from dataset import *
from model import *
from utils import *
from test_process import *


def train(model, optimizer, train_iter, loss_function, total_steps, summary_steps, train_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    model.zero_grad()
    losses = []
    loss_sum = 0.0
    for step in range(summary_steps):
        # 进入每一轮训练中
        # sources, targets分别为英文句子的index序列，中文句子的index序列，shape均为[60, 50]
        sources, targets = next(train_iter)
        sources, targets = sources.to(device), targets.to(device)
        outputs, preds = model(sources, targets, schedule_sampling(total_steps)) 
        # outputs的shape为[60, 50, 3805]，3805为中文词表size - 1
        # preds的shape为[60, 49]
        # targets 的第一個 token 是 <BOS> 所以忽略
        outputs = outputs[:, 1:].reshape(-1, outputs.size(2))
        targets = targets[:, 1:].reshape(-1)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        loss_sum += loss.item()
        if (step + 1) % 5 == 0:
            loss_sum = loss_sum / 5
            print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}      ".format(
                total_steps + step + 1, loss_sum, np.exp(loss_sum)), end=" ")
            losses.append(loss_sum)
            loss_sum = 0.0

    return model, optimizer, losses

if __name__ == '__main__':
    # torch.set_num_threads(1)
    # 设置参数
    config = configurations() 
    print('config:\n', vars(config))
    # 准备训练数据
    train_dataset = EN2CNDataset(
        config.data_path, config.max_output_len, 'training')
    train_loader = data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True)
    # 检查一下训练数据的形式
    # for i,data_i in enumerate(train_loader, 0):
    #     x,y = data_i
    #     print(x+y)
    #     if i==2:
    #         break
    # x = [batch_size,max_output_len]EN2CNDataset, y的size相同
    # 每一条数据形如[1,20,824,...,0,0,0]
    
    train_iter = infinite_iter(train_loader)
    # 准备检验数据
    val_dataset = EN2CNDataset(
        config.data_path, config.max_output_len, 'validation')
    val_loader = data.DataLoader(val_dataset, batch_size=10)
    # 构建模型
    model, optimizer = build_model(
        config, train_dataset.en_vocab_size, train_dataset.cn_vocab_size)
    # 损失函数
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    
    train_losses, val_losses, bleu_scores = [], [], []
    total_steps = 0
    while (total_steps < config.num_steps):
        # 训练模型
        # 每train summary_steps次数后，进行一次小结
        model, optimizer, loss = train(
            model, optimizer, train_iter, loss_function, total_steps, config.summary_steps, train_dataset)
        train_losses += loss
        # 检验模型
        val_loss, bleu_score, result = testf(model, val_loader, loss_function)
        val_losses.append(val_loss)
        bleu_scores.append(bleu_score)
        
        total_steps += config.summary_steps
        print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, bleu score: {:.3f}       ".format(
            total_steps, val_loss, np.exp(val_loss), bleu_score))
        
        # 儲存模型和結果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            save_model(model, optimizer, config.store_model_path, total_steps)
            with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
                for line in result:
                    print(line, file=f)



    # 图形化训练过程
    # 以图表呈现训练的loss变化趋势
    plt_line(train_losses, '次數', 'loss', 'train loss')
    # 以图表呈现测试的loss变化趋势
    plt_line(val_losses, '次數', 'loss', 'validation loss')
    # BLEU score
    plt_line(bleu_scores, '次數', 'BLEU score', 'BLEU score')