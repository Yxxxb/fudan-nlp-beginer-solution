# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2022/04/29
File   :train.py
"""
from embedding import get_batch, Embedding
from torch import optim
from LSTM_GRU import Language
import torch.nn.functional as F


def train():
    print("start training...")
    with open('poetryFromTang.txt', 'rb') as f:
        temp = f.readlines()

    # batch只需要设置为1即可
    a = Embedding(temp)
    a.data_process()
    train = get_batch(a.matrix, 1)
    learning_rate = 0.004
    iter_times = 10

    # 要求两种模型 LSTM GRU
    strategies = ['lstm', 'gru']
    train_loss_records = list()
    models = list()
    for i in range(2):
        print("training model : ", strategies[i])
        model = Language(50, len(a.word_dict), 50, a.tag_dict, a.word_dict, strategy=strategies[i])
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fun = F.cross_entropy
        loss_list = list()
        model = model.cuda()
        for iteration in range(iter_times):
            total_loss = 0
            model.train()
            for i, batch in enumerate(train):
                x = batch.cuda()
                x, y = x[:, :-1], x[:, 1:]
                pred = model(x).transpose(1, 2)
                optimizer.zero_grad()
                loss = loss_fun(pred, y)
                total_loss += loss.item() / (x.shape[1] - 1)
                loss.backward()
                optimizer.step()
            loss_list.append(total_loss / len(train))
            print("========== Iteration", iteration + 1, "==========")
            print("Train loss:", total_loss / len(train))
        train_loss_records.append(loss_list)
        models.append(model)

    return models, train_loss_records, iter_times
