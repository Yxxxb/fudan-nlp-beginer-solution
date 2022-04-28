# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2022/04/15
File   :main.py
"""
from Utils import data_process, embedding, get_batch_data, train
from model import Named_Entity_Recognition
import torch

if __name__ == '__main__':
    # 读数据集（已划分）
    with open('CoNLL-2003/train.txt', 'r') as f:
        temp = f.readlines()
    data = temp[2:]
    train_zip = data_process(data)
    with open('CoNLL-2003/test.txt', 'r') as f:
        temp = f.readlines()
    data = temp[2:]
    test_zip = data_process(data)

    # 读取glove
    with open('glove/glove.6B.50d.txt', 'rb') as f:  # for glove embedding
        lines = f.readlines()
    trained_dict = dict()
    n = len(lines)
    for i in range(n):
        line = lines[i].split()
        trained_dict[line[0].decode("utf-8").upper()] = [float(line[j]) for j in range(1, 51)]

    # Embedding层 在Utils内实现
    embedding = embedding(train_zip, test_zip, trained_dict=trained_dict)
    embedding.get_words()
    embedding.get_id()

    # 训练参数设置
    iter_times = 100
    learning_rate = 0.001
    batch_size = 100
    len_feature = 50
    len_hidden = 50

    # 开始训练
    train_data = get_batch_data(embedding.train_x_matrix, embedding.train_y_matrix, batch_size)
    test_data = get_batch_data(embedding.test_x_matrix, embedding.test_y_matrix, batch_size)
    model = Named_Entity_Recognition(len_feature, embedding.len_words, len_hidden,
                                     embedding.len_tag, 0, 1, 2,
                                     weight=torch.tensor(embedding.embedding, dtype=torch.float))
    train_loss, test_loss, train_record, test_record = train(model, train_data,
                                                             test_data, learning_rate,
                                                             iter_times, batch_size)
