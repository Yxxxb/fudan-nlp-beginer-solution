# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :model_CNN.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class model_CNN(nn.Module):
    def __init__(self, args):
        super(model_CNN, self).__init__()
        self.args = args

        # args
        in_channels = 1  # num of input channels
        label_num = args.label_num  # category

        # Kernel
        kernel_num = args.kernel_num
        kernel_size = args.kernel_size

        # Embedding
        weight = args.weight
        self.embedding_dim = weight.size(1)
        num_embeddings = weight.size(0)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

        # Convolution
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, kernel_num, (ks, self.embedding_dim)) for ks in kernel_size])

        # Drop
        self.dropout = nn.Dropout(args.dropout)

        # Full connection
        self.fullconnection = nn.Linear(len(kernel_size) * kernel_num, label_num)

    def forward(self, x):
        # embedding
        x = self.embedding(x)
        # add dimension
        x = x.view(x.size(0), 1, x.size(1), self.embedding_dim)
        # Convolution
        x = [F.relu(conv(x)) for conv in self.convs]
        # Max pooling
        x = [F.max_pool2d(input=x_item, kernel_size=(x_item.size(2), x_item.size(3))) for x_item in x]
        x = [x_item.view(x_item.size(0), -1) for x_item in x]
        x = torch.cat(x, 1)
        # dropout
        x = self.dropout(x)
        # Full connection
        return self.fullconnection(x)

