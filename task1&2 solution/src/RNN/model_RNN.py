# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :model_RNN.py
"""

import torch
import torch.nn as nn


class model_RNN(nn.Module):
    def __init__(self, args):
        super(model_RNN, self).__init__()

        # args
        label_num = args.label_num

        # Embedding
        weight = args.weight
        embedding_dim = weight.size(1)
        num_embeddings = weight.size(0)

        # type
        self.rnn_type = args.rnn_type
        # dimension
        self.hidden_size = args.hidden_size
        # num of layer
        self.num_layers = args.num_layers
        # bidirectional
        self.bidirectional = args.bidirectional

        # Embedding
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

        # LSTM
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              batch_first=True,
                              bidirectional=self.bidirectional)
        elif self.rnn_type == 'lstm':
            self.lstm = nn.LSTM(input_size=embedding_dim,
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers,
                                batch_first=True,
                                bidirectional=self.bidirectional)
        # Full connection
        if self.bidirectional:
            self.fullconnection = nn.Linear(self.hidden_size * 2, label_num)
        else:
            self.fullconnection = nn.Linear(self.hidden_size, label_num)

    def forward(self, x):
        # Embedding
        x = self.embedding(x)

        if self.rnn_type == 'rnn':
            if self.bidirectional:
                h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size)
            else:
                h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
            out, hn = self.rnn(x, h0)
        elif self.rnn_type == 'lstm':
            if self.bidirectional:
                h0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size)
                c0 = torch.randn(self.num_layers * 2, x.size(0), self.hidden_size)
            else:
                h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
            out, (hn, cn) = self.lstm(x, (h0, c0))

        # Full connection
        return self.fullconnection(out[:, -1, :])
