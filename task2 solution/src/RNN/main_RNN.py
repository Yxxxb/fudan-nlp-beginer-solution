# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :main_RNN.py
"""

import dataProcessing
import model_RNN
import train_RNN
import argparse
import torch


if __name__ == '__main__':

    # 设置argparse参数 详见文档
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--rnn_type', type=str, default='rnn')
    parser.add_argument('--hidden_size', type=int, default=20)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidirectional', type=bool, default=False)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    # 读取分割好的数据
    print('Load Data')
    train_data, val_data, test_data = 'train_data.tsv', 'val_data.tsv', 'test_data.tsv'
    train_itr, val_itr, test_itr, weight = dataProcessing.dataProcessing(train_data, val_data, test_data, args.batch_size)

    args.weight = weight
    args.label_num = 5
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    textRNN = model_RNN.TextRNN(args)
    textRNN.to(args.device)

    # 同train 设置读取RNN和LSTM的不同路径
    if args.test:
        save_path = ''
        if args.rnn_type == 'rnn':
            if args.bidirectional:
                save_path = 'model/rnn/textRNN_rnn2_model.pt'
            else:
                save_path = 'model/rnn/textRnn_rnn1_model.pt'
        elif args.rnn_type == 'lstm':
            if args.bidirectional:
                save_path = 'model/lstm/textRNN_lstm2_model.pt'
            else:
                save_path = 'model/lstm/textRnn_lstm1_model.pt'
        print('Load Model & test')
        textRNN = torch.load(save_path)
        train_RNN.eval(test_itr, textRNN, args)
    else:
        train_RNN.train(train_itr, val_itr, textRNN, args)
