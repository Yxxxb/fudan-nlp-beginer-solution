# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :main_CNN.py
"""

import dataProcessing
import model_CNN
import train_CNN
import argparse
import torch


if __name__ == '__main__':

    # 设置argparse参数 详见文档
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--kernel_size', type=str, default='2,3,4')
    parser.add_argument('--kernel_num', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    # 读取分割好的数据
    print('Load Data')
    train_data, val_data, test_data = 'train_data.tsv', 'val_data.tsv', 'test_data.tsv'
    train_itr, val_itr, test_itr, weight = dataProcessing.dataProcessing(train_data, val_data, test_data, args.batch_size)

    args.weight = weight
    args.label_num = 5
    args.kernel_size = [int(k) for k in args.kernel_size.split(',')]
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    textCNN = model_CNN.TextCNN(args)
    textCNN.to(args.device)

    if args.test:
        print('Load Model & test')
        textCNN = torch.load('model/textCNN_model.pt')
        train_CNN.eval(test_itr, textCNN, args)
    else:
        train_CNN.train(train_itr, val_itr, textCNN, args)
