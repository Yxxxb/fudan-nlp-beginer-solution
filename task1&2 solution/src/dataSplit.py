# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :dataSplit.py
"""
import csv
import pandas as pd

data = pd.read_csv('../data/train.tsv', delimiter='\t')

# 数据分割 train:val:test = 6:2:2
train_data = data.head(int(len(data) * 0.6))
val_data = data.loc[int(len(data) * 0.6): int(len(data) * 0.8)]
test_data = data.loc[int(len(data) * 0.8): len(data)]

# 保存分割后的训练、预测、交叉验证数据
with open('../data/train_data.tsv', 'w') as fw1:
    fw1.write(train_data.to_csv(sep='\t', index=False))
with open('../data/val_data.tsv', 'w') as fw2:
    fw2.write(val_data.to_csv(sep='\t', index=False))
with open('../data/test_data.tsv', 'w') as fw3:
    fw3.write(test_data.to_csv(sep='\t', index=False))
