# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :dataProcessing.py
"""
from torchtext.legacy import data
from torchtext.vocab import Vectors
import torch


def dataProcessing(train_data, val_data, test_data, batch_size):
    # 设置padding 用于维护短词句
    tokenize = lambda x: x.split()
    TEXT = data.Field(sequential=True, tokenize=tokenize, stop_words='english', fix_length=60)
    LABEL = data.Field(sequential=False, use_vocab=False)

    # 载入数据
    fields = [('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)]
    train_data, val_data, test_data = data.TabularDataset.splits(
        path='../data',
        skip_header=True,
        train=train_data,
        validation=val_data,
        test=test_data,
        format='tsv',
        fields=fields
    )

    # 设置迭代器
    train_itr, val_itr = data.Iterator.splits(
        (train_data, val_data),
        batch_sizes=(batch_size, batch_size),
        sort_key=lambda x: len(x.Phrase),
        device=-1
    )
    test_itr = data.Iterator(
        test_data,
        batch_size=batch_size,
        sort=False,
        device=-1
    )

    # 加载glove
    vectors = Vectors(name='../glove/glove.6B.200d.txt')
    TEXT.build_vocab(train_data, val_data, test_data, vectors=vectors)
    LABEL.build_vocab(train_data, val_data, test_data)
    weights = TEXT.vocab.vectors

    return train_itr, val_itr, test_itr, weights
