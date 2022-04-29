# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2022/04/29
File   :main.py
"""
from train import train
from embedding import cat_poem

if __name__ == "__main__":
    # 训练模型 返回模型
    models, train_loss_records = train()

    # 使用LSTM进行测试
    model = models[0]

    # 生成诗句，修改random即可改变是否定长
    poem = cat_poem(model.generate_poem(14, 4, random=False))
    for sent in poem:
        print(sent)

    # 生成藏头诗，修改random即可改变是否定长
    poem = cat_poem(model.generate_head("藏头诗", max_len=14, random=False))
    for sent in poem:
        print(sent)
