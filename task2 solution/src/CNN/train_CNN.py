# -*- coding:utf-8 -*-
"""
Author :Xubing Ye
Number :1953348
Date   :2022/03/17
File   :train_CNN.py
"""

import torch
import torch.nn.functional as F


def train_CNN(train_itr, val_itr, model, args):
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # 优化参数，指定learning rate

    # args
    step = 0
    best_acc = 0
    save_path = 'model/CNN_model.pt'

    model.train()
    print('================== Start Training ==================')
    for epoch in range(1, args.epochs + 1):

        # 每个 epoch
        print('Epoch: {} - {}'.format(epoch, args.epochs))
        for batch in train_itr:
            text, label = batch.Phrase, batch.Sentiment
            text.t_()
            text = text.to(args.device)
            label = label.to(args.device)

            optimizer.zero_grad()  # 梯度置零
            predict = model(text)
            loss = F.cross_entropy(predict, label)  # 交叉熵损失函数
            loss.backward()
            optimizer.step()

            step += 1

            # 每1k轮输出loss与acc
            if step % 1000 == 0:
                predict_y = torch.max(predict, 1)[1].view(label.size())
                accuracy = (predict_y.data == label.data).sum() / batch.batch_size
                print('\rBatch[{}] - loss: {:.6f} - acc: {:.4f}'.format(step, loss.data.item(), accuracy))

            # 每1k轮比较 保存模型
            if step % 1000 == 0:
                val_acc = eval(val_itr, model, args)
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model, save_path)
            elif step % 500 == 0:
                torch.save(model, save_path)
            # 由于准确率在60-70 无需设置截断条件以防止过拟合


def eval(val_itr, model, args):
    model.eval()
    val_loss = 0
    val_correct = 0
    for batch in val_itr:
        text, label = batch.Phrase, batch.Sentiment
        text.t_()
        text = text.to(args.device)
        label = label.to(args.device)

        predict = model(text)  # 预测值
        loss = F.cross_entropy(predict, label)  # 交叉熵损失函数
        val_loss += loss.data.item()

        predict_y = torch.max(predict, 1)[1].view(label.size())
        val_correct += (predict_y.data == label.data).sum()

    data_size = len(val_itr.dataset)
    val_loss /= data_size
    val_acc = val_correct / data_size
    print('Evaluation - loss: {:.6f} - acc: {:.4f}\n'.format(val_loss, val_acc))

    return val_acc
