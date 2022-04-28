# -*- coding:utf-8 -*-
"""
Author :Yxxxb & Xubing Ye
Number :1953348
Date   :2022/04/15
File   :Utils.py
"""
from torch.utils.data import Dataset, DataLoader
from torch import optim
import torch


# embedding 供train调用
class embedding():
    def __init__(self, train_zip, test_zip, trained_dict=None):
        if trained_dict is None:
            trained_dict = dict()
        self.dict_words = dict()
        self.trained_dict = trained_dict
        train_zip.sort(key=lambda x: len(x[0]))
        test_zip.sort(key=lambda x: len(x[0]))
        self.train_x, self.train_y = zip(*train_zip)
        self.test_x, self.test_y = zip(*test_zip)
        self.train_x_matrix = list()
        self.test_x_matrix = list()
        self.train_y_matrix = list()
        self.test_y_matrix = list()
        self.len_words = 1
        self.len_tag = 3
        self.longest = 0
        self.embedding = list()
        # C类标签
        self.tag_dict = {'<pad>': 0, '<begin>': 1, '<end>': 2}

    def get_words(self):
        self.embedding.append([0] * 50)
        for term in self.train_x:
            for word in term:  # Process every word
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:
                        self.embedding.append([0] * 50)
        for term in self.test_x:
            for word in term:
                if word not in self.dict_words:
                    self.dict_words[word] = len(self.dict_words) + 1
                    if word in self.trained_dict:
                        self.embedding.append(self.trained_dict[word])
                    else:
                        self.embedding.append([0] * 50)
        for tags in self.train_y:
            for tag in tags:
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)
        for tags in self.test_y:
            for tag in tags:
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = len(self.tag_dict)
        self.len_tag = len(self.tag_dict)
        self.len_words = len(self.dict_words) + 1

    def get_id(self):
        for term in self.train_x:
            item = [self.dict_words[word] for word in term]
            self.longest = max(self.longest, len(item))
            self.train_x_matrix.append(item)
        for term in self.test_x:
            item = [self.dict_words[word] for word in term]
            self.longest = max(self.longest, len(item))
            self.test_x_matrix.append(item)
        for tags in self.train_y:
            item = [self.tag_dict[tag] for tag in tags]
            self.train_y_matrix.append(item)
        for tags in self.test_y:
            item = [self.tag_dict[tag] for tag in tags]
            self.test_y_matrix.append(item)


# 训练函数（调用model）
def train(model, train_data, test_data, learning_rate, iterations, batch_size):
    # 设置Adam优化器等参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    test_loss = []
    train_record = []
    test_record = []

    # 开始训练
    print('start taining...')
    for iteration in range(iterations):
        model.train()
        for i, batch in enumerate(train_data):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            mask = (y != 0).cuda()
            loss = model(x, y, mask).cuda()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        train_acc = list()
        test_acc = list()
        train_loss = 0
        test_loss = 0
        for i, batch in enumerate(train_data):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            mask = (y != 0).cuda()
            loss = model(x, y, mask).cuda()
            train_loss += loss.item() / batch_size / y.shape[1]
            pred = model.predict(x, mask)
            acc = (pred == y).float()
            len_batch, len_seq = acc.shape
            points = torch.ones((1, len_batch)).cuda()
            for j in range(len_seq):
                points *= acc[:, j]
            train_acc.append(points.mean())

            print("==============Epoch", i + 1, "==============")
            print("Epoch Train loss:", train_loss)
            print("Epoch Train accuracy:", points.mean())

        for i, batch in enumerate(test_data):
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            mask = (y != 0).cuda()
            loss = model(x, y, mask).cuda()
            test_loss += loss.item() / batch_size / y.shape[1]
            pred = model.predict(x, mask)
            acc = (pred == y).float()
            len_batch, len_seq = acc.shape
            points = torch.ones((1, len_batch)).cuda()
            for j in range(len_seq):
                points *= acc[:, j]
            test_acc.append(points.mean())

        trains_acc = sum(train_acc) / len(train_acc)
        tests_acc = sum(test_acc) / len(test_acc)

        train_loss.append(train_loss / len(train_data))
        test_loss.append(test_loss / len(test_data))
        train_record.append(trains_acc)
        test_record.append(tests_acc)
        print("============== Iteration", iteration + 1, "==============")
        print("Train loss:", train_loss / len(train_data))
        print("Test loss:", test_loss / len(test_data))
        print("Train accuracy:", trains_acc)
        print("Test accuracy:", tests_acc)

    return train_loss, test_loss, train_record, test_record


# 数据预处理成zip
def data_process(data):
    sentences = list()
    tags = list()
    sentence = list()
    tag = list()
    for line in data:
        if line == '\n':
            if sentence:
                sentences.append(sentence)
                tags.append(tag)
                sentence = list()
                tag = list()
        else:
            elements = line.split()
            if elements[0] == '-DOCSTART-':
                continue
            sentence.append(elements[0].upper())
            tag.append(elements[-1])
    if sentence:
        sentences.append(sentence)
        tags.append(tag)

    return list(zip(sentences, tags))


class cls(Dataset):
    def __init__(self, sentence, tag):
        self.sentence = sentence
        self.tag = tag

    def __getitem__(self, item):
        return self.sentence[item], self.tag[item]

    def __len__(self):
        return len(self.tag)


def get_batch_data(sentence, tag, batch_size):
    dataset = cls(sentence, tag)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader
