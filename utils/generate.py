# -*- coding: UTF-8 -*-
'''
@Project ：LeakGan
@File    ：_utils.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import os
import json
import torch
from torch import nn
import numpy as np
import nltk
import config as cfg

class Generator:
    def __init__(self,
                 file_path: str,
                 vocab_path: str,
                 vocab_size: int,
                 max_seq_len: int,
                 batch_size: int,
                 train_ratio: float,
                 dataset: str,
                 **kwargs):
        self.file_path = os.path.join(file_path, dataset + '.txt')
        self.vocab_path = vocab_path
        self.dataset = dataset
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        with open(self.file_path, 'rb') as f:
            self.sentences = f.readlines()
        self.vocab = self.get_vocab()

    def get_vocab(self):
        if os.path.exists(self.vocab_path + '\\{}.json'.format(self.dataset)):
            f = open(self.vocab_path + '\\{}.json'.format(self.dataset), 'r')
            dict = json.load(f)
            return dict['vocab']

        total_tokens = []
        for sentence in self.sentences:
            sentence = sentence.decode('UTF-8').strip()
            tokens = nltk.word_tokenize(sentence, language='english')
            total_tokens.extend(tokens)

        total_vocab = list(set(total_tokens))
        total_tokens_num = []
        for vocab in total_vocab:
            total_tokens_num.append(total_tokens.count(vocab))

        index = sorted(np.arange(len(total_vocab)), key=lambda i: total_tokens_num[i], reverse=True)
        sorted_total_vocab = ['', '/s', 'UNK'] + list(np.array(total_vocab)[index])
        sliced_vocab = list(np.array(sorted_total_vocab)[: self.vocab_size])

        dict = {'vocab': sliced_vocab}
        with open(self.vocab_path + '\\{}.json'.format(self.dataset), 'w') as f:
            json.dump(dict, f, indent=2)

        return sliced_vocab

    def get_train_len(self):
        if not int(len(self.sentences)*self.train_ratio) % self.batch_size:
            return int(len(self.sentences)*self.train_ratio)//self.batch_size
        else:
            return int(len(self.sentences)*self.train_ratio)//self.batch_size + 1

    def get_test_len(self):
        if not (len(self.sentences)-int(len(self.sentences)*self.train_ratio)) % self.batch_size:
            return (len(self.sentences)-int(len(self.sentences)*self.train_ratio))//self.batch_size
        else:
            return (len(self.sentences)-int(len(self.sentences)*self.train_ratio))//self.batch_size + 1

    def generate(self, training=True):
        while True:
            if training:
                train_sentences = np.array(self.sentences)[:int(len(self.sentences)*self.train_ratio)]
                np.random.shuffle(train_sentences)
                sources = []
                for i, sentence in enumerate(train_sentences):
                    sentence = sentence.decode('UTF-8').strip()
                    tokens = nltk.word_tokenize(sentence, language='english')
                    source = []
                    for token in tokens:
                        try:
                            source.append(self.vocab.index(token))
                        except ValueError:
                            source.append(self.vocab.index('UNK'))
                    source = source[:self.max_seq_len] if np.greater_equal(len(source), self.max_seq_len) \
                        else source+[self.vocab.index(''), ]*(self.max_seq_len-len(source))
                    sources.append(source)

                    if np.logical_or(np.equal(len(sources), self.batch_size),
                                     np.equal(i, len(train_sentences)-1)):

                        anno_sources = np.array(sources.copy())
                        sources.clear()

                        yield anno_sources

            else:
                test_sentences = np.array(self.sentences)[int(len(self.sentences) * self.train_ratio):]
                np.random.shuffle(test_sentences)
                sources = []
                for i, sentence in enumerate(test_sentences):
                    sentence = sentence.decode('UTF-8').strip()
                    tokens = nltk.word_tokenize(sentence, language='english')
                    for token in tokens:
                        source = []
                        try:
                            source.append(self.vocab.index(token))
                        except ValueError:
                            source.append(self.vocab.index('UNK'))
                    source = source[:self.max_seq_len] if np.greater_equal(len(source), self.max_seq_len) \
                        else source + [self.vocab.index(''), ] * (self.max_seq_len - len(source))
                    sources.append(source)

                    if np.logical_or(np.equal(len(sources), self.batch_size),
                                     np.equal(i, len(test_sentences) - 1)):
                        anno_sources = np.array(sources.copy())
                        sources.clear()

                        yield anno_sources
