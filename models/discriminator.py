# -*- coding: UTF-8 -*-
'''
@Project ：LeakGan
@File    ：discriminator.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''

import math
import torch
import torch.nn as nn
import config as cfg

class CNNDiscriminator(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 vocab_size:int,
                 filter_sizes:list,
                 num_filters:list,
                 padding_idx:int,
                 dropout:float,
                 **kwargs):
        super(CNNDiscriminator, self).__init__(**kwargs)
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.feature_dim = sum(num_filters)
        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embed_dim,
                                       padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n,
                      kernel_size=(f, embed_dim))
            for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(in_features=self.feature_dim,
                                 out_features=self.feature_dim)
        self.feature2out = nn.Linear(in_features=self.feature_dim,
                                     out_features=2 if cfg.loss_mode is 'CrossEntropy' else 1)
        self.dropout = nn.Dropout(p=dropout)
        self.init_params()

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [torch.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [torch.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * torch.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def init_params(self):
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] is 'bias':
                    torch.nn.init.zeros_(param)
                else:
                    stddev = 1 / math.sqrt(param.shape[0])
                    if cfg.dis_init == 'uniform':
                        torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                    elif cfg.dis_init == 'normal':
                        torch.nn.init.normal_(param, std=stddev)
