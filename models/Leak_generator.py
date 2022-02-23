# -*- coding: UTF-8 -*-
'''
@Project ：LeakGan
@File    ：leak_generator.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from models.Leak_discriminator import LeakGAN_D


class LeakGAN_G(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 vocab_size: int,
                 max_seq_len: int,
                 goal_size: int,
                 goal_out_size: int,
                 step_size: int,
                 padding_idx: int):
        super(LeakGAN_G, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.goal_size = goal_size
        self.goal_out_size = goal_out_size  # equals to total_num_filters
        self.step_size = step_size
        self.temperature = 1.5

        self.embeddings = nn.Embedding(num_embeddings=vocab_size,
                                       embedding_dim=embedding_dim,
                                       padding_idx=padding_idx)
        self.worker = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)
        self.manager = nn.LSTM(input_size=goal_out_size, hidden_size=hidden_dim)

        self.work2goal = nn.Linear(in_features=hidden_dim,
                                   out_features=vocab_size * goal_size)
        self.mana2goal = nn.Linear(in_features=hidden_dim,
                                   out_features=goal_out_size)
        self.goal2goal = nn.Linear(in_features=goal_out_size,
                                   out_features=goal_size,
                                   bias=False)

        self.goal_init = nn.Parameter(torch.rand((cfg.batch_size, goal_out_size)), requires_grad=False)

        self.init_params()

    def forward(self, idx, inp, work_hidden, mana_hidden, feature, real_goal, no_log=False, train=False):
        """
        Embeds input and sample on token at a time (seq_len = 1)
        :param idx: index of current token in sentence
        :param inp: [batch_size]
        :param work_hidden: 1 * batch_size * hidden_dim
        :param mana_hidden: 1 * batch_size * hidden_dim
        :param feature: 1 * batch_size * total_num_filters, feature of current sentence
        :param real_goal: batch_size * goal_out_size, real_goal in LeakGAN source code
        :param no_log: no log operation
        :param train: if train
        :return: out, cur_goal, work_hidden, mana_hidden
            - out: batch_size * vocab_size
            - cur_goal: batch_size * 1 * goal_out_size
        """
        emb = self.embeddings(inp).unsqueeze(0)  # 1 * batch_size * embed_dim

        # Manager
        mana_out, mana_hidden = self.manager(feature, mana_hidden)  # mana_out: 1 * batch_size * hidden_dim
        mana_out = self.mana2goal(mana_out.permute([1, 0, 2]))  # batch_size * 1 * goal_out_size
        cur_goal = F.normalize(mana_out, dim=-1)
        _real_goal = self.goal2goal(real_goal)  # batch_size * goal_size
        _real_goal = F.normalize(_real_goal, p=2, dim=-1).unsqueeze(-1)  # batch_size * goal_size * 1

        # Worker
        work_out, work_hidden = self.worker(emb, work_hidden)  # work_out: 1 * batch_size * hidden_dim
        work_out = self.work2goal(work_out).view(-1, self.vocab_size,
                                                 self.goal_size)  # batch_size * vocab_size * goal_size

        # Sample token
        out = torch.matmul(work_out, _real_goal).squeeze(-1)  # batch_size * vocab_size

        # Temperature control
        if idx > 1:
            if train:
                temperature = 1.
            else:
                temperature = self.temperature
        else:
            temperature = self.temperature

        out = temperature * out

        if no_log:
            out = torch.softmax(out, dim=-1)
        else:
            out = torch.log_softmax(out, dim=-1)

        return out, cur_goal, work_hidden, mana_hidden

    def sample(self, num_samples, batch_size, discriminator, start_letter=cfg.start_letter, train=False):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return: samples: batch_size * max_seq_len
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()  # larger than num_samples
        fake_sentences = torch.zeros((batch_size, self.max_seq_len))

        for b in range(num_batch):
            leak_sample, _, _, _ = self.forward_leakgan(fake_sentences, discriminator, if_sample=True,
                                                        no_log=False, start_letter=start_letter, train=False)

            assert leak_sample.shape == (batch_size, self.max_seq_len)
            samples[b * batch_size:(b + 1) * batch_size, :] = leak_sample

        samples = samples[:num_samples, :]

        return samples  # cut to num_samples

    def pretrain_loss(self, target, discriminator, start_letter=cfg.start_letter):
        """
        Returns the pretrain_generator Loss for predicting target sequence.
        Inputs: target, discriminator, start_letter
            - target: batch_size * seq_len
        """
        batch_size, seq_len = target.size()
        _, feature_array, goal_array, leak_out_array = self.forward_leakgan(target, discriminator,
                                                                            if_sample=False, no_log=False,
                                                                            start_letter=start_letter)

        # Manager loss
        mana_cos_loss = self.manager_cos_loss(batch_size, feature_array,
                                              goal_array)  # batch_size * (seq_len / step_size)
        manager_loss = -torch.sum(mana_cos_loss) / (batch_size * (seq_len // self.step_size))

        # Worker loss
        work_nll_loss = self.worker_nll_loss(target, leak_out_array)  # batch_size * seq_len
        work_loss = torch.sum(work_nll_loss) / (batch_size * seq_len)

        return manager_loss, work_loss

    def adversarial_loss(self, target, rewards, discriminator, start_letter=cfg.start_letter):
        """
        Returns a pseudo-loss that gives corresponding policy gradients (on calling .backward()).
        Inspired by the example in http://karpathy.github.io/2016/05/31/rl/
        Inputs: target, rewards, discriminator, start_letter
            - target: batch_size * seq_len
            - rewards: batch_size * seq_len (discriminator rewards for each token)
        """
        batch_size, seq_len = target.size()
        _, feature_array, goal_array, leak_out_array = self.forward_leakgan(target, discriminator,
                                                                            if_sample=False, no_log=False,
                                                                            start_letter=start_letter, train=True)
        # Manager Loss
        mana_cos_loss = self.manager_cos_loss(batch_size, feature_array,
                                              goal_array)  # batch_size * (seq_len / step_size)
        mana_loss = -torch.sum(rewards * mana_cos_loss) / (batch_size * (seq_len // self.step_size))

        # Worker Loss
        work_nll_loss = self.worker_nll_loss(target, leak_out_array)  # batch_size * seq_len
        work_cos_reward = self.worker_cos_reward(feature_array, goal_array)  # batch_size * seq_len
        work_loss = -torch.sum(work_nll_loss * work_cos_reward) / (batch_size * seq_len)

        return mana_loss, work_loss

    def manager_cos_loss(self, batch_size, feature_array, goal_array):
        """
        Get manager cosine distance loss
        :return cos_loss: batch_size * (seq_len / step_size)
        """
        sub_feature = torch.zeros(batch_size, self.max_seq_len // self.step_size, self.goal_out_size)
        real_goal = torch.zeros(batch_size, self.max_seq_len // self.step_size, self.goal_out_size)
        if cfg.device:
            sub_feature = sub_feature.to(cfg.device)
            real_goal = real_goal.to(cfg.device)
        for i in range(self.max_seq_len // self.step_size):
            idx = i * self.step_size
            sub_feature[:, i, :] = feature_array[:, idx + self.step_size, :] - feature_array[:, idx, :]

            if i == 0:
                real_goal[:, i, :] = self.goal_init[:batch_size, :]
            else:
                idx = (i - 1) * self.step_size + 1
                real_goal[:, i, :] = torch.sum(goal_array[:, idx:idx + 4, :], dim=1)

        # L2 noramlization
        sub_feature = F.normalize(sub_feature, p=2, dim=-1)
        real_goal = F.normalize(real_goal, p=2, dim=-1)

        cos_loss = F.cosine_similarity(sub_feature, real_goal, dim=-1)

        return cos_loss

    def worker_nll_loss(self, target, leak_out_array):
        """
        Get NLL loss for worker
        :return loss: batch_size * seq_len
        """
        loss_fn = nn.NLLLoss(reduction='none')
        loss = loss_fn(leak_out_array.permute([0, 2, 1]), target)

        return loss

    def worker_cos_reward(self, feature_array, goal_array):
        """
        Get reward for worker (cosine distance)
        :return: cos_loss: batch_size * seq_len
        """
        for i in range(int(self.max_seq_len / self.step_size)):
            real_feature = feature_array[:, i * self.step_size, :].unsqueeze(1).expand((-1, self.step_size, -1))
            feature_array[:, i * self.step_size:(i + 1) * self.step_size, :] = real_feature
            if i > 0:
                sum_goal = torch.sum(goal_array[:, (i - 1) * self.step_size:i * self.step_size, :], dim=1, keepdim=True)
            else:
                sum_goal = goal_array[:, 0, :].unsqueeze(1)
            goal_array[:, i * self.step_size:(i + 1) * self.step_size, :] = sum_goal.expand((-1, self.step_size, -1))

        offset_feature = feature_array[:, 1:, :]  # f_{t+1}, batch_size * seq_len * goal_out_size
        goal_array = goal_array[:, :self.max_seq_len, :]  # batch_size * seq_len * goal_out_size
        sub_feature = offset_feature - goal_array

        # L2 normalization
        sub_feature = F.normalize(sub_feature, p=2, dim=-1)
        all_goal = F.normalize(goal_array, p=2, dim=-1)

        cos_loss = F.cosine_similarity(sub_feature, all_goal, dim=-1)  # batch_size * seq_len
        return cos_loss

    def forward_leakgan(self, sentences, discriminator, if_sample, no_log=False,
                        start_letter=cfg.start_letter, train=False):
        """
        Get all feature and goals according to given sentences
        :param sentences: batch_size * max_seq_len, not include start token
        :param discriminator: discriminator model
        :param if_sample: if use to sample token
        :param no_log: if use log operation
        :param start_letter:
        :param train: if use temperature parameter
        :return samples, feature_array, goal_array, leak_out_array:
            - samples: batch_size * max_seq_len
            - feature_array: batch_size * (max_seq_len + 1) * total_num_filter
            - goal_array: batch_size * (max_seq_len + 1) * goal_out_size
            - leak_out_array: batch_size * max_seq_len * vocab_size
        """
        batch_size, seq_len = sentences.size()

        feature_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))
        goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))
        leak_out_array = torch.zeros((batch_size, seq_len + 1, self.vocab_size))

        samples = torch.zeros(batch_size, seq_len + 1).long()
        work_hidden = self.init_hidden(batch_size)
        mana_hidden = self.init_hidden(batch_size)
        leak_inp = torch.LongTensor([start_letter] * batch_size)
        real_goal = self.goal_init[:batch_size, :]

        if cfg.device:
            feature_array = feature_array.to(cfg.device)
            goal_array = goal_array.to(cfg.device)
            leak_out_array = leak_out_array.to(cfg.device)

        goal_array[:, 0, :] = real_goal  # g0 = goal_init
        for i in range(seq_len + 1):
            # Get feature
            if if_sample:
                dis_inp = samples[:, :seq_len]
            else:  # to get feature and goal
                dis_inp = torch.zeros(batch_size, seq_len).long()
                if i > 0:
                    dis_inp[:, :i] = sentences[:, :i]  # cut sentences
                    leak_inp = sentences[:, i - 1]

            if cfg.device:
                dis_inp = dis_inp.to(cfg.device)
                leak_inp = leak_inp.to(cfg.device)
            feature = discriminator.get_feature(dis_inp).unsqueeze(0)

            feature_array[:, i, :] = feature.squeeze(0)

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.forward(i, leak_inp, work_hidden, mana_hidden, feature,
                                                                   real_goal, no_log=no_log, train=train)
            leak_out_array[:, i, :] = out

            # ===My implement according to paper===
            # Update real_goal and save goal
            # if 0 < i < 4:  # not update when i=0
            #     real_goal = torch.sum(goal_array, dim=1)  # num_samples * goal_out_size
            # elif i >= 4:
            #     real_goal = torch.sum(goal_array[:, i - 4:i, :], dim=1)
            # if i > 0:
            #     goal_array[:, i, :] = cur_goal.squeeze(1)  # !!!note: save goal after update last_goal
            # ===LeakGAN origin===
            # Save goal and update real_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and not i % self.step_size:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.goal_init[:batch_size, :]

            # Sample one token
            if not no_log:
                out = torch.exp(out)
            out = torch.multinomial(out, 1).view(-1)  # [batch_size] (sampling from each row)
            samples[:, i] = out.data
            leak_inp = out

        # cut to seq_len
        samples = samples[:, :seq_len]
        leak_out_array = leak_out_array[:, :seq_len, :]
        return samples, feature_array, goal_array, leak_out_array

    def batchNLLLoss(self, target, discriminator, start_letter=cfg.start_letter):
        # loss_fn = nn.NLLLoss()
        # batch_size, seq_len = target.size()
        _, _, _, leak_out_array = self.forward_leakgan(target, discriminator, if_sample=False, no_log=False,
                                                       start_letter=start_letter)

        nll_loss = torch.mean(self.worker_nll_loss(target, leak_out_array))

        return nll_loss

    def init_hidden(self, batch_size=1):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if cfg.device:
            return h.to(cfg.device), c.to(cfg.device)
        else:
            return h, c

    def init_goal(self, batch_size):
        goal = torch.rand((batch_size, self.goal_out_size)).normal_(std=0.1)
        goal = nn.Parameter(goal)

        if cfg.device:
            return goal.to(cfg.device)
        else:
            return goal

    def init_params(self):
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] in ['bias', 'bias_ih_l0', 'bias_hh_l0']:
                    torch.nn.init.zeros_(param)
                else:
                    stddev = 1 / math.sqrt(param.shape[0])
                    if cfg.gen_init == 'uniform':
                        torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                    elif cfg.gen_init == 'normal':
                        torch.nn.init.normal_(param, std=stddev)
