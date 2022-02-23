# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import config as cfg
from torch import nn
from utils import rollout
from models.Leak_generator import LeakGAN_G
from models.Leak_discriminator import LeakGAN_D


class LeakGan:
    def __init__(self,
                 gen_embed_dim: int,
                 gen_hidden_dim: int,
                 vocab_size: int,
                 max_seq_len: int,
                 goal_size: int,
                 step_size: int,
                 dis_embed_dim: int,
                 filter_sizes: list,
                 num_filters: list,
                 gen_lr: float,
                 dis_lr: float,
                 batch_size: int,
                 rollout_num: int,
                 dropout: float,
                 padding_idx=None,
                 device=None,
                 ignore_pretrain=False,
                 gen_ckpt_path=None,
                 dis_ckpt_path=None,
                 **kwargs):
        self.gen_embed_dim = gen_embed_dim
        self.gen_hidden_dim = gen_hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.goal_size = goal_size
        self.step_size = step_size
        self.dis_embed_dim = dis_embed_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.batch_size = batch_size
        self.rollout_num = rollout_num
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.device = device

        goal_out_size = sum(self.num_filters)
        self.gen = LeakGAN_G(embedding_dim=self.gen_embed_dim, hidden_dim=self.gen_hidden_dim,
                             vocab_size=self.vocab_size, max_seq_len=self.max_seq_len,
                             goal_size=self.goal_size, goal_out_size=goal_out_size,
                             step_size=self.step_size, padding_idx=self.padding_idx)
        self.dis = LeakGAN_D(embed_dim=self.dis_embed_dim, vocab_size=self.vocab_size,
                             filter_sizes=self.filter_sizes, num_filters=self.num_filters,
                             padding_idx=self.padding_idx, dropout=self.dropout)

        if self.device:
            self.gen = self.gen.to(self.device)
            self.dis = self.dis.to(self.device)
        if ignore_pretrain:
            gen_ckpt = torch.load(gen_ckpt_path)
            gen_state_dict = gen_ckpt['state_dict']
            dis_ckpt = torch.load(dis_ckpt_path)
            dis_state_dict = dis_ckpt['state_dict']
            self.gen.load_state_dict(gen_state_dict)
            self.dis.load_state_dict(dis_state_dict)

        self.rollout_func = rollout.ROLLOUT(self.gen)

        mana_params, work_params = self.split_params()
        self.mana_opt = torch.optim.Adam(mana_params, lr=self.gen_lr)
        self.work_opt = torch.optim.Adam(work_params, lr=self.gen_lr)
        self.gen_opt = [self.work_opt, self.mana_opt]

        self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=self.dis_lr)

        self.pre_mana_loss, self.pre_work_loss, self.adv_mana_loss, self.adv_work_loss = 0, 0, 0, 0
        self.dis_loss, self.dis_acc = 0, 0

    def split_params(self):
        mana_params = list()
        work_params = list()

        mana_params += list(self.gen.manager.parameters())
        mana_params += list(self.gen.mana2goal.parameters())
        mana_params.append(self.gen.goal_init)

        work_params += list(self.gen.embeddings.parameters())
        work_params += list(self.gen.worker.parameters())
        work_params += list(self.gen.work2goal.parameters())
        work_params += list(self.gen.goal2goal.parameters())

        return mana_params, work_params

    def optimize_multi(self, losses):
        for i, (loss, opt) in enumerate(zip(losses, self.gen_opt)):
            opt.zero_grad()
            loss.backward(retain_graph=True if not i else False)
            opt.step()

    def pretrain_generator(self, sources):
        if self.device:
            sources = torch.LongTensor(sources).to(self.device)

        mana_loss, work_loss = self.gen.pretrain_loss(sources, self.dis)
        self.optimize_multi([work_loss, mana_loss])
        self.pre_mana_loss += mana_loss.data.item()
        self.pre_work_loss += work_loss.data.item()
        # print(mana_loss.data.item(), work_loss.data.item())

    def adv_train_generator(self, current_k=0):
        """
        The gen is trained using policy gradients, using the reward from the discriminator.
        Training is done for num_batches batches.
        """
        with torch.no_grad():
            gen_samples = self.gen.sample(self.batch_size, self.batch_size, self.dis,
                                          train=True)
            target = gen_samples.to(self.device)

        rewards = self.rollout_func.get_reward_leakgan(target, self.rollout_num, self.dis,
                                                       current_k)  #.cpu() reward with MC search
        mana_loss, work_loss = self.gen.adversarial_loss(target, rewards, self.dis)

        # update parameters
        self.optimize_multi([work_loss, mana_loss])
        self.adv_mana_loss += mana_loss.data.item()
        self.adv_work_loss += work_loss.data.item()

    def train_discriminator(self, real_sources, fake_sources):
        """
        Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
        Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
        """
        if self.device:
            real_sources = torch.LongTensor(real_sources).to(self.device)
            fake_sources = torch.LongTensor(fake_sources).to(self.device)

        real_predictions = self.dis.forward(real_sources)
        fake_predictions = self.dis.forward(fake_sources)
        predictions = torch.cat((real_predictions, fake_predictions), dim=0)
        labels = torch.cat((torch.ones_like(real_predictions[:, 0], dtype=torch.long),
                            torch.zeros_like(fake_predictions[:, 0], dtype=torch.long)), dim=0) \
            if cfg.loss_mode is 'CrossEntropy' else \
            torch.cat((torch.ones_like(real_predictions),
                       -torch.ones_like(fake_predictions)), dim=0)
        loss = self.dis.discriminator_loss(predictions, labels)
        self.dis_opt.zero_grad()
        loss.backward(retain_graph=False)
        self.dis_opt.step()

        self.dis_loss += loss.data.item()
        # self.dis_acc += (torch.sum(torch.gt(real_predictions.squeeze(1), 0)).data.item() +
        #                  torch.sum(torch.less_equal(fake_predictions.squeeze(1), 0)).data.item())/predictions.size(0)
        self.dis_acc += torch.sum((torch.eq(predictions.argmax(dim=-1), labels))).data.item()/predictions.size(0)
