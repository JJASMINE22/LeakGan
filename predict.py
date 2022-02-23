# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import config as cfg
from leakgan import LeakGan
from train import decode

leak_gan = LeakGan(gen_embed_dim=cfg.gen_embed_dim,
                   gen_hidden_dim=cfg.gen_hidden_dim,
                   vocab_size=cfg.vocab_size,
                   max_seq_len=cfg.max_seq_len,
                   goal_size=cfg.goal_size,
                   step_size=cfg.step_size,
                   dis_embed_dim=cfg.dis_embed_dim,
                   filter_sizes=cfg.filter_sizes,
                   num_filters=cfg.num_filters,
                   gen_lr=cfg.gen_lr,
                   dis_lr=cfg.dis_lr,
                   batch_size=cfg.batch_size,
                   rollout_num=cfg.rollout_num,
                   dropout=cfg.dropout,
                   padding_idx=cfg.padding_idx,
                   device=cfg.device)

gen_ckpt_path = '.\\saved\\pre_generator\\Epoch080_mana_loss-0.395_work_loss0.946.pth.tar'
dis_ckpt_path = '.\\saved\\pre_discriminator\\Epoch150_loss0.05363.pth.tar'
gen_ckpt = torch.load(gen_ckpt_path)
gen_state_dict = gen_ckpt['state_dict']
leak_gan.gen.load_state_dict(gen_state_dict)

dis_ckpt = torch.load(dis_ckpt_path)
dis_state_dict = dis_ckpt['state_dict']
leak_gan.dis.load_state_dict(dis_state_dict)
samples = leak_gan.gen.sample(32, 32, leak_gan.dis)
print('samples: ', decode(samples))