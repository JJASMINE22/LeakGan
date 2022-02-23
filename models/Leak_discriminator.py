# -*- coding: UTF-8 -*-
'''
@Project ：LeakGan
@File    ：Leak_discriminator.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import torch
import config as cfg
from models.discriminator import CNNDiscriminator

class LeakGAN_D(CNNDiscriminator):
    def __init__(self, **kwargs):
        super(LeakGAN_D, self).__init__(**kwargs)

    def discriminator_loss(self, preds, labels):

        if cfg.loss_mode is 'CrossEntropy':
            loss = self.cross_entropy_loss(preds, labels)
        else:
            loss = torch.mean(torch.relu(1. - labels * preds))

        return loss
