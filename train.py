# -*- coding: UTF-8 -*-
'''
@Project ：CNN_LSTM
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import json
import torch
import numpy as np
import config as cfg
from torch import nn
from leakgan import LeakGan
from utils.generate import Generator

with open('.\\vocab\\image_coco.json', 'r') as f:
    dict = json.load(f)
vocab = dict['vocab']

def decode(samples):
    decode_mat = []
    for i, sample in enumerate(samples.numpy()):
        token = ''
        for idx in sample:
            try:
                token += np.array(vocab)[idx] + ' '
            except:
                token += 'UNK '
        decode_mat.append(token.strip())
    decode_mat = np.array(decode_mat).reshape([-1, 1])

    return decode_mat

if __name__ == '__main__':

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

    data_gen = Generator(file_path=cfg.datapath,
                         vocab_size=cfg.vocab_size,
                         max_seq_len=cfg.max_seq_len,
                         batch_size=cfg.batch_size,
                         train_ratio=cfg.train_ratio,
                         dataset=cfg.dataset)

    # === pretrain ===
    for inter_num in range(cfg.inter_epoch):
        # for discriminator
        for step in range(cfg.d_step):
            d_train_gen = data_gen.generate(training=True)
            for epoch in range(cfg.d_epoch):
                for iter in range(data_gen.get_train_len()):
                    real_sources = next(d_train_gen)
                    fake_sources = leak_gan.gen.sample(num_samples=real_sources.shape[0],
                                                       batch_size=real_sources.shape[0],
                                                       discriminator=leak_gan.dis,
                                                       start_letter=cfg.start_letter, train=False)
                    leak_gan.train_discriminator(real_sources, fake_sources)

                print('pretrain discriminator loss is {:.5f}'.format(leak_gan.dis_loss/data_gen.get_train_len()),
                      'pretrain discriminator acc is {:.3f}'.format(leak_gan.dis_acc/data_gen.get_train_len()*100))

                torch.save({'state_dict': leak_gan.dis.state_dict(),
                            'loss': leak_gan.dis_loss/data_gen.get_train_len()},
                           cfg.pre_dis_checkpoint_path + '\\Epoch{:0>3d}_loss{:.5f}.pth.tar'.format(
                               inter_num*cfg.d_step*cfg.d_epoch+step*cfg.d_epoch+epoch+1,
                               leak_gan.dis_loss/data_gen.get_train_len()
                           ))
                leak_gan.dis_loss, leak_gan.dis_acc = 0, 0

        # for generator
        for epoch in range(cfg.MLE_train_epoch):
            g_train_gen = data_gen.generate(training=True)
            for iter in range(data_gen.get_train_len()):
                real_sources = next(g_train_gen)
                leak_gan.pretrain_generator(real_sources)

            print('pretrain mana loss is {:.3f}'.format(leak_gan.pre_mana_loss/data_gen.get_train_len()),
                  'pretrain work loss is {:.3f}'.format(leak_gan.pre_work_loss/data_gen.get_train_len()))

            torch.save({'state_dict': leak_gan.gen.state_dict(),
                        'mana_loss': leak_gan.pre_mana_loss/data_gen.get_train_len(),
                        'work_loss': leak_gan.pre_work_loss/data_gen.get_train_len()},
                       cfg.pre_gen_checkpoint_path + '\\Epoch{:0>3d}_mana_loss{:.3f}_work_loss{:.3f}.pth.tar'.format(
                           inter_num*cfg.MLE_train_epoch + epoch + 1,
                           leak_gan.pre_mana_loss/data_gen.get_train_len(),
                           leak_gan.pre_work_loss/data_gen.get_train_len()
                       ))

            samples = leak_gan.gen.sample(num_samples=cfg.batch_size, batch_size=cfg.batch_size,
                                          discriminator=leak_gan.dis)
            print('generate samples: ', decode(samples))
            leak_gan.pre_mana_loss, leak_gan.pre_work_loss = 0, 0

    print('Starting Adversarial Training...')
    # === ADV Train ===
    for adv_epoch in range(cfg.ADV_train_epoch):
        # for generator
        for step in range(cfg.ADV_g_step):
            leak_gan.adv_train_generator()

        print('adv train mana loss is {:.3f}'.format(leak_gan.adv_mana_loss/cfg.ADV_g_step),
              'adv train work loss is {:.3f}'.format(leak_gan.adv_work_loss/cfg.ADV_g_step))

        torch.save({'state_dict': leak_gan.gen.state_dict(),
                    'mana_loss': leak_gan.adv_mana_loss/cfg.ADV_g_step,
                    'work_loss': leak_gan.adv_work_loss/cfg.ADV_g_step},
                   cfg.adv_gen_checkpoint_path + '\\Epoch{:0>3d}_mana_loss{:.3f}_work_loss{:.3f}.pth.tar'.format(
                       adv_epoch*cfg.ADV_g_step + step + 1,
                       leak_gan.adv_mana_loss/cfg.ADV_g_step,
                       leak_gan.adv_work_loss/cfg.ADV_g_step
                   ))

        samples = leak_gan.gen.sample(num_samples=cfg.batch_size, batch_size=cfg.batch_size,
                                      discriminator=leak_gan.dis)
        print('generate samples: ', decode(samples))
        leak_gan.adv_mana_loss, leak_gan.adv_work_loss = 0, 0

        # for discriminator
        for step in range(cfg.d_step):
            d_train_gen = data_gen.generate(training=True)
            for epoch in range(cfg.d_epoch):
                for iter in range(data_gen.get_train_len()):
                    real_sources = next(d_train_gen)
                    fake_sources = leak_gan.gen.sample(num_samples=real_sources.shape[0],
                                                       batch_size=real_sources.shape[0],
                                                       discriminator=leak_gan.dis,
                                                       start_letter=cfg.start_letter, train=False)
                    leak_gan.train_discriminator(real_sources, fake_sources)

                print('adv discriminator loss is {:.5f}'.format(leak_gan.dis_loss/data_gen.get_train_len()),
                      'adv discriminator acc is {:.3f}'.format(leak_gan.dis_acc/data_gen.get_train_len()*100))

                torch.save({'state_dict': leak_gan.dis.state_dict(),
                            'loss': leak_gan.dis_loss/data_gen.get_train_len()},
                           cfg.adv_dis_checkpoint_path + '\\Epoch{:0>3d}_loss{:.5f}.pth.tar'.format(
                               adv_epoch*cfg.d_step*cfg.d_epoch+step*cfg.d_epoch+epoch+1,
                               leak_gan.dis_loss/data_gen.get_train_len()
                           ))

                leak_gan.dis_loss, leak_gan.dis_acc = 0, 0
