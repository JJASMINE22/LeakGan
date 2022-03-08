# -*- coding: UTF-8 -*-
'''
@Project ：LeakGan
@File    ：rollout.py
@IDE     ：PyCharm 
@Author  ：XinYi Huang
'''
import copy
import torch
import config as cfg

class ROLLOUT:
    def __init__(self, gen):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size
        self.step_size = gen.step_size
        self.goal_out_size = gen.goal_out_size

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, need_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if cfg.device:
            samples = samples.to(cfg.device)

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.gen.forward(inp, hidden, need_hidden=True)

        return samples

    def rollout_mc_search_leakgan(self, sentences, discriminator, given_num):

        batch_size, seq_len = sentences.size()

        goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))

        work_hidden = self.gen.init_hidden(batch_size)
        mana_hidden = self.gen.init_hidden(batch_size)
        real_goal = self.gen.goal_init[:batch_size, :]
        # out = 0

        if cfg.device:
            goal_array = goal_array.to(cfg.device)
            real_goal = real_goal.to(cfg.device)

        # get current state
        for i in range(given_num):
            # Get feature.
            dis_inp = torch.zeros(batch_size, seq_len).long()
            dis_inp[:, :i + 1] = sentences[:, :i + 1]  # cut sentences
            leak_inp = sentences[:, i]
            if cfg.device:
                dis_inp = dis_inp.to(cfg.device)
                leak_inp = leak_inp.to(cfg.device)
            feature = discriminator.get_feature(dis_inp).unsqueeze(0)

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        # MC search
        for i in range(given_num, self.max_seq_len):
            # Sample one token
            out = torch.multinomial(torch.exp(out), 1).view(-1)  # [num_samples] (sampling from each row)
            samples[:, i] = out.data

            # Get feature
            dis_inp = samples
            if cfg.device:
                dis_inp = dis_inp.to(cfg.device)
            feature = discriminator.get_feature(dis_inp).unsqueeze(0)
            leak_inp = out

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        if cfg.device:
            samples = samples.to(cfg.device)

        return samples

    def get_reward_leakgan(self, sentences, rollout_num, discriminator, current_k):
        """
        get reward via Monte Carlo search for LeakGAN
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param discriminator: discriminator model
        :param current_k: current training gen
        :return: reward: batch_size * (max_seq_len / step_size)
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * (self.max_seq_len // self.step_size), batch_size]).float()
            if cfg.device:
                rewards = rewards.to(cfg.device)
            idx = 0
            for i in range(rollout_num):
                for t in range(self.max_seq_len // self.step_size):
                    given_num = t * self.step_size + 1  # 1, 5, 9, ..
                    samples = self.rollout_mc_search_leakgan(sentences, discriminator, given_num)
                    out = discriminator(samples)
                    out = torch.softmax(out, dim=-1) if cfg.loss_mode == 'CrossEntropy' else torch.sigmoid(out)  # or relu
                    reward = out[:, current_k+1] if cfg.loss_mode is 'CrossEntropy' else out[:, current_k]
                    rewards[idx] = reward
                    idx += 1

        # rewards = rewards.view(batch_size, self.max_seq_len // self.step_size, rollout_num)
        # rewards = torch.mean(rewards, dim=-1)
        rewards = rewards.view(rollout_num, self.max_seq_len//self.step_size, batch_size)
        rewards = rewards.mean(dim=0).permute([1, 0])

        return rewards
