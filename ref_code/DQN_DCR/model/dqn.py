'''
DQN DCR策略
@lizw
'''


import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import math


def orthogonal_init(layers, gain=1.0):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, next_avail, done):
        # 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, next_avail, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, next_avail, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, next_avail, done

    def __len__(self):
        # 返回当前存储的量
        return len(self.buffer)


class Q_NET(nn.Module):
    # 用于评估信道资源  由fc和lstm构成
    def __init__(self, n_states, n_actions, hidden_dim, device):
        super(Q_NET, self).__init__()
        self.n_states = n_states
        self.n_action = n_actions
        self.hidden_dim = hidden_dim
        self.device = device
        # 神经网络各层定义
        self.embedding_st = nn.Embedding(7, 3)  # 用于请求类型编码
        self.embedding_h = nn.Embedding(3, 1)  # 用于切换类型编码
        self.lstm = nn.LSTM(96, self.hidden_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, n_actions * self.hidden_dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, state):
        if len(state) == self.n_states:
            u_ = torch.tensor(state[0], device=self.device, dtype=torch.float32)  # CUM
            m_ = torch.tensor(state[1], device=self.device, dtype=torch.float32)  # 请求
            g_ = torch.tensor(state[2], device=self.device, dtype=torch.float32)  # TDD
            # cum处理
            # #编码
            u1 = self.embedding_st(u_[:, :, 0].long())
            u2 = self.embedding_h(u_[:, :, 1].long())
            # #合并 && reshape
            u = torch.cat((u1, u2, u_[:, :, 2].unsqueeze(-1), g_), dim=2).flatten(start_dim=1)
            assert u.shape == torch.Size([5, 96]), 'invalid u'  # 判断cum预处理是否正确
            # # lstm网络处理
            _, (u, _) = self.lstm(u)
            # # 转换为参数矩阵
            u_w = self.mlp1(u)
            u_w = u_w.reshape(self.n_action, -1)
            # 请求信息处理
            # #编码
            m1 = self.embedding_st(m_[:, 0].long())
            m2 = self.embedding_h(m_[:, 1].long())
            # # 合并
            m = torch.cat((m1, m2, m_[:, 2].unsqueeze(-1)), dim=1)
            assert m.shape == torch.Size([m_.shape[0], 5]), 'invalid m'
            m_o = self.mlp2(m)
            # 匹配
            q_table = torch.mm(m_o, u_w.T)  # 每一行代表每个请求的不同动作q值，每一列代表每个动作的不同请求q值
        else:
            raise ValueError('invalid state')
        return q_table


class DQN:
    def __init__(self, n_states, n_actions, cfg):
        self.n_states = n_states  # 状态元素个数
        self.n_actions = n_actions  # 总的动作个数
        self.hidden_dim = cfg.hidden_dim
        self.device = cfg.device  # 设备，cpu或gpu等
        self.gamma = cfg.gamma  # 奖励的折扣因子
        self.batch_size = cfg.batch_size
        # e-greedy策略相关参数
        self.frame_idx = 0  # 用于epsilon的衰减计数
        self.epsilon = lambda frame_idx: cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * math.exp(
            -1. * frame_idx / cfg.epsilon_decay)
        # 定义Q网络
        self.policy_net = Q_NET(self.n_states, self.n_actions, self.hidden_dim, self.device).to(self.device)
        self.target_net = Q_NET(self.n_states, self.n_actions, self.hidden_dim, self.device).to(self.device)
        # 复制网络参数
        for target_param, param in zip(self.target_net.parameters(),
                                       self.policy_net.parameters()):  # 复制参数到目标网路targe_net
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 优化器
        self.memory = ReplayBuffer(cfg.memory_capacity)  # 经验回放

    def choose_action(self, state, avail, mode, net):
        # 选择动作
        """
        avail 为可选的信道集合
        元组形式，每个元素包含一个类别的可用DC编号
        ([1,2,3],[5,8],[9])
        """
        # avail_idx = np.where(avail == 0)[0]  # 获得空闲DC
        if mode == 'train':
            self.frame_idx += 1
        n_t = state[1].shape[0]  # 请求总数
        action = np.zeros((n_t,))  # 初始化输出动作
        # 统计数量
        avail_counts = np.zeros(self.n_actions)
        avail_counts[0] = 1000  # 基于一个足够大的数
        avail_counts[1] = len(avail[0])
        avail_counts[2] = len(avail[1])
        avail_counts[3] = len(avail[2])
        if random.random() > self.epsilon(self.frame_idx) or mode == 'test':
            with torch.no_grad():
                if net == 'policy':
                    q_table = self.policy_net(state).cpu()  # 转换为array
                elif net == 'target':
                    q_table = self.target_net(state).cpu()  # 转换为array
                else:
                    raise ValueError('invalid net_mode')
        else:
            q_table = torch.rand(n_t, self.n_actions)  # 随机动作
        alloc_counts = 0  # 已分配数量
        while (avail_counts[1] + avail_counts[2] + avail_counts[3]) > 0 and alloc_counts < n_t:
            #  按照最大Q分配
            req_id = torch.floor(q_table.argmax() / self.n_actions).long()  # 请求id 从0开始
            dc_type = q_table.argmax() % self.n_actions  # DC信道类别 从0开始
            if avail_counts[dc_type] > 0:
                # 如果有可用DC
                action[req_id] = dc_type  # 分配dc
                avail_counts[dc_type] = avail_counts[dc_type] - 1  # 可用dc数量减1
                alloc_counts += 1  # 已分配数量加1
                q_table[req_id] = -9999  # 使该行为失效
            else:
                q_table[req_id, dc_type] = -9999  # 使该值失效
        return action.astype(np.int32)

    def update(self,):
        if len(self.memory) < self.batch_size:  # 当memory中不满足一个批量时，不更新策略
            return
        # 决策网络更新
        state_batch, action_batch, reward_batch, next_state_batch, next_avail_batch, done_batch = self.memory.sample(
            self.batch_size)
        q_value_batch = torch.zeros(self.batch_size, 1, device=self.device)
        target_q_value_batch = torch.zeros(self.batch_size, 1, device=self.device)
        for i in range(len(state_batch)):
            state = state_batch[i]
            action = torch.tensor(action_batch[i], device=self.device, dtype=torch.int64).unsqueeze(1)
            q_value = self.policy_net(state).gather(dim=1, index=action)
            q_value_batch[i] = torch.sum(q_value)  # 求和
            next_state = next_state_batch[i]
            next_avail = next_avail_batch[i]
            reward = torch.tensor(reward_batch[i], dtype=torch.float32)
            done = torch.tensor(done_batch[i], dtype=torch.float32)
            with torch.no_grad():
                # 基于target网络选择动作 DQN
                # next_action = self.choose_action(next_state, next_avail, 'test', 'target')
                # 基于policy网络选择动作 DDQN
                next_action = self.choose_action(next_state, next_avail, 'test', 'policy')
                next_action = torch.tensor(next_action, device=self.device, dtype=torch.int64).unsqueeze(1)
                next_q_value = self.target_net(next_state).gather(dim=1, index=next_action)
                next_q_value = torch.sum(next_q_value)  # 求和
            target_q_value_batch[i] = reward + self.gamma * next_q_value * (1 - done)

        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        loss = nn.MSELoss()(q_value_batch, target_q_value_batch)  # 计算均方根损失
        # 优化更新模型
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path + 'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path + 'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


# if __name__ == "__main__":
