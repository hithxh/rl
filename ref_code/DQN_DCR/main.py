# DQN DCR for VDE-SAT
# 2023/9/3
# @Lizw


import os
import torch
import numpy as np
import datetime
import time
from common.utils import plot_rewards
from common.utils import save_results
from common.utils import make_dir
from model.dqn import DQN
from environment import Env

curr_path = os.path.dirname(os.path.abspath(__file__))
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class Config:
    def __init__(self) -> None:
        # 环境参数与实验设置
        self.env_name = "data_0"  # 环境名称
        self.algo_name = "DQN"  # 算法名称
        self.cum_length = 5  # 历史CUM长度
        self.continuous = False  # 环境是否为连续动作
        self.seed = 10  # 随机种子，置0则不设置随机种子
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
        # 算法tricks
        self.use_grad_clip = True  # 是否使用梯度裁剪
        self.use_lr_decay = False  # 是否使用递减的学习率
        self.use_orthogonal_init = True  # 是否使用正交初始化
        self.use_adv_norm = False  # 是否使用advantage 标准化
        self.use_state_norm = False  # 是否使用状态标准化
        self.use_reward_norm = False  # 是否使用奖励标准化
        self.use_decoupled = False  # 是否使用解耦框架

        # 算法超参数
        self.batch_size = 128  # mini-batch SGD中的批量大小
        self.gamma = 0.95  # 强化学习中的折扣因子
        self.lr = 1e-4  # 学习率
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 2000
        self.hidden_dim = 256
        self.update_fre = 12
        self.target_update = 10  # 策略更新频率
        self.memory_capacity = 10000  # 经验池容量
        self.train_eps = 1000  # 训练的回合数
        self.test_eps = 3  # 测试的回合数
        ################################################################################

        # 保存结果相关参数
        # 保存结果的路径
        self.result_path = curr_path + "/outputs/" + self.algo_name + self.env_name + '/' + '/results/'
        # 保存模型的路径
        self.model_path = curr_path + "/outputs/" + self.algo_name + self.env_name + '/' + '/models/'
        self.save_fig = True  # 是否保存图片
        ################################################################################


def env_agent_config(cfg_):
    # 创建环境和智能体
    env_ = Env(cfg_.env_name, cfg_.algo_name, cfg_.cum_length)  # 创建环境
    n_states = env_.observation_space  # 状态变量元素个数
    n_actions = env_.action_dim  # 动作维度
    agent_ = DQN(n_states, n_actions, cfg_)  # 创建智能体
    if cfg_.seed != 0:  # 设置随机种子
        torch.manual_seed(cfg_.seed)
        np.random.seed(cfg_.seed)
    return env_, agent_


def train(cfg_, env_, agent_):
    print('start training!')
    print(f'env:{cfg_.env_name}, algo:{cfg_.algo_name}, device:{cfg_.device}')
    rewards_ = []  # 记录所有回合的奖励
    ma_rewards_ = []  # 记录所有回合的滑动平均奖励
    iters = 0  # 更新次数
    for i_ep in range(cfg_.train_eps):
        t0 = time.time()
        state, avail = env_.reset(seed=cfg_.seed)  # 得到初始状态
        done = False
        ep_reward = 0
        t = 0   # 起始时间
        while not done:
            # 卫星决策
            action = agent_.choose_action(state, avail, 'train', 'policy')  # 选择动作
            # 执行决策
            reward, next_state, next_avail, done = env_.step(state, action, avail, t)  # 执行动作
            t += 1  # 时间索引
            ep_reward += reward  # 在标准化之前添加
            # print(f' state:{state},action:{action:.2f},reward:{reward.item():.2f},time_stamp:{t}')
            if not done:  # 最后一个不存入(人为结束回合，最后一个经验无效)
                agent_.memory.push(state, action, reward.item(), next_state, next_avail, done)

            # 策略更新
            agent_.update()
            iters += 1
            state = next_state  # 更新状态
            avail = next_avail
            if iters % cfg_.target_update == 0:
                agent_.target_net.load_state_dict(agent_.policy_net.state_dict())  # 更新目标网络
        rewards_.append(ep_reward)
        if ma_rewards_:
            ma_rewards_.append(0.9 * ma_rewards_[-1] + 0.1 * ep_reward)  # 平滑
        else:
            ma_rewards_.append(ep_reward)
        if (i_ep + 1) % 1 == 0:
            t1 = time.time()
            delta_t = t1 - t0
            print(f"episode:{i_ep + 1}/{cfg_.train_eps}, returns:{ep_reward:.2f}, time_consuming:{delta_t:.2f}s")
    print('finish training!')
    return rewards_, ma_rewards_


def evaluate(cfg_, env_, agent_):
    ep_reward = 0
    for i_ep in range(cfg_.test_eps):
        state = env_.reset()
        done = False
        t = 0
        while not done:
            # 卫星决策
            action = agent_.choose_action(state)  # 选择动作
            # 执行决策
            t += 1  # 时间索引
            next_state, reward, done = env_.step(action, t)  # 执行动作
            ep_reward += reward  # 在标准化之前添加
            # print(f'state:{state},action:{action:.2f},reward:{reward.item():.2f},time_stamp:{t}')
            state = next_state  # 更新状态

    return ep_reward / cfg_.test_eps


if __name__ == "__main__":
    cfg = Config()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)  # 创建保存结果和模型路径的文件夹
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")
    # # 测试
    # # env, agent = env_agent_config(cfg)
    # # agent.load(path=cfg.model_path)
    # # rewards, ma_rewards = evaluate(cfg, env, agent)
    # # save_results(rewards, ma_rewards, tag='test', path=cfg.result_path)
    # # plot_rewards(rewards, ma_rewards, cfg, tag="test")
