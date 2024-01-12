import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import argparse
from collections import deque
import random 

buffer = deque(maxlen=1000)
class Actor(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
class Critic(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class ActorCritic(nn.Module):
    def __init__(self,n_states, n_actions,
                 actor_lr, critic_lr, gamma):
        super().__init__()
        self.actor = Actor(n_states, n_actions)
        self.critic = Critic(n_states)
        self.critic_target = Critic(n_states)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.gamma = gamma
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def get_action(self, state):
        probs = self.actor(state)
        m = Categorical(probs)
        action = m.sample()
        logp_action = m.log_prob(action)
        return action, logp_action
    
    def compute_value_loss(self, bs, blogp_a, br, bd, bns):
        # 目标价值。
        with torch.no_grad():
            target_value = br + self.args.discount * torch.logical_not(bd) * self.V_target(bns).squeeze()

        # 计算value loss。
        value_loss = F.mse_loss(self.V(bs).squeeze(), target_value)
        return value_loss

    def compute_policy_loss(self, bs, blogp_a, br, bd, bns):
        # 建议对比08_a2c.py，比较二者的差异。
        with torch.no_grad():
            value = self.V(bs).squeeze()

        policy_loss = 0
        for i, logp_a in enumerate(blogp_a):
            policy_loss += -logp_a * value[i]
        policy_loss = policy_loss.mean()
        return policy_loss

    def soft_update(self, tau=0.01):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.V_target, self.V, tau)

    def update(self,batch):
        if len(buffer) < batch:
            return
        transitions = random.sample(buffer, batch)
        state, action, logp_action, reward, next_state, done = zip(*transitions)
        state = torch.cat(state)
        action = torch.tensor(action)
        logp_action = torch.cat(logp_action)
        reward = torch.tensor(reward)
        next_state = torch.cat(next_state)
        done = torch.tensor(done,dtype=torch.bool)
        
        
        # 预测的当前时刻的state_value
        td_value = self.critic(state)
        # 目标的当前时刻的state_value
        td_target = reward + self.gamma * self.critic_target(next_state).max(1)[0] * (~done)
        # 时序差分的误差计算，目标的state_value与预测的state_value之差
        td_delta = td_target - td_value
        
        # 对每个状态对应的动作价值用log函数
        log_probs = torch.log(self.actor(state).gather(1, action))
        # 策略梯度损失
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 值函数损失，预测值和目标值之间
        critic_loss = torch.mean(F.mse_loss(self.critic(state).gather(1,action.unsqueeze(1)).squeeze(1), td_target.detach()))
        # 优化器梯度清0
        self.actor_optimizer.zero_grad()  # 策略梯度网络的优化器
        self.critic_optimizer.zero_grad()  # 价值网络的优化器
        # 反向传播
        actor_loss.backward()
        critic_loss.backward()
        # 参数更新
        self.actor_optimizer.step()
        self.critic_optimizer.step()

def train(args, env, agent):
    return_list = []
    for i in range(args.episodes):
        episode_return = 0
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        while True:
            action, logp_action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.append((state, action, logp_action,reward, next_state, done))
            state = next_state
            episode_return += reward
            if done:
                break
        agent.update(args.batch_size)
        if i%10 == 0:
            agent.critic_target.load_state_dict(agent.critic.state_dict())
        mean_reward = episode_return/(i+1)
        return_list.append(mean_reward)
        print(f"Episode{i+1} reward:{mean_reward}")
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN Returns')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default=10_000, type=int, help="Episodes.")
    parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
    parser.add_argument("--gamma", default=0.99, type=float, help="Discount factor.")
    parser.add_argument("--actor_lr", default=0.001, type=float, help="Actor learning rate.")
    parser.add_argument("--critic_lr", default=0.001, type=float, help="Critic learning rate.")
    args = parser.parse_args()
    env = gym.make('CartPole-v1')
    n_states = env.observation_space.shape[0]  # 状态数 4
    n_actions = env.action_space.n  # 动作数 2
    agent = ActorCritic(n_states, n_actions,
                 args.actor_lr, args.critic_lr, args.gamma)
    train(args, env, agent)
if __name__ == '__main__':
    main()