import gym 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import argparse
import numpy as np

GAMA = 0.95
buffer = deque(maxlen=1000)
Learning_rate = 0.001
EXPLORATION_DECAY = 0.999
EXPLORATION_MIN = 0.001
EXPLORATION_MAX = 1.0

class QNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_states, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.qnet = QNet(n_states, n_actions)
        self.target_qnet = QNet(n_states, n_actions)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=Learning_rate)
        self.exploration_rate = EXPLORATION_MAX

    def get_action(self, state):
        if random.random() < self.exploration_rate:
            action = np.random.randint(0, self.n_actions)
        else:
            qvalues = self.qnet(state)
            action = qvalues.argmax().item()
        return action
    def learn(self,batch):
        if len(buffer) < batch:
            return
        transitions = random.sample(buffer, batch)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.cat(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.cat(next_state)
        done = torch.tensor(done,dtype=torch.bool)
        q = self.qnet(state)
        q = q.gather(1, action.unsqueeze(1)).squeeze(1)
        q_target = reward + GAMA*(~done)*self.target_qnet(next_state).max(1)[0]
        loss = F.mse_loss(q_target, q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)
def train(args, env, agent):
    agent.qnet.train()
    return_list = []
    
    for i in range(args.episodes):
        episode_reward = 0
        state,_ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        while True:
            action = agent.get_action(state)
            next_state, reward, terminated,truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.append((state, action, reward, next_state, done))
            episode_reward += reward
            state = next_state
            if done:
                break
        agent.learn(args.batch_size)
        if i%10 == 0:
            agent.target_qnet.load_state_dict(agent.qnet.state_dict())
        # mean_reward = episode_reward/(i+1)
        return_list.append(episode_reward)
        print(f"Episode{i+1} reward:{episode_reward}")
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN Returns')
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int,default=3000,help='训练次数')
    parser.add_argument('--batch_size', type=int,default=64,help='batch_size')
    args = parser.parse_args()
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQN(n_states, n_actions)
    train(args, env, agent)

if __name__ == "__main__":
    main()