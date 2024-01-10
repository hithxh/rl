import gym 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
buffer = deque(maxlen=1000)
class QNet(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, n_actions)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.qnet = QNet(n_observations, n_actions)
        self.target_qnet = QNet(n_observations, n_actions)
        self.target_qnet.load_state_dict(self.qnet.state_dict())
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=0.001)
    def get_action(self, state):
        if random.random() < 0.8:
            qvalues = self.qnet(state)
            action = qvalues.argmax().item()
        else:
            action = np.random.randint(self.n_actions)
        return action
    def learn(self,batch):
        transitions = random.sample(buffer, batch)
        state, action, reward, next_state, done = zip(*transitions)
        state = torch.cat(state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_state = torch.cat(next_state)
        done = torch.tensor(done)
        q = self.qnet(state)
        q = q.gather(1, action.unsqueeze(1)).squeeze(1)
        q_target = reward + (~done)*self.target_qnet(next_state).max(1)[0].unsqueeze(1)
        loss = F.mse_loss(q, q_target)
        # print(f"Loss:{loss.item()}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def main():
    env = gym.make("CartPole-v1")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    dqn = DQN(n_observations, n_actions)
    dqn.qnet.train()
    return_list = []
    for i in range(65000):
        episode_reward = 0
        episode_length = 0
        print(f"Episode:{i} ")
        state,info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        while True:
            action = dqn.get_action(state)
            next_state, reward, terminated,truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            buffer.append((state, action, reward, next_state, done))
            episode_reward += reward
            episode_length += 1
            state = next_state
            if len(buffer) > 10000:
                dqn.learn(64)
            if done==True:
                break
        if i%10 == 0:
            dqn.target_qnet.load_state_dict(dqn.qnet.state_dict())
        return_list.append(episode_reward)
        print(f"Episode reward:{episode_reward}, Episode length:{episode_length}")
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN Returns')
    plt.show()

if __name__ == "__main__":
    main()