import gym
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

env = gym.make("CartPole-v1")
def main() -> int:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Online and offline model for learning

    model = DQN(env.observation_space, env.action_space, 128).to(device)

    target = DQN(env.observation_space, env.action_space, 24).to(device)

    # target.eval()

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    loss_fn = nn.MSELoss()


    memory = Memory(10_000)

    obs = env.reset()
    tot_rew = 0
    for it in range(65_000):
        # print("it = ", it)
        # Do this for the batch norm
        # model.eval()

        # Maybe explore
        if np.random.random() <= epsilon_greedy(1.0, .01, 15_000, it):
            action = env.action_space.sample()
            

            

        else:
            state = wrap_input(obs, device).unsqueeze(0)
            action  = model(state).argmax().item()
            


            # print("epsilon_greedy(1.0, .01, 15_000, it) = ", epsilon_greedy(1.0, .01, 15_000, it))
            
            # print("check = ", model(state).detach().numpy())
            # print("action = ", action)
            


        # Act in environment and store the memory

        next_state, reward, done, info = env.step(action)
        tot_rew += reward
        if done:
            next_state = np.zeros(env.observation_space.shape)
        memory.store([obs, action, reward, int(done), next_state])
        done = done

        obs = next_state

        if done:
            print("tot_rew = ", tot_rew)
            obs= env.reset()
            tot_rew = 0

        # Train
        if len(memory) > 500:
            model.train()
            states, actions, rewards, dones, next_states = memory.sample(128)

            # Wrap and move all values to the cpu

            states = wrap_input(states, device)
            # print("states.shape = ",states.shape)
            actions = wrap_input(actions, device, torch.int64, reshape=True)
            next_states = wrap_input(next_states, device)
            rewards = wrap_input(rewards, device, reshape=True)
            dones = wrap_input(dones, device, reshape=True)

            # Get current q-values
            qs = model(states)
            # print("qs.shape = ", qs.shape)
            qs = torch.gather(qs, dim=1, index=actions)

            # Compute target q-values
            with torch.no_grad():
                next_qs, _ = model(next_states).max(dim=1)
                next_qs = next_qs.reshape(-1, 1)

            target_qs = rewards + .9 * (1 - dones) * next_qs.reshape(-1, 1)

            # Compute loss
            loss = loss_fn(qs, target_qs)
            # print("loss.shape = ", loss)
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            # nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Backprop
            optimizer.step()

            # soft update
        #     with torch.no_grad():
        #         for target_param, local_param in zip(target.parameters(), model.parameters()):
        #             target_param.data.copy_(1e-2 * local_param.data + (1 - 1e-2) * target_param.data)


        # if it % 200 == 0:
        #     target.load_state_dict(model.state_dict())

# models.py
class FlatExtractor(nn.Module):
    '''Does nothing but pass the input on'''
    def __init__(self, obs_space):
        super(FlatExtractor, self).__init__()

        self.n_flatten = 1

    def forward(self, obs):
        return obs


class DQN(nn.Module):
    def __init__(self, obs_space, act_space, layer_size):
        super(DQN, self).__init__()

        # Feature extractor
        if len(obs_space.shape) == 1:
            self.feature_extractor = env.observation_space.shape[0]

        elif len(obs_space.shape) == 3:
            self.feature_extractor = NatureCnn(obs_space)
        else:
            raise NotImplementedErorr("This type of environment is not supported")
        

        # Neural network
        self.net = nn.Sequential(
            nn.Linear(self.feature_extractor, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, act_space.n),
        )

    def forward(self, obs):

        return self.net(obs)

# memory.py
import random
from collections import deque

class Memory(object):
    def __init__(self, maxlen):
        self.memory = deque(maxlen=maxlen)

    def store(self, experience):
        self.memory.append(experience)

    def sample(self, n_samples):
        return zip(*random.sample(self.memory, n_samples))

    def __len__(self):
        return len(self.memory)

# utils.py
def wrap_input(arr, device, dtype=torch.float, reshape=False):
    output = torch.from_numpy(np.array(arr)).type(dtype).to(device)
    if reshape:
        output = output.reshape(-1, 1)

    return output

def epsilon_greedy(start, end, n_steps, it):
    return max(start - (start - end) * (it / n_steps), end)

main()