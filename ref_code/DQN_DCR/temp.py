import torch
import torch.nn as nn
import torch.optim as optim


class NET(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, device):
        super(NET, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.device = device
        self.embed = nn.Embedding(4, 3)
        self.fc = nn.Sequential(
            nn.Linear(self.n_states, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )

    def forward(self, inputs):
        inputs = self.embed(inputs)
        inputs_ = torch.zeros(1, 3, 6, 11, device=self.device, dtype=torch.float32)
        for i in range(inputs_.shape[1]):
            for j in range(inputs_.shape[2]):
                inputs_[:, i, j, :] = inputs[:, j, :, i]
        outputs = self.fc(inputs_)

        return outputs


net = NET(11, 1, 64, 'cuda').to('cuda')
labels = torch.rand(1, 3, 6, 1, device='cuda')

for i in range(10):
    state = torch.round(torch.rand(1, 6, 11) * 3).long().to('cuda')
    optimizer = optim.Adam(net.parameters(), lr=1e-4)  # 优化器
    values = net(state)
    optimizer.zero_grad()
    loss = nn.MSELoss()(labels, values)
    loss.backward()
    optimizer.step()
    w = list(net.embed.named_parameters())
    print(w[0])
