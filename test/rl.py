import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from feedforwardnetwork import NeuralNetwork
import numpy as np

from memory import ReplayMemory

import gym

class QNetwork(nn.Module):
    """Network that maps states to q_values of actions."""
    def __init__(self, input_size, num_actions=2):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, num_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x

class DQNAgent:
    def __init__(self, model, discount_factor, memory, batch_size, device):
        self.model = model.to(device)
        self.memory = memory
        self.optimiser = optim.Adam(self.model.parameters(), amsgrad=True)
        self.batch_size = batch_size
        self.device = device
        self.discount_factor = discount_factor

    def select_actions(self, state, epsilon):
        with torch.no_grad():
            action_probabilities = self.model(state)

            max_action = torch.max(action_probabilities, 1)[1]
            rand_action = torch.randint(0, action_probabilities.size(1), size=([action_probabilities.size(0)]), device=self.device, dtype=torch.long)
            bernoulli = torch.bernoulli(torch.ones(action_probabilities.size(0), device=self.device)*epsilon).long().view(-1, 1)
            return torch.gather(torch.cat((max_action.view(-1, 1), rand_action.view(-1, 1)), 1), 1, bernoulli).view(-1)

    def compute_q_val(self, state, action):
        qactions = self.model(state)

        return torch.gather(qactions, 1, action.view(-1, 1))

    def compute_target(self, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
        m = torch.cat(((self.discount_factor*torch.max(self.model(next_state), 1)[0]).view(-1, 1), torch.zeros(reward.size(), device=self.device).view(-1, 1)), 1)

        return reward.view(-1, 1) + torch.gather(m, 1, done.long().view(-1, 1))

    def train(self):
        state, action, next_state, reward, done = zip(*self.memory.sample(self.batch_size))

        state = torch.cat(([s.view(1, -1) for s in state]), 0)
        action = torch.cat(([a.view(-1) for a in action]), 0)
        next_state = torch.cat(([ns.view(1, -1) for ns in next_state]), 0)
        reward = torch.cat(([r.view(-1) for r in reward]), 0)
        done = torch.cat(([d.view(1, -1) for d in done]), 0)

        q_val = self.compute_q_val(state, action)

        target = self.compute_target(reward, next_state, done)

        criterion = nn.MSELoss()

        loss = criterion(q_val, target.detach())

        self.optimiser.zero_grad()

        loss.backward()

        self.optimiser.step()

        return loss.item()

# if __name__ == '__main__':
#     batch_size = 64
#     envs = [gym.make('CartPole-v0') for _ in range(batch_size)]
#     epochs = 1000
#     max_tries = 200
#     discount_factor = 0.99
#
#     model = QNetwork(4, 2)
#     memory = ReplayMemory(100000)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#
#     agent = DQNAgent(model, discount_factor, memory, 2, device)
#
#     for epoch in range(epochs):
#         state = torch.tensor([envs[i].reset() for i in range(batch_size)], device=device, dtype=torch.float32)
#         total_done = torch.zeros([batch_size,1], device=device)
#         for i in range(max_tries):
#             action = agent.select_actions(state, 0.01)
#
#             next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(envs, action)])
#             done = torch.tensor(done, dtype=torch.float32, device=device).view(-1, 1)
#             next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
#             reward = torch.tensor(reward, dtype=torch.float32, device=device)
#
#             total_done += done
#
#             agent.memory.push(state, action, next_state, reward, done)
#
#             agent.train()
#
#             state = torch.tensor([env.reset() if d else s.tolist() for env, s, d in zip(envs, next_state, done)], dtype=torch.float32, device=device)
#
#         # state = torch.tensor([envs[i].reset() for i in range(batch_size)], device=device, dtype=torch.float32)
#         # total_done = torch.zeros([batch_size, 1], device=device)
#         # for i in range(max_tries):
#         #     action = agent.select_actions(state, 0.0)
#         #
#         #     next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(envs, action)])
#         #     done = torch.tensor(done, dtype=torch.float32, device=device).view(-1, 1)
#         #     next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
#         #
#         #     total_done += done
#         #
#         #     state = torch.tensor([env.reset() if d else s.tolist() for env, s, d in zip(envs, next_state, done)], dtype=torch.float32, device=device)
#
#         print(epoch, (max_tries-(total_done.sum().item()/batch_size))/max_tries)
#
#     step_list = []
#     for i in range(100):
#         done = False
#         envs = [gym.make('CartPole-v0')]
#         state = torch.tensor([envs[0].reset()], device=device, dtype=torch.float32)
#         steps = 0
#         while not done:
#             action = agent.select_actions(state, 0.0)
#
#             next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(envs, action)])
#             done = done[0]
#             next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
#
#             state = next_state
#             steps += 1
#             # envs[0].render()
#         step_list.append(steps)
#         # envs[0].close()
#     print(np.array(step_list).sum()/100)

class CartpoleTask(object):
    def __init__(self, batch_size, device, discount_factor, memory, lamarckism=False):
        self.batch_size = batch_size
        self.device = device
        self.lamarckism = lamarckism
        self.envs = [gym.make('CartPole-v0') for _ in range(self.batch_size)]
        self.discount_factor = discount_factor
        self.memory = memory

    def evaluate(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, device=self.device)

        agent = DQNAgent(network, self.discount_factor, self.memory, self.batch_size, self.device)

        max_tries = 200

        state = torch.tensor([self.envs[i].reset() for i in range(self.batch_size)], device=self.device, dtype=torch.float32)
        total_done = torch.zeros([self.batch_size,1], device=self.device)
        for i in range(max_tries):
            action = agent.select_actions(state, 0.01)

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, action)])
            done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

            total_done += done

            agent.memory.push(state, action, next_state, reward, done)

            agent.train()

            state = torch.tensor([env.reset() if d else s.tolist() for env, s, d in zip(self.envs, next_state, done)], dtype=torch.float32, device=device)

        fitness = (max_tries - (total_done.sum().item()/self.batch_size))/max_tries

        if self.lamarckism:
            genome.weights_to_genotype(network)

        return {'fitness': fitness, 'info': 0}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.99)
