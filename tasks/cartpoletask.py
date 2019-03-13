import torch
import torch.nn as nn
import torch.optim as optim
from feedforwardnetwork import NeuralNetwork

import gym

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

class CartpoleTask(object):
    def __init__(self, batch_size, device, discount_factor, memory, lamarckism=False, rl=False):
        self.batch_size = batch_size
        self.device = device
        self.lamarckism = lamarckism
        self.rl = rl
        self.envs = [gym.make('CartPole-v0') for _ in range(self.batch_size)]
        self.discount_factor = discount_factor
        self.memory = memory

    def evaluate(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)
        network.reset()
        agent = DQNAgent(network, self.discount_factor, self.memory, self.batch_size, self.device)

        max_tries = 100

        state = torch.tensor([self.envs[i].reset() for i in range(self.batch_size)], device=self.device, dtype=torch.float32)
        total_done = torch.zeros([self.batch_size, 1], device=self.device)

        if self.rl:
            for i in range(max_tries):
                action = agent.select_actions(state, 0.01)

                next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, action)])
                done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

                total_done += done

                agent.memory.push(state, action, next_state, reward, done)

                agent.train()

                state = torch.tensor([env.reset() if d else s.tolist() for env, s, d in zip(self.envs, next_state, done)], dtype=torch.float32, device=self.device)

        if self.lamarckism:
            genome.weights_to_genotype(network)

        network = NeuralNetwork(genome, batch_size=1, device=self.device)

        agent = DQNAgent(network, self.discount_factor, self.memory, 1, self.device)

        max_tries = 200
        tries = 0

        for _ in range(10):
            network.reset()
            state = torch.tensor([self.envs[0].reset()], device=self.device, dtype=torch.float32)
            for _ in range(max_tries):
                action = agent.select_actions(state, 0.0)

                next_state, reward, done, info = (self.envs[0].step(int(action[0])))
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                state = next_state

                tries += 1

                if done:
                    break

        fitness = (tries/10)/max_tries

        return {'fitness': fitness, 'info': 0}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.999)
