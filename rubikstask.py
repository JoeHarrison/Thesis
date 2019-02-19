import rubiks
from feedforwardnetwork import NeuralNetwork
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np


def compute_q_val(network, state, action):
    qactions = network(state)
    return torch.gather(qactions,1,action.view(-1,1))


def compute_target(model, reward, next_state, done, discount_factor,device):
    # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
    m = torch.cat(((discount_factor*torch.max(model(next_state),1)[0]).view(-1,1),torch.zeros(reward.size(), device=device).view(-1, 1)), 1)
    return reward.view(-1, 1) + torch.gather(m, 1, done.long().view(-1, 1))


def train(network, optimizer, batch_size, discount_factor, state, action, reward, next_state, done, device):
    q_val = compute_q_val(network, state, action)

    with torch.no_grad():
        target = compute_target(network, reward, next_state, done, discount_factor, device)

    loss = F.smooth_l1_loss(q_val, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


class RubiksTask(object):
    def __init__(self, batch_size, device, lamarckism = False):
        self.batch_size = batch_size
        self.device = device
        self.lamarckism = lamarckism

        self.difficulty = 1

        self.envs = [rubiks.RubiksEnv(2) for _ in range(self.batch_size)]

    def _increase_difficulty(self):
        self.difficulty += 1

    def evaluate(self, genome, verbose=False):
        # Should always be a genome
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, device=self.device)

        optimizer = optim.Adam(network.parameters(), amsgrad=True)

        max_tries = self.difficulty
        tries = 0
        fitness = torch.zeros(self.batch_size, 1, dtype=torch.float32, device=self.device)
        state = torch.tensor([self.envs[i].reset(self.difficulty) for i in range(self.batch_size)], device=self.device)
        network.reset()

        while tries < max_tries:
            action_probabilities = network(state)
            # Not taking epsilon steps
            actions = torch.max(action_probabilities, 1)[1]

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, actions)])
            done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
            next_state = torch.tensor(next_state, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

            losses = train(network, optimizer, self.batch_size, 0.99, state, actions, reward, next_state, done, self.device)

            # Reset each state that is done
            next_state = torch.tensor([env.reset(self.difficulty) if d else s.tolist() for env, s, d in zip(self.envs, next_state, done)], dtype=torch.float32, device=self.device)

            state = next_state
            fitness += done
            tries += 1

        if self.lamarckism:
            genome.weights_to_genotype(network)

        fitness = float((fitness > 0).sum().item()) / self.batch_size

        if fitness > 0.75:
            self._increase_difficulty()

        return {'fitness': fitness, 'info': self.difficulty}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.99)
