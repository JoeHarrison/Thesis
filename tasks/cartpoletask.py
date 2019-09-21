import random
import torch
import torch.nn as nn
from feedforwardnetwork import NeuralNetwork
from reinforcement_learning.replay_memories import ReplayMemory
import rubiks2
import time
import copy
import numpy as np
import gym

class CartpoleTask(object):
    def __init__(self, batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum, use_single_activation_function=False):
        self.criterion = torch.nn.SmoothL1Loss()
        self.batch_size = batch_size
        self.device = device

        self.generation = 0

        self.memory = memory
        self.discount_factor = discount_factor

        self.baldwin = baldwin
        self.lamarckism = lamarckism
        self.memory = ReplayMemory(14*10000)

        self.use_single_activation_function = use_single_activation_function

        self.target_network = None

        self.env = gym.make("CartPole-v1")

    def compute_q_val(self, model, state, action):
        q_actions = model(state)

        return torch.gather(q_actions, 1, action.view(-1, 1))

    def compute_target(self, model, reward, next_state, done):
        return reward + self.discount_factor * model(next_state).max(1)[0] * (1-done)

    def compute_target_ddqn(self, model, target_model, reward, next_state, done):
        return reward.view(-1, 1) + self.discount_factor * torch.gather(target_model(next_state), 1, model(next_state).max(1)[1].view(-1, 1)) * (1-done).view(-1, 1)

    def b(self, network, optimiser):
        if len(self.memory) < 128:
            return

        batch, _, _ = self.memory.sample(32, 1, self.device)
        state, action, next_state, reward, done = zip(*batch)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.float32, device=self.device)

        network.reset()
        q_val = self.compute_q_val(network, state, action)

        network.reset()
        with torch.no_grad():
            target = self.compute_target_ddqn(network, self.target_network, reward, next_state, done).view(-1, 1)

        loss = self.criterion(q_val, target)

        optimiser.zero_grad()
        loss.backward()

        optimiser.step()

        return loss

    def backprop(self, network):
        # optimiser = torch.optim.Adam(network.parameters())
        optimiser = torch.optim.RMSprop(params=network.parameters(), momentum=0.95, lr=0.0001)

        if not self.target_network:
            self.target_network = copy.deepcopy(network)

        for i in range(10000):
            loss = self.b(network, optimiser)

            if i % 1000 == 0 and i > 0:
                network.reset()
                self.target_network = copy.deepcopy(network)

        return network

    def evaluate(self, genome, generation):
        if genome.rl_training and self.baldwin:
            print(genome.name, genome.specie, id(genome))
            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)
            network = self.backprop(network)
            if self.lamarckism:
                genome.weights_to_genotype(network)
            genome.rl_training = False

        network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)
        total_reward = 0.0

        done = 0.0
        tries = 0
        max_tries = 1000

        state = self.env.reset()

        while tries < max_tries and not done:
            network.reset()
            q_values = network(torch.tensor(state, dtype=torch.float32, device=self.device))
            action = q_values.max(1)[1].view(1, 1).item()
            next_state, reward, done, info = self.env.step(int(action))
            self.memory.push((state, action, next_state, reward, done))
            state = next_state
            tries += 1

            total_reward += reward


        return {'fitness': total_reward, 'info': 0, 'generation': generation, 'reset_species': 0}

    def solve(self, network):
        if self.difficulty == 14:
            return int(self.evaluate(network, self.generation)['fitness'] > 0.95)
        else:
            return 0
