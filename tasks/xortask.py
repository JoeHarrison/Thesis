
import numpy as np
import random
import torch
from torch import optim
from feedforwardnetwork import NeuralNetwork
from copy import deepcopy

class XORTask(object):
    def __init__(self, batch_size, device):
        self.INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        self.TARGETS = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.device = device
        self.generation = 0

    def evaluate(self, genome, generation):

        if generation > self.generation:
            self.difficulty_set = False
            self.generation = generation

        network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)

        if genome.rl_training:
            optimiser = torch.optim.Adam(network.parameters())

            for i in range(1000):
                network.reset()

                r = random.randint(0, 3)

                outputs = network(self.INPUTS[r])

                loss = self.criterion(outputs, self.TARGETS[r])

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()


            genome.weights_to_genotype(network)

            genome.rl_training = False



        network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)

        network.reset()

        outputs = network(self.INPUTS)

        loss = 1.0/(1.0+torch.sqrt(self.criterion(outputs, self.TARGETS)))

        return {'fitness': loss.item(), 'info': 0, 'reset_species': 0}

    def solve(self, genome):
        return int(genome.stats['fitness'] > 0.9)
