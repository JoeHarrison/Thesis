import numpy as np
import random
import torch
from feedforwardnetwork import NeuralNetwork

class XORTask(object):
    def __init__(self, batch_size, device):
        self.INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        self.TARGETS = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
        self.EPSILON = 1e-100
        self.criterion = torch.nn.MSELoss()
        self.device = device

    def evaluate(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, device=self.device)

        outputs = network(self.INPUTS)

        loss = 1.0/(1.0+torch.sqrt(self.criterion(outputs, self.TARGETS)))

        return {'fitness': loss.item(), 'info': 0}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.9)
