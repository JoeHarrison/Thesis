import numpy as np
import random
import torch
from torch import optim
from feedforwardnetwork import NeuralNetwork

class XORTask(object):
    def __init__(self, batch_size, device):
        self.INPUTS = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
        self.TARGETS = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)
        network.reset()

        outputs = network(self.INPUTS)

        loss = 1.0/(1.0+torch.sqrt(self.criterion(outputs, self.TARGETS)))
        
        return {'fitness': loss.item(), 'info': 0}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.9)
