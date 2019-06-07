import numpy as np
import random
import torch
from torch import optim
from feedforwardnetwork import NeuralNetwork

def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s)..
    """
    required = set(outputs)
    s = set(outputs)
    while 1:
        # Find nodes not in S whose output is consumed by a node in s.
        t = set(a for (a, b) in connections if b in s and a not in s)

        if not t:
            break

        layer_nodes = set(x for x in t if x not in inputs)
        if not layer_nodes:
            break

        required = required.union(layer_nodes)
        s = s.union(t)

    return list(required)


def generate_zero():
    return random.uniform(0, 49) / 100


def generate_one():
    return random.uniform(50, 100) / 100

def generate_both(num_data_points, p):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        if random.random() < p:
            Xs.append([generate_zero(), generate_zero(), 0]); Ys.append([0])
            # or(1, 0) -> 1
            Xs.append([generate_one(), generate_zero(), 0]); Ys.append([1])
            # or(0, 1) -> 1
            Xs.append([generate_zero(), generate_one(), 0]); Ys.append([1])
            # or(1, 1) -> 1
            Xs.append([generate_one(), generate_one(), 0]); Ys.append([1])
        else:
            # xor(0, 0) -> 0
            Xs.append([generate_zero(), generate_zero(), 1]); Ys.append([0])
            # xor(1, 0) -> 1
            Xs.append([generate_one(), generate_zero(), 1]); Ys.append([1])
            # xor(0, 1) -> 1
            Xs.append([generate_zero(), generate_one(), 1]); Ys.append([1])
            # xor(1, 1) -> 0
            Xs.append([generate_one(), generate_one(), 1]); Ys.append([0])
    return Xs, Ys


def generate_or_XY(num_data_points):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        # or(0, 0) -> 0
        Xs.append([generate_zero(), generate_zero(), 0]); Ys.append([0])
        # or(1, 0) -> 1
        Xs.append([generate_one(), generate_zero(), 0]); Ys.append([1])
        # or(0, 1) -> 1
        Xs.append([generate_zero(), generate_one(), 0]); Ys.append([1])
        # or(1, 1) -> 1
        Xs.append([generate_one(), generate_one(), 0]); Ys.append([1])
    return Xs, Ys

def generate_xor_XY(num_data_points):
    Xs, Ys = [], []
    for _ in range(num_data_points):
        # xor(0, 0) -> 0
        Xs.append([generate_zero(), generate_zero(), 1]); Ys.append([0])
        # xor(1, 0) -> 1
        Xs.append([generate_one(), generate_zero(), 1]); Ys.append([1])
        # xor(0, 1) -> 1
        Xs.append([generate_zero(), generate_one(), 1]); Ys.append([1])
        # xor(1, 1) -> 0
        Xs.append([generate_one(), generate_one(), 1]); Ys.append([0])
    return Xs, Ys

class XORTaskCurriculum(object):
    def __init__(self, batch_size, device, baldwin, lamarckism):
        self.INPUTSOR = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], device=device)
        self.TARGETSOR = torch.tensor([[0.0], [1.0], [1.0], [1.0]], device=device)

        self.INPUTSXOR = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], device=device)
        self.TARGETSXOR = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.device = device
        self.generation = 0
        self.difficulty = 0
        self.difficulty_set = False

        self.baldwin = baldwin
        self.lamarckism = lamarckism

    def backprop(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)

        optimiser = torch.optim.Adam(network.parameters(), amsgrad=True)
        criterion = torch.nn.MSELoss()

        for epoch in range(1000):
            network.reset()
            optimiser.zero_grad()
            if self.difficulty == 0:
                Xs, Ys =generate_both(int(self.batch_size/4), 0.1)
                Xs = torch.tensor(Xs, device=self.device)
                Ys = torch.tensor(Ys, dtype=torch.float, device=self.device)
                # if np.random.rand() < 0.1:
                #     Xs, Ys = generate_xor_XY(int(self.batch_size/4))
                #     Xs = torch.tensor(Xs, device=self.device)
                #     Ys = torch.tensor(Ys, dtype=torch.float, device=self.device)
                # else:
                #     Xs, Ys = generate_or_XY(int(self.batch_size/4))
                #     Xs = torch.tensor(Xs, device=self.device)
                #     Ys = torch.tensor(Ys, dtype=torch.float, device=self.device)

                outputs = network(Xs)

                loss = criterion(outputs, Ys)

                loss.backward()

                optimiser.step()
            else:
                Xs, Ys =generate_both(int(self.batch_size/4), 0.9)
                Xs = torch.tensor(Xs, device=self.device)
                Ys = torch.tensor(Ys, dtype=torch.float, device=self.device)
                # if np.random.rand() < 0.1:
                #     Xs, Ys = generate_or_XY(int(self.batch_size/4))
                #     Xs = torch.tensor(Xs, device=self.device)
                #     Ys = torch.tensor(Ys, dtype=torch.float, device=self.device)
                # else:
                #     Xs, Ys = generate_xor_XY(int(self.batch_size/4))
                #     Xs = torch.tensor(Xs, device=self.device)
                #     Ys = torch.tensor(Ys, dtype=torch.float, device=self.device)

                outputs = network(Xs)

                loss = criterion(outputs, Ys)

                loss.backward()

                optimiser.step()

        genome.rl_training = False

        if self.lamarckism:
            genome.weights_to_genotype(network)
            return genome
        else:
            return network



    def evaluate(self, genome, generation):


        if generation > self.generation:
            self.difficulty_set = False
            self.generation = generation

        if not isinstance(genome, NeuralNetwork):
            if genome.rl_training and self.baldwin:
                network = self.backprop(genome)
                if not isinstance(network, NeuralNetwork):
                    network = NeuralNetwork(genome, batch_size=4, device=self.device)
                network.batch_size = 4
            else:
                network = NeuralNetwork(genome, batch_size=4, device=self.device)
        network.reset()

        # Evaluation
        if self.difficulty == 0:

            Xs = self.INPUTSOR
            Ys = self.TARGETSOR

            outputs = network(Xs)

            loss = 1.0/(1.0+torch.sqrt(self.criterion(outputs, Ys)))



            if loss >= 0.95:
                print('----------------------')
                print('Number neurons', len(required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)) + len(genome.input_keys) + len(genome.output_keys))
                print('Number enabled connections', np.sum([1 for conn in genome.connection_genes.values() if conn[4]]))
                print('Generation', generation)
                self.difficulty += 1
                self.difficulty_set = True

            if generation >= 500 and not self.difficulty_set:
                print('----------------------')
                print('0.95 not reached')
                print('Number neurons', len(required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)) + len(genome.input_keys) + len(genome.output_keys))
                print('Number enabled connections', np.sum([1 for conn in genome.connection_genes.values() if conn[4]]))
                print('Generation', generation)
                self.difficulty += 1
                self.difficulty_set = True
        else:
            Xs = self.INPUTSXOR
            Ys = self.TARGETSXOR

            outputs = network(Xs)

            loss = 1.0/(1.0+self.criterion(outputs, Ys))

        if self.difficulty_set:
            return {'fitness': loss.item(), 'info': self.difficulty - 1, 'generation': generation}
        else:
            return {'fitness': loss.item(), 'info': self.difficulty, 'generation': generation}

    def solve(self, network):
        if self.difficulty > 0:
            return int(self.evaluate(network, self.generation)['fitness'] > 0.95)
        else:
            return 0
