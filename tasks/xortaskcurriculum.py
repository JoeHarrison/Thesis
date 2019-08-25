import numpy as np
import random
import torch
from torch import optim
from feedforwardnetwork import NeuralNetwork
from collections import defaultdict

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


class XORTaskCurriculum(object):
    def __init__(self, batch_size, device, baldwin, lamarckism, use_single_activation_function):
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

        self.use_single_activation_function = use_single_activation_function

    def generate_both(self, p):
        r_idx = np.random.randint(0,4)
        if random.random() < p:
            return self.INPUTSXOR[r_idx], self.TARGETSXOR[r_idx]
        else:
            return self.INPUTSOR[r_idx], self.TARGETSOR[r_idx]

    def backprop(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)

        optimiser = torch.optim.Adam(network.parameters())
        criterion = torch.nn.MSELoss()

        for epoch in range(1000):
            network.reset()
            optimiser.zero_grad()
            if self.difficulty == 0:
                Xs, Ys = self.generate_both(0.0)

                outputs = network(Xs)

                loss = criterion(outputs, Ys)

                loss.backward()

                optimiser.step()
            else:
                Xs, Ys = self.generate_both(1.0)

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
                    network = NeuralNetwork(genome, batch_size=4, device=self.device, use_single_activation_function=self.use_single_activation_function)
                network.batch_size = 4
            else:
                network = NeuralNetwork(genome, batch_size=4, device=self.device, use_single_activation_function=self.use_single_activation_function)
        network.reset()

        # Evaluation
        if self.difficulty == 0:

            Xs = self.INPUTSOR
            Ys = self.TARGETSOR

            outputs = network(Xs)

            loss = 1.0/(1.0+torch.sqrt(self.criterion(outputs, Ys)))

            if loss >= 0.99:
                print('----------------------')

                tmp_neurons = defaultdict(int)
                req = required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)
                for neuron in genome.neuron_genes:
                    if neuron[0] in req and neuron[0] not in genome.output_keys:
                        layer = neuron[3]
                        tmp_neurons[layer] += 1

                for key in sorted(list(tmp_neurons.keys())):
                    print('Layer: ', key, 'number of neurons: ', tmp_neurons[key])

                if len(list(tmp_neurons.keys()))==0:
                    print('No hidden layers')

                print('Number enabled connections', np.sum([1 for conn in genome.connection_genes.values() if conn[4]]))
                print('Generation', generation)
                print(loss)

                network.reset()

                print(1.0/(1.0+torch.sqrt(self.criterion(self.TARGETSXOR, network(self.INPUTSXOR)))))
                self.difficulty += 1
                self.difficulty_set = True

            if generation >= 500 and not self.difficulty_set:
                print('----------------------')
                print('0.99 not reached')

                tmp_neurons = defaultdict(int)
                req = required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)
                for neuron in genome.neuron_genes:
                    if neuron[0] in req and neuron[0] not in genome.output_keys:
                        layer = neuron[3]
                        tmp_neurons[layer] += 1

                for key in sorted(list(tmp_neurons.keys())):
                    print('Layer: ', key, 'number of neurons: ', tmp_neurons[key])

                if len(list(tmp_neurons.keys()))==0:
                    print('No hidden layers')

                print('Number enabled connections', np.sum([1 for conn in genome.connection_genes.values() if conn[4]]))
                print('Generation', generation)

                print(loss)

                network.reset()

                print(1.0/(1.0+torch.sqrt(self.criterion(self.TARGETSXOR, network(self.INPUTSXOR)))))
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
            return int(self.evaluate(network, self.generation)['fitness'] > 0.99)
        else:
            return 0
