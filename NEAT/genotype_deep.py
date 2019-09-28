import random
import numpy as np
from itertools import product
from copy import deepcopy
import time

import torch

from naming.namegenerator import NameGenerator


class Genotype_Deep(object):
    def __init__(self,
                 new_individual_name,
                 inputs=144,
                 outputs=6,
                 nonlinearities=['tanh', 'relu', 'sigmoid', 'identity', 'elu'],
                 p_add_node=0.1,
                 p_delete_node=0.03,
                 p_mutate_node=0.8,
                 p_add_interlayer_node=0.1,
                 distance_excess_weight=1.0,
                 distance_disjoint_weight=1.0,
                 distance_weight=0.4):

        # Name the new individual
        self.name = next(new_individual_name)
        self.specie = None

        self.inputs = inputs
        self.outputs = outputs
        self.nonlinearities = nonlinearities

        self.p_add_node = p_add_node
        self.p_delete_node = p_delete_node
        self.p_mutate_node = p_mutate_node

        self.distance_excess_weight = distance_excess_weight
        self.distance_disjoint_weight = distance_disjoint_weight
        self.distance_weight = distance_weight

        self.p_add_interlayer_node = p_add_interlayer_node

        self.nodes = []

        self._initialise_topology()

        self.rl_training = False

    def _initialise_hyperparameters(self):
        pass

    def _np_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _update_hyperparameters(self):
        pass

    def change_specie(self, specie):
        self.specie = specie

    def _initialise_topology(self):
        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'num_nodes': self.inputs, 'activation_function': nonlinearity, 'weights': None, 'biases': None})

        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'num_nodes': self.outputs, 'activation_function': nonlinearity, 'weights': None, 'biases': None,
             'p_mutate_interlayer_weights': 0.1})

    def add_node(self):
        position = random.randint(2, len(self.nodes))
        nonlinearity = random.choice(self.nonlinearities)
        num_nodes = random.randint(1, 128)

        self.nodes.insert(position, {'num_nodes': num_nodes, 'activation_function': nonlinearity, 'weights': None,
                                     'biases': None, 'p_mutate_interlayer_weights': 0.1})

    def delete_node(self):
        pass

    def mutate_node(self):
        if random.random() < self.p_add_interlayer_node and len(self.nodes) - 1 > 2:
            position = random.randint(2, len(self.nodes) - 1)
            self.nodes[position]['num_nodes'] += random.randint(-5, 5)
            self.nodes[position]['activation_function'] = random.choice(self.nonlinearities)

        for i in range(1, len(self.nodes)):
            if random.random() < self.nodes[i]['p_mutate_interlayer_weights']:
                if self.nodes[i]['weights'] is not None:
                    t = torch.rand_like(self.nodes[i]['weights'])
                    self.nodes[i]['weights'] += (t < 0.1).float() * torch.randn_like(self.nodes[i]['weights']) * 0.1

    def mutate(self):
        if len(self.nodes) == 2:
            self.add_node()
        if random.random() < self.p_add_node:
            self.add_node()
        elif random.random() < self.p_mutate_node:
            self.mutate_node()

    def recombinate(self, other):
        child = deepcopy(self)
        child.name = self.name[:3] + other.name[3:]
        child.nodes = []

        min_layers = min(len(self.nodes), len(other.nodes))
        max_layers = max(len(self.nodes), len(other.nodes))

        for i in range(max_layers):
            node = None
            if i < min_layers:
                node = random.choice((self.nodes[i], other.nodes[i]))
            else:
                try:
                    node = self.nodes[i]
                except IndexError:
                    node = other.nodes[i]
            child.nodes.append(deepcopy(node))

        return child

    def distance(self, other):
        e = 0.0
        d = 0.0
        w = 0.0
        m = 0

        max_layers = max(len(self.nodes), len(other.nodes))
        min_layers = min(len(self.nodes), len(other.nodes))

        for i in range(1, min_layers):
            min_row = min(self.nodes[i]['weights'].size(0), other.nodes[i]['weights'].size(0))
            min_column = min(self.nodes[i]['weights'].size(1), other.nodes[i]['weights'].size(1))

            m += min_row * min_column
            w += torch.abs(
                self.nodes[i]['weights'].data[:min_row, :min_column] - other.nodes[i]['weights'].data[:min_row,
                                                                       :min_column]).sum().item()

            e += self.nodes[i]['weights'].data.sum() - self.nodes[i]['weights'].data[:min_row, :min_column].sum().item()
            e += other.nodes[i]['weights'].data.sum() - other.nodes[i]['weights'].data[:min_row, :min_column].sum().item()

        for i in range(min_layers, max_layers):
            if len(self.nodes) == len(other.nodes):
                break
            elif len(other.nodes) == max_layers:
                d += 1
            else:
                d += 1

        w = (w / m) if m > 0 else w

        return w * self.distance_weight + e * self.distance_excess_weight + d * self.distance_disjoint_weight


if __name__ == "__main__":
    first_name_generator = NameGenerator('../naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()

    g = Genotype_Deep(new_individual_name, 144, 6)

    from feedforwardnetwork_deep import NeuralNetwork_Deep

    g.add_node()

    n = NeuralNetwork_Deep()
    n.create_network(g)

    g.add_node()

    n = NeuralNetwork_Deep()
    n.create_network(g)

    g.mutate_node()

    n = NeuralNetwork_Deep()
    n.create_network(g)

    print(n)

    print(n.model[0].weight.size())
    print('b', n.model[0].bias.size())
    print(n.model[2].weight.size())
    print('b', n.model[2].bias.size())
    print(n.model[4].weight.size())
    print('b', n.model[4].bias.size())

    print(n(torch.tensor([0.1] * 144)))
