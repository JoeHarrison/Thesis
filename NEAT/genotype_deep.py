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
                 p_change_layer_nonlinearity=0.01,
                 distance_excess_weight=1.0,
                 distance_disjoint_weight=1.0,
                 distance_weight=0.4,
                 initial_p_mutate_interlayer_weights=0.1,
                 min_initial_nodes=1,
                 max_initial_nodes=128):

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
        self.p_change_layer_nonlinearity = p_change_layer_nonlinearity

        self.initial_p_mutate_interlayer_weights = initial_p_mutate_interlayer_weights
        self.min_initial_nodes = min_initial_nodes
        self.max_initial_nodes = max_initial_nodes

        self.nodes = []

        self._initialise_topology()

        self.rl_training = False

    def change_specie(self, specie):
        self.specie = specie

    def _initialise_topology(self):
        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'num_nodes': self.inputs, 'activation_function': nonlinearity, 'weights': None, 'biases': None,
             'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights})

        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'num_nodes': self.outputs, 'activation_function': nonlinearity, 'weights': None, 'biases': None,
             'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights})

    def add_node(self):
        position = random.randint(2, len(self.nodes))
        nonlinearity = random.choice(self.nonlinearities)
        num_nodes = random.randint(self.min_initial_nodes, self.max_initial_nodes)

        self.nodes.insert(position, {'num_nodes': num_nodes, 'activation_function': nonlinearity, 'weights': None,
                                     'biases': None,
                                     'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights})

    def delete_node(self):
        pass

    def mutate_node(self):
        if random.random() < self.p_add_interlayer_node and len(self.nodes) - 1 > 2:
            position = random.randint(2, len(self.nodes) - 1)
            # Ensure that no more nodes can be deleted than there exist.
            self.nodes[position]['num_nodes'] += random.randint(-min(self.nodes[position]['num_nodes'] - 1, 1), 1)

        if random.random() < self.p_change_layer_nonlinearity:
            position = random.randint(1, len(self.nodes) - 1)
            self.nodes[position]['activation_function'] = random.choice(self.nonlinearities)

        for i in range(1, len(self.nodes)):
            if random.random() < self.nodes[i]['p_mutate_interlayer_weights']:
                if self.nodes[i]['weights'] is not None:
                    t = torch.rand_like(self.nodes[i]['weights'])
                    self.nodes[i]['weights'] += (t < 0.01).float() * torch.randn_like(self.nodes[i]['weights']) * 0.1

    def mutate(self):
        if random.random() < self.p_add_node:
            self.add_node()
        elif random.random() < self.p_mutate_node:
            self.mutate_node()

    def recombinate(self, other):
        best = self if self.stats['fitness'] > other.stats['fitness'] else other
        worst = self if self.stats['fitness'] < other.stats['fitness'] else other

        child = deepcopy(best)
        child.name = best.name[:3] + worst.name[3:]
        child.nodes = []

        min_layers = min(len(best.nodes), len(worst.nodes))
        max_layers = max(len(best.nodes), len(worst.nodes))

        for i in range(max_layers):
            node = None
            if i < min_layers:
                node = random.choice((best.nodes[i], worst.nodes[i]))
            else:
                try:
                    node = best.nodes[i]
                except IndexError:
                    node = worst.nodes[i]
            child.nodes.append(deepcopy(node))

        return child

    def distance(self, other):
        e = 0.0
        d = 0.0
        w = 0.0
        m_w = 0

        max_layers = max(len(self.nodes), len(other.nodes))
        min_layers = min(len(self.nodes), len(other.nodes))

        for i in range(1, min_layers):

            min_row = min(self.nodes[i]['weights'].size(0), other.nodes[i]['weights'].size(0))
            min_column = min(self.nodes[i]['weights'].size(1), other.nodes[i]['weights'].size(1))

            m_w += min_row * min_column
            w += torch.abs(
                self.nodes[i]['weights'].data[:min_row, :min_column] - other.nodes[i]['weights'].data[:min_row,
                                                                       :min_column]).sum().item()

            e += self.nodes[i]['weights'].data.size(0) - self.nodes[i]['weights'].data[:min_row, :min_column].size(0)
            e += self.nodes[i]['weights'].data.size(1) - self.nodes[i]['weights'].data[:min_row, :min_column].size(1)
            e += other.nodes[i]['weights'].data.size(0) - other.nodes[i]['weights'].data[:min_row, :min_column].size(0)
            e += other.nodes[i]['weights'].data.size(1) - other.nodes[i]['weights'].data[:min_row, :min_column].size(1)

        for i in range(min_layers, max_layers):
            if len(self.nodes) == len(other.nodes):
                break
            else:
                d += 1

        w = (w / m_w) if m_w > 0 else w

        return w * self.distance_weight + e * self.distance_excess_weight + d * self.distance_disjoint_weight


if __name__ == "__main__":
    first_name_generator = NameGenerator('../naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()

    g1 = Genotype_Deep(new_individual_name, 144, 6)
    g2 = Genotype_Deep(new_individual_name, 144, 6)

    from feedforwardnetwork_deep import NeuralNetwork_Deep

    n1 = NeuralNetwork_Deep()
    n2 = NeuralNetwork_Deep()

    n1.create_network(g1)
    n2.create_network(g2)

    g1.add_node()
    g2.add_node()

    n1 = NeuralNetwork_Deep()
    n2 = NeuralNetwork_Deep()

    n1.create_network(g1)
    n2.create_network(g2)


    print(g1.recombinate(g2).nodes)






