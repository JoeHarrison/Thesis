import random
import numpy as np
from itertools import product
from copy import deepcopy
import time

import torch

from naming.namegenerator import NameGenerator
from tasks.rubikstaskRL_deep import RubiksTask_Deep


class Genotype_Deep(object):
    def __init__(self,
                 new_individual_name,
                 inputs=144,
                 outputs=6,
                 nonlinearities=['tanh', 'relu', 'sigmoid', 'identity', 'elu'],
                 feedforward=True,
                 p_add_node=0.1,
                 p_add_connection=0.3,
                 p_mutate_weight=0.8,
                 p_mutate_bias=0.2,
                 p_mutate_size=0.1,
                 p_mutate_non_linearity=0.2,
                 p_add_interlayer_node=0.1,
                 p_change_layer_nonlinearity=0.01,
                 initial_p_mutate_weights=0.03,
                 initial_sigma=0.1,
                 initial_p_mutate_biases=0.03,
                 distance_excess_weight=1.0,
                 distance_disjoint_weight=1.0,
                 distance_weight=0.4,
                 initial_p_mutate_interlayer_weights=0.1,
                 min_initial_nodes=32,
                 max_initial_nodes=256):

        # Variables pertaining to individual naming
        self.name = next(new_individual_name)
        self.specie = None

        # Variables pertaining to network structure
        self.inputs = inputs
        self.outputs = outputs
        self.nonlinearities = nonlinearities
        self.feedforward = feedforward

        self.initial_sigma = initial_sigma
        self.sigma_epsilon = 0.01

        # Variables pertaining to the three large ways of changing the network
        self.p_add_node = p_add_node
        self.p_add_connection = p_add_connection
        self.p_mutate_weight = p_mutate_weight
        self.p_mutate_bias = p_mutate_bias
        self.p_mutate_non_linearity = p_mutate_non_linearity
        self.p_mutate_size = p_mutate_size

        # Distance variables
        self.distance_excess_weight = distance_excess_weight
        self.distance_disjoint_weight = distance_disjoint_weight
        self.distance_weight = distance_weight

        # Mutating nodes
        self.p_add_interlayer_node = p_add_interlayer_node
        self.p_change_layer_nonlinearity = p_change_layer_nonlinearity

        # Node specific parameters
        self.initial_p_mutate_interlayer_weights = initial_p_mutate_interlayer_weights
        self.min_initial_nodes = min_initial_nodes
        self.max_initial_nodes = max_initial_nodes

        # Connection specific parameters
        self.initial_p_mutate_weights = initial_p_mutate_weights
        self.initial_p_mutate_biases = initial_p_mutate_biases
        self.nodes = []

        # Innovation number, input node, output node, enabled, Weights, Biases
        self.connections = {}

        self.global_hyperparameters = {}

        self._initialise_hyperparameters()
        self._initialise_topology()

        self.specie = 'Initial surname'

    def get_tau(self):
        return 1 / np.sqrt(len(self.connections))

    def get_tau_n(self):
        return 1 / np.sqrt(2 * np.sqrt(len(self.connections)))

    def get_tau_n_prime(self):
        return 1 / np.sqrt(2 * len(self.connections))

    def change_specie(self, specie):
        """Changes the specie of the individual. This happens when a specie representative is picked with a different surname"""
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

    def _initialise_hyperparameters(self):
        self.global_hyperparameters['p_add_node'] = [self.p_add_node, self.initial_sigma, True]
        self.global_hyperparameters['p_add_connection'] = [self.p_add_node, self.initial_sigma, True]
        self.global_hyperparameters['p_mutate_weight'] = [self.p_mutate_weight, self.initial_sigma, True]
        self.global_hyperparameters['p_mutate_bias'] = [self.p_mutate_bias, self.initial_sigma, True]
        self.global_hyperparameters['p_mutate_non_linearity'] = [self.p_mutate_non_linearity, self.initial_sigma, True]
        self.global_hyperparameters['p_mutate_size'] = [self.p_mutate_size, self.initial_sigma, True]

    def _update_parameters(self):
        pass
        # tau = self.get_tau()
        # tau_n = self.get_tau_n()
        # tau_n_prime = self.get_tau_n_prime()
        #
        # single_r = np.random.randn()
        # for connection in self.connections:
        #     self.connections[connection][7] = self.connections[connection][7] * np.exp(
        #         tau_n_prime * single_r + tau_n * np.random.randn())
        #     self.connections[connection][7] = self.sigma_epsilon if self.connections[connection][
        #                                                                 7] < self.sigma_epsilon else \
        #         self.connections[connection][7]
        #     self.connections[connection][6] += self.connections[connection][7] * np.random.randn()
        #
        #     self.connections[connection][9] = self.connections[connection][9] * np.exp(
        #         tau_n_prime * single_r + tau_n * np.random.randn())
        #     self.connections[connection][9] = self.sigma_epsilon if self.connections[connection][
        #                                                                 9] < self.sigma_epsilon else \
        #         self.connections[connection][9]
        #     self.connections[connection][8] += self.connections[connection][8] * np.random.randn()
        #
        # for hk in self.global_hyperparameters:
        #     self.global_hyperparameters[hk][1] = self.global_hyperparameters[hk][1] * np.exp(tau * np.random.randn())
        #     self.global_hyperparameters[hk][1] = self.sigma_epsilon if self.global_hyperparameters[hk][
        #                                                                  1] < self.sigma_epsilon else \
        #         self.global_hyperparameters[hk][1]
        #     self.global_hyperparameters[hk][0] += np.random.randn() * self.global_hyperparameters[hk][1]
        #
        #     # If hyperparameter is a probability or clip between zero and one
        #     if self.global_hyperparameters[hk][2]:
        #         self.global_hyperparameters[hk][0] = np.clip(self.global_hyperparameters[hk][0], 0.0, 1.0)
        #     else:
        #         self.global_hyperparameters[hk][0] = np.clip(self.global_hyperparameters[hk][0], 0.0)

    def add_node(self):
        position = random.randint(2, len(self.nodes))
        nonlinearity = random.choice(self.nonlinearities)
        num_nodes = random.randint(self.min_initial_nodes, self.max_initial_nodes)

        self.nodes.insert(position, {'num_nodes': num_nodes, 'activation_function': nonlinearity, 'weights': None,
                                     'biases': None,
                                     'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights})

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

    def mutate(self, innovations={}, global_innovation_number=0):
        # Update local hyperparameters stored in connections and nodes and global hyperparameters.
        self._update_parameters()

        if random.random() < self.global_hyperparameters['p_add_node'][0]:
            self.add_node()
        else:
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


        return (self.distance_excess_weight * e +
                self.distance_disjoint_weight * d +
                self.distance_weight * w)
