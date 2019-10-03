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
                 feedforward=True,
                 p_add_connection=0.1,
                 p_add_node=0.1,
                 p_delete_node=0.03,
                 p_mutate_node=0.8,
                 p_add_interlayer_node=0.1,
                 p_change_layer_nonlinearity=0.01,
                 initial_p_mutate_weights=0.03,
                 initial_sigma_mutation_weights=0.1,
                 distance_excess_weight=1.0,
                 distance_disjoint_weight=1.0,
                 distance_weight=0.4,
                 initial_p_mutate_interlayer_weights=0.1,
                 min_initial_nodes=32,
                 max_initial_nodes=256):

        # Name the new individual
        self.name = next(new_individual_name)
        self.specie = None

        self.inputs = inputs
        self.outputs = outputs
        self.nonlinearities = nonlinearities
        self.feedforward = True

        self.p_add_node = p_add_node
        self.p_delete_node = p_delete_node
        self.p_mutate_node = p_mutate_node

        self.distance_excess_weight = distance_excess_weight
        self.distance_disjoint_weight = distance_disjoint_weight
        self.distance_weight = distance_weight

        self.p_add_connection = p_add_connection

        self.p_add_interlayer_node = p_add_interlayer_node
        self.p_change_layer_nonlinearity = p_change_layer_nonlinearity

        self.initial_p_mutate_interlayer_weights = initial_p_mutate_interlayer_weights
        self.min_initial_nodes = min_initial_nodes
        self.max_initial_nodes = max_initial_nodes

        self.initial_p_mutate_weights = initial_p_mutate_weights
        self.initial_sigma_mutation_weights = initial_sigma_mutation_weights

        self.nodes = []
        self.connections = {}

        self._initialise_topology()

        self.rl_training = False
        self.specie = 'Initial surname'

    def change_specie(self, specie):
        """Changes the specie of the individual. This happens when a specie representative is picked with a different surname"""
        self.specie = specie

    def _initialise_topology(self):
        """Initialises the topology. The input layer's weight and biases are never changed.
        The weights and biases of the output layer are initialised in the feedforward module."""

        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'node_id': 0, 'num_nodes': self.inputs, 'activation_function': nonlinearity,
             'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights,
             'p_mutate_weights': self.initial_p_mutate_weights,
             'sigma_mutation_weights': self.initial_sigma_mutation_weights, 'fforder': 0})

        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'node_id': 1, 'num_nodes': self.outputs, 'activation_function': nonlinearity,
             'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights,
             'p_mutate_weights': self.initial_p_mutate_weights,
             'sigma_mutation_weights': self.initial_sigma_mutation_weights, 'fforder': 2048})

        self.connections[(0, 1)] = [0, 0, 1, True, None, None]

    def _update_parameters(self):
        for node in self.nodes:
            node['sigma_mutation_weights'] *= np.exp(0.1*np.random.randn())
            node['p_mutate_weights'] += np.random.randn()*node['sigma_mutation_weights']

    def add_node(self, maximum_innovation_number, innovations):

        possible_to_split = [(fr, to) for (fr, to) in self.connections.keys() if self.nodes[fr]['fforder'] + 1 < self.nodes[to]['fforder']]

        if possible_to_split:
            split_neuron = self.connections[random.choice(possible_to_split)]

            # Disable old connection
            split_neuron[3] = False

            input_neuron, output_neuron = split_neuron[1:3]

            fforder = (self.nodes[input_neuron]['fforder'] + self.nodes[output_neuron]['fforder']) * 0.5
            nonlinearity = random.choice(self.nonlinearities)
            num_nodes = random.randint(self.min_initial_nodes, self.max_initial_nodes)
            new_node_id = len(self.nodes)

            self.nodes.append({'node_id': new_node_id, 'num_nodes': num_nodes, 'activation_function': nonlinearity,
                                         'p_mutate_interlayer_weights': self.initial_p_mutate_interlayer_weights,
                                         'p_mutate_weights': self.initial_p_mutate_weights,
                                         'sigma_mutation_weights': self.initial_sigma_mutation_weights, 'fforder': fforder})

            if (input_neuron, new_node_id) in innovations:
                innovation_number = innovations[(input_neuron, new_node_id)]
            else:
                maximum_innovation_number += 1
                innovation_number = innovations[(input_neuron, new_node_id)] = maximum_innovation_number

            self.connections[(input_neuron, new_node_id)] = [innovation_number, input_neuron, new_node_id, True, None, None]

            if (new_node_id, output_neuron) in innovations:
                innovation_number = innovations[(new_node_id, output_neuron)]
            else:
                maximum_innovation_number += 1
                innovation_number = innovations[(new_node_id, output_neuron)] = maximum_innovation_number

            self.connections[(new_node_id, output_neuron)] = [innovation_number, new_node_id, output_neuron, True, None, None]


    def add_connection(self, maximum_innovation_number, innovations):
        potential_connections = product(range(len(self.nodes)), range(self.inputs, len(self.nodes)))
        potential_connections = (connection for connection in potential_connections if connection not in self.connections)

        if self.feedforward:
            potential_connections = ((f, t) for (f, t) in potential_connections if self.nodes[f]['fforder'] < self.nodes[t]['fforder'])

        potential_connections = list(potential_connections)

        if potential_connections:
            (fr, to) = random.choice(potential_connections)
            if (fr, to) in innovations:
                innovation = innovations[(fr, to)]
            else:
                maximum_innovation_number += 1
                innovation = innovations[(fr, to)] = maximum_innovation_number

            connection_gene = [innovation, fr, to, True, None, None]
            self.connections[(fr, to)] = connection_gene

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

        for conn in self.connections.keys():
            if random.random() < 0.1:
                if self.connections[conn][4] is not None:
                    t = torch.rand_like(self.connections[conn][4])
                    self.connections[conn][4] += (t < 0.1).float() * torch.randn_like(self.connections[conn][4]) * 0.1

    def mutate(self, innovations={}, global_innovation_number=0):
        self._update_parameters()

        maximum_innovation_number = global_innovation_number

        if len(self.connections.values()):
            maximum_innovation_number = max(global_innovation_number, max(cg[0] for cg in self.connections.values()))

        if random.random() < self.p_add_node:
            self.add_node(maximum_innovation_number, innovations)
        elif random.random() < self.p_add_connection:
            self.add_connection(maximum_innovation_number, innovations)
        else:
            self.mutate_node()

    def recombinate(self, other):
        # TODO recombination can revert to old method.
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
        self_connections = dict(((c[0], c) for c in self.connections.values()))
        other_connections = dict(((c[0], c) for c in other.connections.values()))

        all_innovations = list(self_connections.keys()) + list(other_connections.keys())

        if len(all_innovations) == 0:
            return 0

        minimum_innovation = min(all_innovations)

        e = 0
        d = 0
        w = 0.0
        m = 0

        for innovation_key in all_innovations:
            if innovation_key in self_connections and innovation_key in other_connections:
                # TODO!!


                w += np.abs(self_connections[innovation_key][3] - other_connections[innovation_key][3])
                m += 1
            elif innovation_key in self_connections or innovation_key in other_connections:
                # Disjoint genes
                if innovation_key < minimum_innovation:
                    d += 1
                # Excess genes
                else:
                    e += 1

        # Average weight differences of matching genes
        w = (w / m) if m > 0 else w

        return (self.distance_excess_weight * e +
                self.distance_disjoint_weight * d +
                self.distance_weight * w)

if __name__ == "__main__":
    first_name_generator = NameGenerator('../naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()

    g1 = Genotype_Deep(new_individual_name, 144, 6)

    from feedforwardnetwork_deep import NeuralNetwork_Deep

    n1 = NeuralNetwork_Deep('cpu')

    n1.create_network(g1)

    g1.connections = n1.create_genome_from_network()

    print(n1(torch.tensor([0.0]*144)))

