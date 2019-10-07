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
                 p_add_connection=0.3,
                 p_add_node=0.5,
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

        # Variables pertaining to individual naming
        self.name = next(new_individual_name)
        self.specie = None


        self.inputs = inputs
        self.outputs = outputs
        self.nonlinearities = nonlinearities
        self.feedforward = feedforward

        self.p_add_node = p_add_node
        self.p_add_connection = p_add_connection
        self.p_mutate_node = p_mutate_node

        # Distance variables
        self.distance_excess_weight = distance_excess_weight
        self.distance_disjoint_weight = distance_disjoint_weight
        self.distance_weight = distance_weight

        self.p_add_interlayer_node = p_add_interlayer_node
        self.p_change_layer_nonlinearity = p_change_layer_nonlinearity

        self.initial_p_mutate_interlayer_weights = initial_p_mutate_interlayer_weights
        self.min_initial_nodes = min_initial_nodes
        self.max_initial_nodes = max_initial_nodes

        self.initial_p_mutate_weights = initial_p_mutate_weights
        self.initial_sigma_mutation_weights = initial_sigma_mutation_weights

        self.nodes = []

        # Innovation number, input node, output node, enabled, Weights, Biases
        self.connections = {}
        self.global_hyperparameters = {}

        self._initialise_topology()

        self.specie = 'Initial surname'

    def change_specie(self, specie):
        """Changes the specie of the individual. This happens when a specie representative is picked with a different surname"""
        self.specie = specie

    def _initialise_topology(self):
        """Initialises the topology. The input layer's weight and biases are never changed.
        The weights and biases of the output layer are initialised in the feedforward_deep module."""

        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'node_id': 0, 'num_nodes': self.inputs, 'activation_function': nonlinearity, 'fforder': 0})

        nonlinearity = random.choice(self.nonlinearities)
        self.nodes.append(
            {'node_id': 1, 'num_nodes': self.outputs, 'activation_function': nonlinearity, 'fforder': 2048})

        self.connections[(0, 1)] = [0, 0, 1, True, None, None]

    def _update_parameters(self):
        for connection in self.connections:
            pass

        for k in self.global_hyperparameters:
            pass


    def add_node(self, maximum_innovation_number, innovations):
        if len(innovations.values())!=len(set(innovations.values())):
            print('Add node 1')
            print('max innov:', maximum_innovation_number)
            print('innovs:', innovations)
            print([conn[:4] for conn in self.connections.values()])
            raise IndexError

        possible_to_split = [(fr, to) for (fr, to) in self.connections.keys() if
                             self.nodes[fr]['fforder'] + 1 < self.nodes[to]['fforder']]

        if possible_to_split:
            random_node = random.choice(possible_to_split)
            print(random_node)
            split_node = self.connections[random_node]

            # Disable old connection
            split_node[3] = False

            input_node, output_node = split_node[1:3]

            fforder = (self.nodes[input_node]['fforder'] + self.nodes[output_node]['fforder']) * 0.5
            nonlinearity = random.choice(self.nonlinearities)
            num_nodes = random.randint(self.min_initial_nodes, self.max_initial_nodes)
            new_node_id = len(self.nodes)

            self.nodes.append({'node_id': new_node_id, 'num_nodes': num_nodes, 'activation_function': nonlinearity, 'fforder': fforder})

            if (input_node, new_node_id) in innovations:
                innovation_number = innovations[(input_node, new_node_id)]
            else:
                maximum_innovation_number += 1
                innovations[(input_node, new_node_id)] = maximum_innovation_number
                innovation_number = maximum_innovation_number

            if len(innovations.values())!=len(set(innovations.values())):
                print('Add node 2')
                print('max innov:', maximum_innovation_number)
                print('innovs:', innovations)
                print([conn[:4] for conn in self.connections.values()])
                raise IndexError


            self.connections[(input_node, new_node_id)] = [innovation_number, input_node, new_node_id, True, None, None]

            if len(innovations.values())!=len(set(innovations.values())):
                print('Add node 3')
                print('max innov:', maximum_innovation_number)
                print('innovs:', innovations)
                print([conn[:4] for conn in self.connections.values()])
                raise IndexError

            if (new_node_id, output_node) in innovations:
                innovation_number = innovations[(new_node_id, output_node)]
            else:
                maximum_innovation_number += 1
                innovations[(new_node_id, output_node)] = maximum_innovation_number
                innovation_number = maximum_innovation_number

            if len(innovations.values())!=len(set(innovations.values())):
                print('Add node 4')
                print('max innov:', maximum_innovation_number)
                print('innovs:', innovations)
                print([conn[:4] for conn in self.connections.values()])
                raise IndexError

            self.connections[(new_node_id, output_node)] = [innovation_number, new_node_id, output_node, True, None,
                                                            None]

            if len(innovations.values())!=len(set(innovations.values())):
                print('Add node 5')
                print('max innov:', maximum_innovation_number)
                print('innovs:', innovations)
                print([conn[:4] for conn in self.connections.values()])
                raise IndexError

    def add_connection(self, maximum_innovation_number, innovations):
        if len(innovations.values())!=len(set(innovations.values())):
            print('Add connection 1')
            print('max innov:', maximum_innovation_number)
            print('innovs:', innovations)
            print([conn[:4] for conn in self.connections.values()])
            raise IndexError

        potential_connections = product(range(len(self.nodes)), range(1, len(self.nodes)))
        potential_connections = (connection for connection in potential_connections if
                                 connection not in self.connections)

        if self.feedforward:
            potential_connections = ((f, t) for (f, t) in potential_connections if
                                     self.nodes[f]['fforder'] < self.nodes[t]['fforder'])

        potential_connections = list(potential_connections)

        if potential_connections:
            (fr, to) = random.choice(potential_connections)
            if (fr, to) in innovations:
                innovation = innovations[(fr, to)]
            else:
                maximum_innovation_number += 1
                innovations[(fr, to)] = maximum_innovation_number
                innovation = maximum_innovation_number


            if len(innovations.values())!=len(set(innovations.values())):
                print('Add connection 2')
                print('max innov:', maximum_innovation_number)
                print('innovs:', innovations)
                print([conn[:4] for conn in self.connections.values()])
                raise IndexError

            connection_gene = [innovation, fr, to, True, None, None]
            self.connections[(fr, to)] = connection_gene

            if len(innovations.values())!=len(set(innovations.values())):
                print('Add connection 3')
                print('max innov:', maximum_innovation_number)
                print('innovs:', innovations)
                print([conn[:4] for conn in self.connections.values()])
                raise IndexError

    def mutate_node(self):
        if random.random() < self.p_add_interlayer_node and len(self.nodes) - 1 > 2:
            position = random.randint(2, len(self.nodes) - 1)
            # Ensure that no more nodes can be deleted than there exist.
            self.nodes[position]['num_nodes'] += random.randint(-min(self.nodes[position]['num_nodes'] - 1, 1), 1)

        if random.random() < self.p_change_layer_nonlinearity:
            position = random.randint(1, len(self.nodes) - 1)
            self.nodes[position]['activation_function'] = random.choice(self.nonlinearities)

        for conn in self.connections.keys():
            # p_mutate_weight
            if random.random() < 0.1:
                # Mutate Weights by multiplying a mask with random numbers
                if self.connections[conn][4] is not None:
                    t = torch.rand_like(self.connections[conn][4])
                    self.connections[conn][4] += (t < 0.1).float() * torch.randn_like(self.connections[conn][4]) * 0.1

            # p_mutate_bias (In NEAT this happens in the node loop)
            if random.random() < 0.1:
                if self.connections[conn][5] is not None:
                    t = torch.rand_like(self.connections[conn][5])
                    self.connections[conn][5] += (t < 0.1).float() * torch.randn_like(self.connections[conn][5]) * 0.1

        for node in self.nodes:
            # p_mutate_non_linearity
            if random.random() < 0.1:
                node['activation_function'] = random.choice(self.nonlinearities)

            # p_mutate_number_of_nodes
            # Ensure that only the number of neurons within a hidden layer are changed
            if random.random() < 0.1 and node['node_id'] not in [0, 1]:
                node['num_nodes'] += random.randint(-5, 5)
                node['num_nodes'] = np.clip(node['num_nodes'], a_min=1, a_max=None)

    def mutate(self, innovations={}, global_innovation_number=0):
        # Update local hyperparameters stored in connections and nodes and global hyperparameters.
        self._update_parameters()

        if random.random() < self.p_add_node:
            global_innovation_number = max([global_innovation_number] + list(innovations.values()))
            maximum_innovation_number = max(global_innovation_number, max(cg[0] for cg in self.connections.values()))
            self.add_node(maximum_innovation_number, innovations)
        elif random.random() < self.p_add_connection:
            global_innovation_number = max([global_innovation_number] + list(innovations.values()))
            maximum_innovation_number = max(global_innovation_number, max(cg[0] for cg in self.connections.values()))
            self.add_connection(maximum_innovation_number, innovations)
        else:
            self.mutate_node()

        if len([c[0] for c in self.connections.values()])!=len(set([c[0] for c in self.connections.values()])):
            print('mutate')
            raise IndexError

    def recombinate(self, other):
        child = deepcopy(self)
        child.nodes = []
        child.connections = {}

        max_nodes = max(len(self.nodes), len(other.nodes))
        min_nodes = min(len(self.nodes), len(other.nodes))

        for i in range(max_nodes):
            node = None
            if i < min_nodes:
                node = random.choice((self.nodes[i], other.nodes[i]))
            else:
                try:
                    node = self.nodes[i]
                except IndexError:
                    node = other.nodes[i]
            child.nodes.append(deepcopy(node))

        self_connections = dict(((c[0], c) for c in self.connections.values()))
        other_connections = dict(((c[0], c) for c in other.connections.values()))

        if len(self_connections) > 0 or len(other_connections) > 0:
            max_innovation_number = max(list(self_connections.keys()) + list(other_connections.keys()))

            for i in range(max_innovation_number + 1):
                connection = None
                if i in self_connections and i in other_connections:
                    connection = random.choice((self_connections[i], other_connections[i]))
                    enabled = self_connections[i][3] and other_connections[i][3]
                else:
                    if i in self_connections:
                        connection = self_connections[i]
                        enabled = connection[3]
                    elif i in other_connections:
                        connection = other_connections[i]
                        enabled = connection[3]
                if connection is not None:
                    child.connections[(connection[1], connection[2])] = deepcopy(connection)
                    child.connections[(connection[1], connection[2])][
                        3] = enabled

                def is_feedforward(item):
                    ((fr, to), cg) = item
                    # add layers?
                    return child.nodes[fr]['fforder'] < child.nodes[to]['fforder']

                if self.feedforward:
                    child.connections = dict(filter(is_feedforward, child.connections.items()))

        child.name = self.name[:3] + other.name[3:]

        if len([c[0] for c in self.connections.values()])!=len(set([c[0] for c in self.connections.values()])):
            print('recombinate')
            raise IndexError

        return child

    def distance(self, other):
        self_connections = dict(((c[0], c) for c in self.connections.values()))
        other_connections = dict(((c[0], c) for c in other.connections.values()))

        all_innovations = list(set(list(self_connections.keys()) + list(other_connections.keys())))

        if len(all_innovations) == 0:
            return 0

        minimum_innovation = min(all_innovations)

        e = 0
        d = 0
        w = 0.0
        m = 0

        for innovation_key in all_innovations:
            if innovation_key in self_connections and innovation_key in other_connections:

                min_row = min(self_connections[innovation_key][4].size(0), other_connections[innovation_key][4].size(0))
                min_column = min(self_connections[innovation_key][4].size(1),
                                 other_connections[innovation_key][4].size(1))

                m += min_row * min_column
                w += torch.abs(
                    self_connections[innovation_key][4].data[:min_row, :min_column] - other_connections[innovation_key][
                                                                                          4].data[:min_row,
                                                                                      :min_column]).sum().item()

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

    g1.connections = {(0, 1): [0, 0, 1, False, None, None], (0, 2): [1, 0, 2, False, None, None], (2, 1): [2, 2, 1, False, None, None], (2, 3): [3, 2, 3, True, None, None],
                      (3, 1): [4, 3, 1, True, None, None], (0, 3): [5, 0, 3, True, None, None], (0, 4): [7, 0, 4, True, None, None], (4, 2): [8, 4, 2, True, None, None],
                      (0, 5): [9, 0, 5, True, None, None], (5, 1): [13, 5, 1, True, None, None], (4, 1): [8, 4, 1, True, None, None]}


    g1.nodes.append(
        {'node_id': 4, 'num_nodes': random.randint(5, 300), 'activation_function': 'tanh', 'fforder': 512})

    g1.nodes.append(
        {'node_id': 2, 'num_nodes': random.randint(5, 300), 'activation_function': 'sigmoid', 'fforder': 1024})

    g1.nodes.append(
        {'node_id': 3, 'num_nodes': random.randint(5, 300), 'activation_function': 'tanh', 'fforder': 1024})

    g1.nodes.append(
        {'node_id': 5, 'num_nodes': random.randint(5, 300), 'activation_function': 'tanh', 'fforder': 1024})

    from feedforwardnetwork_deep import NeuralNetwork_Deep

    n2 = NeuralNetwork_Deep('cpu')
    n2.create_network(g1)
    g1.connections = n2.create_genome_from_network()

    child = g1.recombinate(g1)

    n1 = NeuralNetwork_Deep('cpu')

    n1.create_network(child)

    child.connections = n1.create_genome_from_network()

    rl = RubiksTask_Deep(128, 'cpu', True, True, 0.99, None, 'LBF', use_single_activation_function=False)

    rl.evaluate(child, 0)

    print(n1(torch.tensor([-1.0] * 144)))
