import copy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

string_to_activation = {
    'identity': Identity(),
    'leaky_relu': nn.LeakyReLU(),
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'elu': nn.ELU()
}

class NeuralNetwork_Deep(nn.Module):
    def __init__(self):
        super(NeuralNetwork_Deep, self).__init__()

        self.modules = []
        self.model = None

    def forward(self, x):
        return self.model(x)

    def apply_weights(self, node):
        if node['weights'] is None:
            node['weights'] = self.modules[-1].weight.data
        else:
            rows = self.modules[-1].weight.data.size(0)
            cols = self.modules[-1].weight.data.size(1)

            # Add row if necessary
            difference = self.modules[-1].weight.data.size(0) - node['weights'].size(0)
            if difference > 0:
                node['weights'] = torch.cat((node['weights'], torch.zeros(difference, node['weights'].size(1))), 0)

            # Add column if necessary
            difference = self.modules[-1].weight.data.size(1) - node['weights'].size(1)
            if difference > 0:
                try:
                    node['weights'] = torch.cat((node['weights'], torch.zeros(node['weights'].size(0), difference)), 1)
                except:
                    print(node['weights'])
                    print(torch.zeros(node['weights'].size(0), difference))

            self.modules[-1].weight.data = node['weights'][:rows, :cols]

    def apply_bias(self, node):
        if node['biases'] is None:
            node['biases'] = self.modules[-1].bias.data
        else:
            cols = self.modules[-1].bias.data.size(0)

            difference = self.modules[-1].bias.data.size(0) - node['biases'].size(0)
            if difference > 0:
                node['biases'] = torch.cat((node['biases'], torch.zeros(difference)), 0)

            self.modules[-1].bias.data = node['biases'][:cols]

    def create_genome_from_network(self):
        for layer in range(0, len(self.model) - 2, 2):
            self.nodes[int(layer/2) + 2]['weights'] = self.model[layer].weight.data
            self.nodes[int(layer/2) + 2]['biases'] = self.model[layer].bias.data

        self.nodes[1]['weights'] = self.model[-2].weight.data
        self.nodes[1]['biases'] = self.model[-2].bias.data



        return self.nodes

    def create_network(self, genome):
        # Extract sequence and nodes
        self.nodes = genome.nodes

        # Keep track of the size of the previous layer
        previous_input_size = self.nodes[0]['num_nodes']

        # 0 and 1 are reserved for the input and output layer and are fixed
        for i in range(2, len(self.nodes)):
            self.modules.append(nn.Linear(previous_input_size, self.nodes[i]['num_nodes']))

            self.apply_weights(self.nodes[i])
            self.apply_bias(self.nodes[i])

            self.modules.append(string_to_activation[self.nodes[i]['activation_function']])
            previous_input_size = self.nodes[i]['num_nodes']

        self.modules.append(nn.Linear(previous_input_size, self.nodes[1]['num_nodes']))

        self.apply_weights(self.nodes[1])
        self.apply_bias(self.nodes[1])

        self.modules.append(string_to_activation[self.nodes[1]['activation_function']])

        self.model = nn.Sequential(*self.modules)


