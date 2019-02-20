import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from genotype import Genotype
from namegenerator import NameGenerator

import random

# Activation functions
def identity(x):
    return x

def leaky_relu(x):
    return F.leaky_relu(x)

def tanh(x):
    return torch.tanh(x)


def relu(x):
    return F.relu(x)

def sigmoid(x):
    return torch.sigmoid(x)

string_to_activation = {
    'identity': identity,
    'leaky_relu': leaky_relu,
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh

}

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

def dense_from_coo(shape, conns, dtype=torch.float32, device=torch.device('cpu')):
    mat = torch.zeros(shape, dtype=dtype, device=device)
    idxs, weights = conns

    if len(idxs) == 0:
        return mat
    rows, cols = np.array(idxs).transpose()
    mat[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(weights, dtype=dtype, device=device)

    return mat

class WeightLinear(nn.Module):
    def __init__(self, in_features, out_features, weights):
        super(WeightLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

        if weights is not None:
            self.linear.weight.data = weights

    def forward(self, x):
        return self.linear(x)

class NeuralNetwork(nn.Module):
    def __init__(self, genome, batch_size=1, device=torch.device('cpu'), dtype=torch.float32):
        super(NeuralNetwork, self).__init__()

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        # Build list of neurons that are required for output excluding input neurons
        required = required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)

        self.input_keys = genome.input_keys
        self.hidden_keys = [k[0] for k in genome.neuron_genes if k[0] not in genome.output_keys and k[0] not in genome.input_keys]

        self.output_keys = genome.output_keys

        self.n_inputs = len(self.input_keys)
        self.n_hidden = len(self.hidden_keys)
        self.n_outputs = len(self.output_keys)

        if self.n_hidden >0:
            self.n_layers = max([k[3] for k in genome.neuron_genes if k[0] not in genome.output_keys and k[0] not in genome.input_keys])
            self.hidden_biases = torch.tensor([genome.neuron_genes[k][2] for k in self.hidden_keys], dtype=dtype, device=self.device, requires_grad = True)
            #activations here?
            self.hidden_activations = [string_to_activation[genome.neuron_genes[k][1]] for k in self.hidden_keys]

        self.output_biases = torch.tensor([genome.neuron_genes[k][2] for k in self.output_keys], dtype=dtype, device=self.device, requires_grad = True)
        #Activations here?
        self.output_activations = [string_to_activation[genome.neuron_genes[k][1]] for k in self.output_keys]

        self.input_key_to_idx = {k: i for i, k in enumerate(self.input_keys)}
        self.hidden_key_to_idx = {k: i for i, k in enumerate(self.hidden_keys)}
        self.output_key_to_idx = {k: i for i, k in enumerate(self.output_keys)}

        def key_to_idx(key):
            if key in self.input_keys:
                return self.input_key_to_idx[key]
            elif key in self.hidden_keys:
                return self.hidden_key_to_idx[key]
            elif key in self.output_keys:
                return self.output_key_to_idx[key]

        input_to_output = ([], [])
        input_to_hidden = ([], [])
        hidden_to_hidden = ([], [])
        output_to_hidden = ([], [])
        hidden_to_output = ([], [])
        output_to_output = ([], [])

        for connection in genome.connection_genes.values():
            # Continue if connection not enabled
            if not connection[4]:
                continue

            input_key = connection[1]
            output_key = connection[2]

            if input_key in self.input_keys and output_key in self.hidden_keys:
                idxs, vals = input_to_hidden
            elif input_key in self.hidden_keys and output_key in self.hidden_keys:
                idxs, vals = hidden_to_hidden
            elif input_key in self.output_keys and output_key in self.hidden_keys:
                idxs, vals = output_to_hidden
            elif input_key in self.input_keys and output_key in self.output_keys:
                idxs, vals = input_to_output
            elif input_key in self.hidden_keys and output_key in self.output_keys:
                idxs, vals = hidden_to_output
            elif input_key in self.output_keys and output_key in self.output_keys:
                idxs, vals = output_to_output

            idxs.append((key_to_idx(connection[2]), key_to_idx(connection[1])))  # to, from
            vals.append(connection[3])

        i2o_weights = dense_from_coo((self.n_outputs, self.n_inputs), input_to_output, dtype=dtype, device=self.device)
        self.input_to_output = WeightLinear(self.n_inputs, self.n_outputs, weights=i2o_weights)

        o2o_weights = dense_from_coo((self.n_outputs, self.n_outputs), output_to_output, dtype=dtype, device=self.device)
        self.output_to_output = WeightLinear(self.n_outputs, self.n_outputs, weights=o2o_weights)

        if self.n_hidden > 0:
            i2h_weights = dense_from_coo((self.n_hidden, self.n_inputs), input_to_hidden, dtype=dtype, device=self.device)
            self.input_to_hidden = WeightLinear(self.n_inputs, self.n_hidden, weights=i2h_weights)

            h2h_weights = dense_from_coo((self.n_hidden, self.n_hidden), hidden_to_hidden, dtype=dtype, device=self.device)
            self.hidden_to_hidden = WeightLinear(self.n_hidden, self.n_hidden, weights=h2h_weights)

            h2o_weights = dense_from_coo((self.n_outputs, self.n_hidden), hidden_to_output, dtype=dtype, device=self.device)
            self.hidden_to_output = WeightLinear(self.n_hidden, self.n_outputs, weights=h2o_weights)

        self.reset()

    def reset(self):
        if self.n_hidden > 0:
            self.activations = torch.zeros(self.batch_size, self.n_hidden, dtype=self.dtype, device=self.device)
        else:
            self.activations = None
        self.outputs = torch.zeros(self.batch_size, self.n_outputs, dtype=self.dtype, device=self.device)

    def forward(self, x):
        # inputs = torch.tensor(x, dtype=self.dtype)
        inputs = x.type(self.dtype).to(self.device)

        activations_for_output = self.activations
        if self.n_hidden > 0:
            #self.n_hidden needs to be n_hidden_layers
            for i in range(self.n_layers):
                hidden_inputs = self.input_to_hidden(inputs) + \
                                  self.hidden_to_hidden(self.activations) + \
                                  self.hidden_biases

                self.activations = torch.cat(([self.hidden_activations[i](hidden_inputs[:,i]).view(-1,1) for i in range(self.n_hidden)]), dim=1)

            activations_for_output = self.activations

        output_inputs = self.input_to_output(inputs) + self.output_to_output(self.outputs)

        if self.n_hidden > 0:
            output_inputs += self.hidden_to_output(activations_for_output)

        output_inputs += self.output_biases

        self.outputs = torch.cat(([self.output_activations[i](output_inputs[:,i]).view(-1,1) for i in range(self.n_outputs)]), dim=1)

        return self.outputs

# np.random.seed(3)
# torch.manual_seed(3)
# random.seed(3)
#
# first_name_generator = NameGenerator('names.csv', 3, 12)
# new_individual_name = first_name_generator.generate_name()
# surname_generator = NameGenerator('surnames.csv', 3, 12)
# new_specie_name = surname_generator.generate_name()
#
# genome = Genotype(new_individual_name, 2, 1)
#
# # genome.neuron_genes = [[0, 'sigmoid', 5.0, 0, 0, 1.0], [1, 'tanh', 6.0, 0, 2048, 1.0], [2, 'sigmoid', 0.5, 2048, 4096, 1.0], [3, 'tanh', 3.0, 1, 2048, 1.0]]
# # genome.connection_genes = {(0, 2): [0, 0, 2, 2.0, False], (1, 2): [1, 1, 2, 4.0, True], (0,3): [2, 0, 3, 2.5, True], (3,2): [3, 3, 2, 3.5, True]}
#
# genome.neuron_genes = [[0, 'sigmoid', 0.1, 0, 1],[1, 'sigmoid', 0.2, 0, 1],[2, 'tanh', 0.3, 2048, 1],[3, 'sigmoid', 0.2, 1, 1],[4, 'tanh', 0.1, 1, 1],[5, 'sigmoid', 0.2, 2, 1],[6, 'tanh', 0.3, 2, 1],[7, 'sigmoid', 0.2, 2, 1]]
# genome.connection_genes = {(0,3): [1,0,3,0.5,True],(1,4): [1,1,4,0.6,True],(1,7): [1,1,7,0.7,True],(1,6): [1,1,6,0.6,True],(3,5): [1,3,5,0.5,True],(3,6): [1,3,6,0.6,True],(4,6): [1,4,6,0.7,True],(5,2): [1,5,2,0.6,True],(6,2): [1,6,2,0.5,True]}
#
# print(genome.neuron_genes)
# print(genome.connection_genes)
#
# network = NeuralNetwork(genome)
# input = np.array([[0,0],[0,1],[1,0],[1,1]])
# output = network(np.array([[0,0],[0,1],[1,0],[1,1]]))
# # output = network(np.array([[0,0],[0,1],[1,0],[1,1]]))
# print(input)
# print(output)
