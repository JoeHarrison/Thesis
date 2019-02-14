# Imports
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Activation functions
def leaky_relu(x):
    return F.leaky_relu(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return F.relu(x)


def sigmoid(x):
    return torch.sigmoid(x)


def identity(x):
    return x


string_to_activation = {
    'leaky_relu': leaky_relu,
    'relu': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'identity': identity
}


def dense_from_coo(shape, conns, dtype=torch.float64):
    mat = torch.zeros(shape, dtype=dtype)
    activation_matrix = np.full(shape[0], identity)
    idxs, weights, activations = conns

    activations = [string_to_activation[activation] for activation in activations]

    if len(idxs) == 0:
        return mat, activation_matrix
    rows, cols = np.array(idxs).transpose()
    mat[torch.tensor(rows), torch.tensor(cols)] = torch.tensor(
        weights, dtype=dtype)
    activation_matrix[rows] = activations
    return mat, activation_matrix

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, non_linearities, mask=None, weights=None, bias=None):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

        self.non_linearities = non_linearities

        if mask is None:
            mask = torch.ones((out_features, in_features))

        if weights is not None:
            self.linear.weight.data = weights

        # if bias is not None:
        #     self.linear.bias.data = bias

        self.linear.weight.data *= mask

    def forward(self, x):
        x = self.linear(x)
        return x
        # return torch.cat(([self.non_linearities[i](input_x).view(1) for i, input_x in enumerate(x[0])]),
        #                      dim=0).view(1, -1)


def required_for_output(inputs, outputs, connections):
    """
    Collect the nodes whose state is required to compute the final network output(s).
    :param inputs: list of the input identifiers
    :param outputs: list of the output node identifiers
    :param connections: list of (input, output) connections in the network.
    NOTE: It is assumed that the input identifier set and the node identifier set are disjoint.
    By convention, the output node ids are always the same as the output index.
    Returns a set of identifiers of required nodes.
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

class NeuralNetwork(nn.Module):
    def __init__(self, genome, batch_size=1, use_current_activs=False, n_internal_steps=1, dtype=torch.float32):
        super(NeuralNetwork, self).__init__()
        self.genome = genome
        # Only use node that are connected tot the output.
        required = required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)

        # Make list of input, hidden and output keys
        input_keys = genome.input_keys
        hidden_keys = [k[0] for k in genome.neuron_genes if
                       k[0] not in genome.output_keys and k[0] not in genome.input_keys]
        output_keys = genome.output_keys

        # Make list of hidden and output reponses
        input_responses = [genome.neuron_genes[k][5] for k in input_keys]
        hidden_responses = [genome.neuron_genes[k][5] for k in hidden_keys]
        output_responses = [genome.neuron_genes[k][5] for k in output_keys]

        # Make list of hidden and output biases
        input_biases = [genome.neuron_genes[k][2] for k in input_keys]
        hidden_biases = [genome.neuron_genes[k][2] for k in hidden_keys]
        output_biases = [genome.neuron_genes[k][2] for k in output_keys]

        self.hidden_activations = [string_to_activation[genome.neuron_genes[k][1]] for k in hidden_keys]
        self.output_activations = [string_to_activation[genome.neuron_genes[k][1]] for k in output_keys]

        # Number of inputs
        n_inputs = len(input_keys)
        n_hidden = len(hidden_keys)
        n_outputs = len(output_keys)

        # Key to index
        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}

        def key_to_idx(key):
            if key in input_keys:
                return input_key_to_idx[key]
            elif key in hidden_keys:
                return hidden_key_to_idx[key]
            elif key in output_keys:
                return output_key_to_idx[key]

        input_to_hidden = ([], [], [])
        hidden_to_hidden = ([], [], [])
        output_to_hidden = ([], [], [])
        input_to_output = ([], [], [])
        hidden_to_output = ([], [], [])
        output_to_output = ([], [], [])


        for connection in genome.connection_genes.values():
            if not connection[4]:
                continue

            if connection[2] not in required and connection[1] not in required:
                continue

            input_key = connection[1]
            output_key = connection[2]

            if input_key in input_keys and output_key in hidden_keys:
                idxs, vals, activations = input_to_hidden
            elif input_key in hidden_keys and output_key in hidden_keys:
                idxs, vals, activations = hidden_to_hidden
            elif input_key in output_keys and output_key in hidden_keys:
                idxs, vals, activations = output_to_hidden
            elif input_key in input_keys and output_key in output_keys:
                idxs, vals, activations = input_to_output
            elif input_key in hidden_keys and output_key in output_keys:
                idxs, vals, activations = hidden_to_output
            elif input_key in output_keys and output_key in output_keys:
                idxs, vals, activations = output_to_output
            else:
                raise ValueError('Invalid connection from key {} to key {}'.format(input_key, output_key))

            idxs.append((key_to_idx(connection[2]), key_to_idx(connection[1])))  # to, from
            vals.append(connection[3])
            activations.append(genome.neuron_genes[output_key][1])

        self.use_current_activs = use_current_activs

        self.n_internal_steps = n_internal_steps
        self.dtype = dtype

        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

        self.input_responses = torch.tensor(input_responses, dtype=dtype)
        self.output_responses = torch.tensor(output_responses, dtype=dtype)

        self.input_biases = torch.tensor(input_biases, dtype=dtype)
        self.output_biases = torch.tensor(output_biases, dtype=dtype)

        self.input_to_output, self.i2o_nonlinearity = dense_from_coo((n_outputs, n_inputs), input_to_output,
                                                                     dtype=dtype)

        self.output_to_output, self.o2o_nonlinearity = dense_from_coo((n_outputs, n_outputs), output_to_output,
                                                                      dtype=dtype)

        if n_hidden > 0:
            self.hidden_responses = torch.tensor(hidden_responses, dtype=dtype)
            self.hidden_biases = torch.tensor(hidden_biases, dtype=dtype)

        if n_hidden > 0:
            self.input_to_hidden, self.i2h_nonlinearity = dense_from_coo((n_hidden, n_inputs), input_to_hidden,
                                                                         dtype=dtype)
            self.hidden_to_hidden, self.h2h_nonlinearity = dense_from_coo((n_hidden, n_hidden), hidden_to_hidden,
                                                                          dtype=dtype)
            self.output_to_hidden, self.o2h_nonlinearity = dense_from_coo((n_hidden, n_outputs), output_to_hidden,
                                                                          dtype=dtype)
            self.hidden_to_output, self.h2o_nonlinearity = dense_from_coo((n_outputs, n_hidden), hidden_to_output,
                                                                          dtype=dtype)

            self.lin_input_to_hidden = MaskedLinear(n_inputs, n_hidden, self.i2h_nonlinearity,
                                                    weights=self.input_to_hidden, bias=self.hidden_biases)

            self.lin_hidden_to_hidden = MaskedLinear(n_hidden, n_hidden, self.h2h_nonlinearity,
                                                     weights=self.hidden_to_hidden, bias=self.hidden_biases)
            self.lin_output_to_hidden = MaskedLinear(n_outputs, n_hidden, self.o2h_nonlinearity,
                                                     weights=self.output_to_hidden, bias=self.hidden_biases)

            self.lin_hidden_to_output = MaskedLinear(n_hidden, n_outputs, self.h2o_nonlinearity,
                                                     weights=self.hidden_to_output, bias=self.output_biases)

        self.lin_input_to_output = MaskedLinear(n_inputs, n_outputs, self.i2o_nonlinearity,
                                                weights=self.input_to_output, bias=self.output_biases)
        self.lin_output_to_output = MaskedLinear(n_outputs, n_outputs, self.o2o_nonlinearity,
                                                 weights=self.output_to_output, bias=self.output_biases)

        self.reset(batch_size)

    def reset(self, batch_size=1):
        if self.n_hidden > 0:
            self.activs = torch.zeros(batch_size, self.n_hidden, dtype=self.dtype)
        else:
            self.activs = None
        self.outputs = torch.zeros(batch_size, self.n_outputs, dtype=self.dtype)

    def forward(self, inputs):
        '''
        inputs: (batch_size, n_inputs)
        returns: (batch_size, n_outputs)
        '''
        inputs = torch.tensor(inputs, dtype=self.dtype)

        activs_for_output = self.activs
        if self.n_hidden > 0:
            for _ in range(self.n_internal_steps):
                pre_activs = self.hidden_responses * (
                        self.lin_input_to_hidden(inputs) +
                        self.lin_hidden_to_hidden(self.activs) +
                        self.lin_output_to_hidden(self.outputs)
                ) + self.hidden_biases

                self.activs = torch.cat(([self.hidden_activations[i](input_x).view(1) for i, input_x in enumerate(pre_activs[0])]), dim=0).view(1,-1)

            if self.use_current_activs:
                activs_for_output = self.activs

        output_inputs = self.lin_input_to_output(inputs) + self.lin_output_to_output(self.outputs)

        if self.n_hidden > 0:
            output_inputs += self.lin_hidden_to_output(activs_for_output)
        pre_activ = self.output_responses * output_inputs + self.output_biases
        self.outputs = torch.cat(([self.output_activations[i](input_x).view(1) for i, input_x in enumerate(pre_activ[0])]), dim=0).view(1,-1)

        return self.outputs
