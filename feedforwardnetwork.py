import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Activation functions
def identity(x):
    return x

def leaky_relu(x):
    return F.leaky_relu(x)

def elu(x):
    return F.elu(x)

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
    'tanh': tanh,
    'elu' : elu
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
        self.mask = weights != 0
        self.mask_inverse = weights == 0
        self.mask_float = self.mask.float()

        if weights is not None:
            self.linear.weight.data = weights
            self.linear.weight.register_hook(self.backward_hook)

    def forward(self, x):
        return self.linear(x)

    def backward_hook(self, grad):

        grad_out = grad.clone()
        grad_out[self.mask_inverse] = 0.0

        return grad_out


class NeuralNetwork(nn.Module):
    def __init__(self, genome, batch_size=1, device=torch.device('cpu'), dtype=torch.float32, use_single_activation_function=False):
        super(NeuralNetwork, self).__init__()

        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device
        self.use_single_activation_function = use_single_activation_function

        # Build list of neurons that are required for output excluding input neurons
        required = required_for_output(genome.input_keys, genome.output_keys, genome.connection_genes)

        self.input_keys = genome.input_keys
        self.hidden_keys = [k[0] for k in genome.neuron_genes if k[0] not in genome.output_keys and k[0] not in genome.input_keys]

        self.output_keys = genome.output_keys

        self.n_inputs = len(self.input_keys)
        self.n_hidden = len(self.hidden_keys)
        self.n_outputs = len(self.output_keys)

        if self.n_hidden > 0:
            self.n_layers = max([k[3] for k in genome.neuron_genes if k[0] not in genome.output_keys and k[0] not in genome.input_keys])
            self.hidden_biases = nn.Parameter(torch.tensor([genome.neuron_genes[k][2] for k in self.hidden_keys], dtype=dtype, device=self.device, requires_grad=True))
            self.hidden_activations = [string_to_activation[genome.neuron_genes[k][1]] for k in self.hidden_keys]

        self.output_biases = nn.Parameter(torch.tensor([genome.neuron_genes[k][2] for k in self.output_keys], dtype=dtype, device=self.device, requires_grad=True))
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

            o2h_weights = dense_from_coo((self.n_hidden, self.n_outputs), output_to_hidden, dtype=dtype, device=self.device)
            self.output_to_hidden = WeightLinear(self.n_outputs, self.n_hidden, weights=o2h_weights)

        self.reset()

    def reset(self):
        if self.n_hidden > 0:
            self.activations = torch.zeros(self.batch_size, self.n_hidden, dtype=self.dtype, device=self.device)
        else:
            self.activations = None
        self.outputs = torch.zeros(self.batch_size, self.n_outputs, dtype=self.dtype, device=self.device)

    def forward(self, x):
        inputs = x.type(self.dtype).to(self.device)

        activations_for_output = self.activations
        if self.n_hidden > 0:
            for i in range(self.n_layers):
                hidden_inputs = self.input_to_hidden(inputs) + \
                                  self.hidden_to_hidden(self.activations) + \
                                  self.output_to_hidden(self.outputs) + \
                                  self.hidden_biases
                if self.use_single_activation_function:
                    self.activations = torch.tanh(hidden_inputs)
                else:
                    self.activations = torch.cat(([self.hidden_activations[i](hidden_inputs[:, i]).view(-1, 1) for i in range(self.n_hidden)]), dim=1)

            activations_for_output = self.activations

        output_inputs = self.input_to_output(inputs) + self.output_to_output(self.outputs)

        if self.n_hidden > 0:
            output_inputs += self.hidden_to_output(activations_for_output)

        output_inputs = output_inputs + self.output_biases

        if self.use_single_activation_function:
            self.outputs = torch.tanh(output_inputs)
        else:
            self.outputs = torch.cat(([self.output_activations[i](output_inputs[:, i]).view(-1, 1) for i in range(self.n_outputs)]), dim=1)

        return self.outputs

if __name__ == "__main__":
    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(2,2)
            self.s1 = nn.Sigmoid()
            self.fc2 = nn.Linear(2,2)
            self.s2 = nn.Sigmoid()
            self.fc1.weight = torch.nn.Parameter(torch.Tensor([[0.0, 0.2],[0.250,0.30]]))
            self.fc1.bias = torch.nn.Parameter(torch.Tensor([0.35]))
            self.fc2.weight = torch.nn.Parameter(torch.Tensor([[0.4,0.45],[0.5,0.55]]))
            self.fc2.bias = torch.nn.Parameter(torch.Tensor([0.6]))
            self.hook = self.register_backward_hook(self.hook_fn)

        def forward(self, x):
            x = self.fc1(x)
            x = self.s1(x)
            x = self.fc2(x)
            x = self.s2(x)
            return x

        def hook_fn(self, module, input, output):
            print(input)
            self.input = input
            self.output = output

    net = Net()

    data = torch.Tensor([0.05, 0.1])

    # output of last layer
    out = net(data)
    target = torch.Tensor([0.01, 0.99])  # a dummy target, for example
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    loss.backward()


    # from naming.namegenerator import NameGenerator
    # from NEAT.genotype import Genotype
    #
    # first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    # new_individual_name = first_name_generator.generate_name()
    #
    # gen = Genotype(new_individual_name, 2, 1)
    #
    # gen.neuron_genes = [[0, 'relu', 0.0, 0, 0.0, 1.0], [1, 'relu', 0.0, 0, 1024.0, 1.0], [2,  'relu', 0.0, 2048, 2048.0, 1.0], [3,  'relu', 0.0, 1, 1024.0, 1.0], [4,  'relu', 0.0, 2, 1536.0, 1.0], [5, 'relu', 0.18701591191571043, 1, 512.0, 1.0], [6, 'relu', 0.0, 1, 512.0, 1.0], [7, 'relu', 0.0, 2, 768.0, 1.0]]
    #
    # gen.connection_genes = {(0, 2): [0, 0, 2, 3.0, False], (1, 2): [1, 1, 2, 3.0, True], (0, 3): [2, 0, 3, 1.3543900040487509, True], (3, 2): [3, 3, 2, -1.7074518345019327, True], (0, 4): [4, 0, 4, 1.6063807318468386, True], (4, 2): [5, 4, 2, 2.7212401858775306, True], (0, 5): [6, 0, 5, 1.2006299281998838, True], (5, 3): [7, 5, 3, -3.0, False], (5, 2): [8, 5, 2, -1.374675084173348, True], (5, 4): [9, 5, 4, 1.1369768644513045, True], (5, 7): [12, 5, 7, -0.18695628785579665, True], (7, 3): [13, 7, 3, 0.5542666714061728, True], (7, 4): [16, 7, 4, -3.0, True], (6, 2): [8, 6, 2, -1.4279939502396626, True]}
    #
    # net = NeuralNetwork(gen, use_single_activation_function=False)
    # t = time.time()
    # for i in range(10000):
    #     output = net(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    # print(time.time()-t)
    #
    # net = NeuralNetwork(gen, use_single_activation_function=True)
    # t = time.time()
    # for i in range(10000):
    #     output = net(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    # print(time.time()-t)
