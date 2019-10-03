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
    def __init__(self, device):
        super(NeuralNetwork_Deep, self).__init__()

        self.device = device
        self.modules = []
        self.model = None
        self.nodes = None
        self.connections = None

    def forward(self, x):
        intermediate_inputs = {}

        # All enabled connections from the input
        conn_mods = [(input_node, output_node) for (input_node, output_node) in self.connections.keys()
                     if input_node == 0 and self.connections[(input_node, output_node)][3]]

        for conn in conn_mods:
            input_neuron, output_neuron = conn
            intermediate_inputs[output_neuron] = self.activ[str(conn)](self.conn[str(conn)](x))

        copy_nodes = copy.deepcopy(self.nodes)
        copy_nodes.sort(key=lambda k: k['fforder'])
        for node in copy_nodes[1:]:
            node_id = node['node_id']

            # Retrieve all connection modules that share the same input node and are enabled
            conn_mods = [(input_node, output_node) for (input_node, output_node) in self.connections.keys()
                         if input_node == node_id and self.connections[(input_node, output_node)][3]]

            for conn in conn_mods:
                input_neuron, output_neuron = conn

                if output_neuron not in intermediate_inputs.keys():
                    intermediate_inputs[output_neuron] = self.conn[str(conn)](intermediate_inputs[(input_neuron)])
                else:
                    # If input for upcoming node already exists: add the inputs together
                    intermediate_inputs[output_neuron] += self.conn[str(conn)](intermediate_inputs[(input_neuron)])

        return intermediate_inputs[1]

    def apply_weights(self, connection):
        if self.connections[connection][4] is None:
            self.connections[connection][4] = self.conn[str(connection)].weight.data
        else:
            rows = self.conn[str(connection)].weight.data.size(0)
            cols = self.conn[str(connection)].weight.data.size(1)

            # Add row if necessary
            difference = rows - self.connections[connection][4].size(0)
            if difference > 0:
                self.connections[connection][4] = torch.cat((self.connections[connection][4], torch.zeros(difference, self.connections[connection][4].size(1), device=self.device)), 0)

            # Add column if necessary
            difference = cols - self.connections[connection][4].size(1)
            if difference > 0:
                self.connections[connection][4] = torch.cat((self.connections[connection][4], torch.zeros(self.connections[connection][4].size(0), difference, device=self.device)), 1)

            self.conn[str(connection)].weight.data = self.connections[connection][4][:rows, :cols]

    def apply_bias(self, connection):
        if self.connections[connection][5] is None:
            self.connections[connection][5] = self.conn[str(connection)].bias.data
        else:
            cols = self.conn[str(connection)].bias.data.size(0)

            difference = cols - self.connections[connection][5].size(0)
            if difference > 0:
                self.connections[connection][5] = torch.cat((self.connections[connection][5], torch.zeros(difference, device=self.device)), 0)

            self.conn[str(connection)].bias.data = self.connections[connection][5][:cols]

    def create_genome_from_network(self):
        for connection in self.connections.keys():
            if self.connections[connection][3]:
                self.connections[connection][4] = self.conn[str(connection)].weight.data
                self.connections[connection][5] = self.conn[str(connection)].bias.data

        return self.connections

    def create_network(self, genome):
        self.nodes = copy.deepcopy(genome.nodes)

        self.connections = genome.connections

        # Initialise separate Module dictionaries for the connection weights and activations
        self.conn = nn.ModuleDict({})
        self.activ = nn.ModuleDict({})

        for connection in self.connections.keys():
            if not self.connections[connection][3]:
                continue

            input_neuron, output_neuron = connection

            self.conn[str(connection)] = nn.Linear(self.nodes[input_neuron]['num_nodes'], self.nodes[output_neuron]['num_nodes'])
            self.activ[str(connection)] = string_to_activation[self.nodes[output_neuron]['activation_function']]
            self.apply_weights(connection)
            self.apply_bias(connection)

    def act(self, state, epsilon, mask, device):
        if np.random.rand() > epsilon:
            state = torch.tensor([state], dtype=torch.float32, device=device)
            mask = torch.tensor([mask], dtype=torch.float32, device=device)
            q_values = self.forward(state) + mask
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = np.random.randint(self.num_actions)
        return action

