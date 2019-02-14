import random
import numpy as np
from itertools import product
from copy import deepcopy

class Genotype(object):
    def __init__(self, 
                 new_individual_name,
                 inputs = 144, 
                 outputs = 12, 
                 nonlinearities = ['relu','sigmoid','tanh'],
                 topology = None,
                 feedforward = True,
                 max_depth = None,
                 max_nodes = float('inf'),
                 response_default = 1.0,
                 initial_weight_stdev = 2.0,
                 bias_as_node = False,
                 p_add_neuron = 0.03, 
                 p_add_connection = 0.3, 
                 p_mutate_weight = 0.8,
                 p_reset_weight = 0.1,
                 p_reenable_connection = 0.01,
                 p_disable_connection = 0.01, 
                 p_reenable_parent = 0.25, 
                 p_mutate_bias = 0.2,
                 p_mutate_response = 0.0,
                 p_mutate_type = 0.2,
                 stdev_mutate_weight = 1.5,
                 stdev_mutate_bias = 0.5,
                 stdev_mutate_response = 0.5,
                 weight_range = (-50.,50.),
                 distance_excess_weight = 1.0, 
                 distance_disjoint_weight = 1.0, 
                 distance_weight = 0.4):
        
        self.name = next(new_individual_name)
        self.specie = None
        
        self.inputs = inputs
        self.outputs = outputs
        self.nonlinearities = nonlinearities
        self.feedforward = feedforward
        self.bias_as_node = bias_as_node
        
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        self.response_default = response_default
        self.initial_weight_stdev = initial_weight_stdev
        self.stdev_mutate_weight = stdev_mutate_weight
        self.stdev_mutate_bias = stdev_mutate_bias
        self.stdev_mutate_response = stdev_mutate_response
        self.weight_range = weight_range
        
        # Mutation Probabilities
        self.p_add_neuron = p_add_neuron
        self.p_add_connection = p_add_connection
        self.p_mutate_weight = p_mutate_weight
        self.p_reset_weight = p_reset_weight
        self.p_reenable_connection = p_reenable_connection
        self.p_disable_connection = p_disable_connection
        self.p_reenable_parent = p_reenable_parent
        self.p_mutate_bias = p_mutate_bias
        self.p_mutate_response = p_mutate_response
        self.p_mutate_type = p_mutate_type
        
        # Distance weights
        self.distance_excess_weight = distance_excess_weight
        self.distance_disjoint_weight = distance_disjoint_weight
        self.distance_weight = distance_weight
        
        # Tuples of: id, non_linearity, bias, layer, ff_order, response
        self.neuron_genes = []
        # Tuples of: innovation number, input, output, weight, enabled
        self.connection_genes = {}
        # Hyperparameter genes
        self.hyperparameter_genes = []
        
        self.input_keys = []
        self.output_keys = []
        
        self._initialise_topology(topology)
    
    def change_specie(self,specie):
        self.specie = specie
        
    def _initialise_topology(self, topology):
#         if self.bias_as_node:
#             self.inputs += 1
        
        self.max_layer = 2048 if (self.max_depth is None) else (self.max_depth - 1)
        
        if topology is None:
            # Initialise inputs
            for i in range(self.inputs):
                self.neuron_genes.append([i, random.choice(self.nonlinearities),1.0,0, i * 2048, self.response_default])
                self.input_keys.append(i)
            # Initialise outputs
            for i in range(self.outputs):
                self.neuron_genes.append([(self.inputs + i), random.choice(self.nonlinearities),1.0,self.max_layer, (self.inputs + i) * 2048, self.response_default])
                self.output_keys.append((self.inputs + i))
            # Initialise connections
            innovation_number = 0
            for i in range(self.inputs):
                for j in range(self.inputs,self.inputs + self.outputs):
                    weight = self._initialise_weight(self.inputs,self.outputs)
                    self.connection_genes[(i,j)] = [innovation_number, i, j, weight ,True]
                    innovation_number += 1
        else:
            raise NotImplementedError

                
    def _initialise_weight(self, input_neurons, output_neurons):
        weight = np.random.rand()*np.sqrt(1/(input_neurons + output_neurons))
        return weight
        
    def recombinate(self, other):
        child = deepcopy(self)
        child.neuron_genes = []
        child.connection_genes = {}
        
        max_neurons = max(len(self.neuron_genes), len(other.neuron_genes))
        min_neurons = min(len(self.neuron_genes), len(other.neuron_genes))
        
        for i in range(max_neurons):
            neuron_gene = None
            if i < min_neurons:
                neuron_gene = random.choice((self.neuron_genes[i], other.neuron_genes[i]))
            else:
                try:
                    neuron_gene = self.neuron_genes[i]
                except IndexError:
                    neuron_gene = other.neuron_genes[i]
            child.neuron_genes.append(deepcopy(neuron_gene))
            
        self_connections = dict(((c[0], c) for c in self.connection_genes.values()))
        other_connections = dict(((c[0], c) for c in other.connection_genes.values()))
        max_innovation_number = max(list(self_connections.keys()) + list(other_connections.keys()))
        
        for i in range(max_innovation_number + 1):
            connection_gene = None
            if i in self_connections and i in other_connections:
                connection_gene = random.choice((self_connections[i],other_connections[i]))
                enabled = self_connections[i][4] and other_connections[i][4]
            else:
                if i in self_connections:
                    connection_gene = self_connections[i]
                    enabled = connection_gene[4]
                elif i in other_connections:
                    connection_gene = other_connections[i]
                    enabled = connection_gene[4]
            if connection_gene is not None:
                child.connection_genes[(connection_gene[1],connection_gene[2])] = deepcopy(connection_gene)
                child.connection_genes[(connection_gene[1],connection_gene[2])][4] = enabled or np.random.rand() < self.p_reenable_parent

            def is_feedforward(item):
                ((fr, to), cg) = item
                return child.neuron_genes[fr][3] < child.neuron_genes[to][3] and child.neuron_genes[fr][4] < child.neuron_genes[to][4]

            if self.feedforward:
                child.connection_genes = dict(filter(is_feedforward, child.connection_genes.items()))
        return child
    
    def add_neuron(self, maximum_innovation_number, innovations):
        possible_to_split = self.connection_genes.keys()
            
        if self.max_depth is not None:
            possible_to_split = [(fr, to) for (fr, to) in possible_to_split if self.neuron_genes[fr][3] + 1 < self.neuron_genes[to][3]]
        else:
            possible_to_split = list(possible_to_split)
        #possible_to_split = [(fr,to) for (fr, to) in possible_to_split if self.neuron_genes[fr][3] + 1 < self.neuron_genes[to][3]]
        
        
        if possible_to_split:
            # Choose connection to split
            split_neuron = self.connection_genes[random.choice(possible_to_split)]
            # Disable old connection
            split_neuron[4] = False

            input_neuron, output_neuron, weight = split_neuron[1:4]
            fforder = (self.neuron_genes[input_neuron][4] + self.neuron_genes[output_neuron][4]) * 0.5
            nonlinearity = random.choice(self.nonlinearities)
            layer = self.neuron_genes[input_neuron][3] + 1
            

            new_id = len(self.neuron_genes)

            neuron = [new_id, nonlinearity, 1.0, layer, fforder, self.response_default]

            self.neuron_genes.append(neuron)

            if (input_neuron, new_id) in innovations:
                innovation_number = innovations[(input_neuron,new_id)]
            else:
                maximum_innovation_number += 1
                innovation_number = innovations[(input_neuron,new_id)] = maximum_innovation_number

            # 1.0 to initialise_weight?
            self.connection_genes[(input_neuron, new_id)] = [innovation_number, input_neuron, new_id, 1.0, True]

            if (new_id, output_neuron) in innovations:
                innovation_number = innovations[(new_id, output_neuron)]
            else:
                maximum_innovation_number += 1
                innovation_number = innovations[(new_id, output_neuron)] = maximum_innovation_number

            self.connection_genes[(new_id, output_neuron)] = [innovation_number, new_id, output_neuron, weight, True]
    
    def add_connection(self, maximum_innovation_number, innovations):
        potential_connections = product(range(len(self.neuron_genes)),range(self.inputs, len(self.neuron_genes)))
        potential_connections = (connection for connection in potential_connections if connection not in self.connection_genes)

        if self.feedforward:
            potential_connections = ((f, t) for (f, t) in potential_connections if self.neuron_genes[f][3] < self.neuron_genes[t][3] and self.neuron_genes[f][4] < self.neuron_genes[t][4])

        potential_connections = list(potential_connections)
        
        if potential_connections:
            (fr, to) = random.choice(potential_connections)
            if (fr, to) in innovations:
                innovation = innovations[(fr, to)]
            else:
                maximum_innovation_number += 1
                innovation = innovations[(fr, to)] = maximum_innovation_number
            # get number of neurons in layers of fr and to
            connection_gene = [innovation, fr, to, self._initialise_weight(2,2), True]
            self.connection_genes[(fr, to)] = connection_gene
    
    def mutate(self, innovations = {}, global_innovation_number = 0):
        
        maximum_innovation_number = max(global_innovation_number, max(cg[0] for cg in self.connection_genes.values()))
        # TODO: move to separate functions
        if len(self.neuron_genes) < self.max_nodes and np.random.rand() < self.p_add_neuron:
            self.add_neuron(maximum_innovation_number, innovations)
                
        elif np.random.rand() < self.p_add_connection:
            self.add_connection(global_innovation_number, innovations)
            
        else:
            for cg in self.connection_genes.values():
                if np.random.rand() < self.p_mutate_weight:
                    cg[3] += np.random.normal(0.0, self.stdev_mutate_weight)
                    cg[3] = np.clip(cg[3], self.weight_range[0], self.weight_range[1])
                    # clipping?
                if np.random.rand() < self.p_reset_weight:
                    cg[3] = np.random.normal(0.0,self.stdev_mutate_weight)
                    
                # bigger chance to disable in this way
                if np.random.rand() < self.p_reenable_connection:
                    cg[4] = True
                    
                if np.random.rand() < self.p_disable_connection:
                    cg[4] = False
                    
            for neuron_gene in self.neuron_genes[self.inputs:]:
                if np.random.rand() < self.p_mutate_bias:
                    neuron_gene[2] += np.random.normal(0.0, 1)

                    neuron_gene[2] = np.clip(neuron_gene[2], self.weight_range[0], self.weight_range[1])
                
                if np.random.rand() < self.p_mutate_type:
                    neuron_gene[1] = random.choice(self.nonlinearities)
                    
                # if np.random.rand() < self.p_mutate_response:
                #     neuron_gene[5] += np.random.normal(0.0, self.stdev_mutate_response)
                    
        return self
        
    def distance(self, other):
        self_connections = dict(((c[0], c) for c in self.connection_genes.values()))
        other_connections = dict(((c[0], c) for c in other.connection_genes.values()))

        all_innovations = list(self_connections.keys()) + list(other_connections.keys())

        minimum_innovation = min(all_innovations)
        
        e = 0
        d = 0
        w = 0.0
        m = 0
        
        for innovation_key in all_innovations:
            if innovation_key in self_connections and innovation_key in other_connections:
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
        w = (w/m) if m>0 else w
        
        return (self.distance_excess_weight * e +
               self.distance_disjoint_weight * d +
               self.distance_weight * w)
