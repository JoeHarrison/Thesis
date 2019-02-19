import numpy as np
import torch
from genotype import Genotype
from namegenerator import NameGenerator
from population import Population
from feedforwardnetwork import NeuralNetwork
from xortask import XORTask
from rubikstask import RubiksTask

def rubikstask(device):
    first_name_generator = NameGenerator('names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    inputs = 144
    outputs = 12
    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity']
    topology = None
    feedforward = True
    max_depth = None
    max_nodes = float('inf')
    response_default = 1.0
    bias_as_node = False
    initial_weight_stdev = 1.0
    p_add_neuron = 0.1
    p_add_connection = 0.25
    p_mutate_weight = 0.75
    p_reset_weight = 0.1
    p_reenable_connection = 0.01
    p_disable_connection = 0.01
    p_reenable_parent = 0.25
    p_mutate_bias = 0.25
    p_mutate_response = 0.0
    p_mutate_type = 0.05
    stdev_mutate_weight = 1.0
    stdev_mutate_bias = 1.0
    stdev_mutate_response = 0.5
    weight_range = (-1., 1.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight)

    population_size = 128
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 16
    minimum_elitism_size = 5
    young_age = 10
    young_multiplier = 1.2
    old_age = 30
    old_multiplier = 0.2
    stagnation_age = 15
    reset_innovations = False
    survival = 0.2

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved, tournament_selection_k, verbose, max_cores, compatibility_threshold, compatibility_threshold_delta, target_species, minimum_elitism_size, young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations, survival)
    task = RubiksTask(batch_size=128, device=device, lamarckism=True)
    result = population.epoch(evaluator = task, generations = 1000, solution = task)
    print(result['champions'][-1].neuron_genes)
    print(result['champions'][-1].connection_genes)

def xortask():
    first_name_generator = NameGenerator('names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    inputs = 2
    outputs = 1
    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity']
    topology = None
    feedforward = True
    max_depth = None
    max_nodes = float('inf')
    response_default = 1.0
    bias_as_node = False
    initial_weight_stdev = 2.0
    p_add_neuron = 0.03
    p_add_connection = 0.3
    p_mutate_weight = 0.8
    p_reset_weight = 0.1
    p_reenable_connection = 0.01
    p_disable_connection = 0.01
    p_reenable_parent=0.25
    p_mutate_bias = 0.2
    p_mutate_response = 0.0
    p_mutate_type = 0.2
    stdev_mutate_weight = 1.5
    stdev_mutate_bias = 0.5
    stdev_mutate_response = 0.5
    weight_range = (-1., 1.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight)

    population_size = 512
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 16
    minimum_elitism_size = 5
    young_age = 10
    young_multiplier = 1.2
    old_age = 30
    old_multiplier = 0.2
    stagnation_age = 15
    reset_innovations = False
    survival = 0.2

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved, tournament_selection_k, verbose, max_cores, compatibility_threshold, compatibility_threshold_delta, target_species, minimum_elitism_size, young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations, survival)
    task = XORTask()
    result = population.epoch(evaluator = task, generations = 1000, solution = task)
    print(result['champions'][-1].neuron_genes)
    print(result['champions'][-1].connection_genes)

if __name__ == "__main__":
    # np.random.seed(3)
    # torch.manual_seed(3)
    # random.seed(3)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print('Using %s' % device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    first_name_generator = NameGenerator('names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    genome = Genotype(new_individual_name, 2, 1)

    genome.neuron_genes = [[0, 'tanh', 1.0, 0, 0, 1.0], [1, 'tanh', 1.0, 0, 2048, 1.0], [2, 'sigmoid', -1.7126596576379352, 2048, 4096, 1.0], [3, 'tanh', -0.8893340119071926, 1, 2048.0, 1.0], [4, 'tanh', 2.354260308870598, 1, 3072.0, 1.0], [5, 'identity', 0.644710733714112, 1, 3072.0, 1.0], [6, 'identity', 1.4286189036280563, 1, 1536.0, 1.0], [7, 'sigmoid', 2.5686020859248497, 1, 2560.0, 1.0], [8, 'sigmoid', 1.0, 2, 3584.0, 1.0]]
    genome.connection_genes = {(0, 2): [0, 0, 2, 0.9570420296626129, True], (1, 2): [1, 1, 2, 6.394578277464045, True], (3, 2): [3, 3, 2, 5.0302733690649895, True], (0, 4): [4, 0, 4, -5.599159832810173, True], (4, 2): [5, 4, 2, -8.128200451027551, True], (1, 4): [6, 1, 4, 7.973423212790442, True], (5, 2): [8, 5, 2, 3.452690602189743, True], (0, 5): [9, 0, 5, -1.7373626007990397, True], (0, 6): [13, 0, 6, 8.990206888452919, True], (1, 7): [15, 1, 7, 2.209906926826694, True], (7, 2): [18, 7, 2, 2.290157694853284, True], (3, 8): [22, 3, 8, 0.5174994659566082, True], (8, 2): [23, 8, 2, 5.256223627935843, True], (5, 8): [28, 5, 8, 4.100292144020446, True]}

    # print(genome.neuron_genes)
    # print(genome.connection_genes)

    network = NeuralNetwork(genome)
    # print(network(np.array([[0,0]])))


    # print(output)
    rubikstask(device)
