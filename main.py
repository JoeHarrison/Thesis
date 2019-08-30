import torch
import torch.nn as nn

from NEAT.genotype import Genotype
from naming.namegenerator import NameGenerator
from NEAT.population import Population
from tasks.xortaskcurriculum import XORTaskCurriculum
from tasks.rubikstask2 import RubiksTask

from reinforcement_learning.replay_memories import ReplayMemory, PrioritizedReplayMemory
from feedforwardnetwork import NeuralNetwork
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict

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

def xortaskcurriculum(device, batch_size, baldwin, lamarckism, verbose):
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    inputs = 3
    outputs = 1

    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity', 'elu']
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
    p_reenable_parent = 0.25
    p_mutate_bias = 0.2
    p_mutate_response = 0.0
    p_mutate_type = 0.01
    stdev_mutate_weight = 1.0
    stdev_mutate_bias = 0.5
    stdev_mutate_response = 0.5
    weight_range = (-3.0, 3.0)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type = 'partially_connected'
    initial_sigma = 0.00

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight,initialisation_type, initial_sigma)

    population_size = 150
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3

    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 15
    minimum_elitism_size = 5
    young_age = 10
    young_multiplier = 1.2
    old_age = 30
    old_multiplier = 0.2
    stagnation_age = 15
    reset_innovations = False
    survival = 0.2

    # Setting this to true will limit the network to using relu only
    use_single_activation_function = False

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved, tournament_selection_k, verbose, max_cores, compatibility_threshold, compatibility_threshold_delta, target_species, minimum_elitism_size, young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations, survival)
    task = XORTaskCurriculum(batch_size, device, baldwin, lamarckism, use_single_activation_function)
    result = population.epoch(evaluator=task, generations=1000, solution=task)

    if result['stats']['solved'][-1]:
        individual = result['champions'][-1]
    else:
        individual = result['champions'][np.argmax(np.multiply(result['stats']['fitness_max'], result['stats']['info_max']))]

    if(baldwin and not lamarckism):
        task.lamarckism = True
        individual = task.backprop(individual)

    net = NeuralNetwork(individual, device=device, use_single_activation_function=use_single_activation_function)

    net.reset()

    criterion = torch.nn.MSELoss()

    TARGETSOR = torch.tensor([[0.0], [1.0], [1.0], [1.0]], device=device)

    output = net(torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]))
    OR_loss = 1.0/(1.0+criterion(output, TARGETSOR))

    net.reset()

    TARGETSXOR = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

    output = net(torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
    XOR_loss = 1.0/(1.0+criterion(output, TARGETSXOR))

    return OR_loss, XOR_loss, result['champions'][-1], population.generation

def rubikstask(device, batch_size):
    # Initialise name generators for individuals in NEAT population
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Genotype Parameters
    inputs = 144
    outputs = 6
    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity', 'elu']
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
    p_reenable_parent = 0.25
    p_mutate_bias = 0.2
    p_mutate_response = 0.0
    p_mutate_type = 0.01
    stdev_mutate_weight = 1.0
    stdev_mutate_bias = 0.5
    stdev_mutate_response = 0.5
    weight_range = (-3.0, 3.0)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.6

    initialisation_type = 'fully_connected'
    initial_sigma = 0.00

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
    population_size = 250
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

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                            tournament_selection_k, verbose, max_cores, compatibility_threshold,
                            compatibility_threshold_delta, target_species, minimum_elitism_size,
                            young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                            survival)

    # Reinforcement Learning parameters
    memory = PrioritizedReplayMemory(100000)

    discount_factor = 0.99

    # Task parameters
    lamarckism = False
    baldwin = False

    # Curriculum settings
    curriculum = 'Naive'

    task = RubiksTask(batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum)
    result = population.epoch(evaluator=task, generations=1000, solution=task)

def rubikstasktune(device, batch_size):
    # Initialise name generators for individuals in NEAT population
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Immutable parameters genotype
    inputs = 144
    outputs = 6
    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity', 'elu']
    topology = None
    feedforward = True
    max_depth = None
    max_nodes = float('inf')
    response_default = 1.0
    bias_as_node = False
    p_mutate_response = 0.0
    initial_sigma = 0.00

    # Immutable parameters population
    verbose = False
    max_cores = 1
    reset_innovations = False
    stop_when_solved = True

    baldwin = False
    lamarckism = False
    discount_factor = 0.99
    memory = None
    curriculum = 'Naive'

    best_level = 0
    best_fitness = 0.0
    best_geno = None
    best_pop = None

    try:
        for i in tqdm(range(100)):
            # Tunable parameters genotype
            initial_weight_stdev = np.random.uniform(0.1, 2.0)
            p_add_neuron = np.random.rand()
            p_add_connection = np.random.rand()
            p_mutate_weight = np.random.rand()
            p_reset_weight = np.random.rand()
            p_reenable_connection = np.random.rand()
            p_disable_connection = np.random.rand()
            p_reenable_parent = np.random.rand()
            p_mutate_bias = np.random.rand()

            p_mutate_type = np.random.rand()
            stdev_mutate_weight = np.random.uniform(0.1, 2.0)
            stdev_mutate_bias = np.random.uniform(0.1, 2.0)
            stdev_mutate_response = np.random.uniform(0.1, 2.0)
            weight_range = (-np.random.uniform(0.5, 2.0), np.random.uniform(0.5, 2.0))

            distance_excess_weight = np.random.rand()
            distance_disjoint_weight = np.random.rand()
            distance_weight = np.random.rand()

            initialisation_type = np.random.choice(['partially_connected', 'fully_connected'])

            print('initial_weight_stdev', initial_weight_stdev)
            print('p_add_neuron', p_add_neuron)
            print('p_add_connection', p_add_connection)
            print('p_mutate_weight', p_mutate_weight)
            print('p_reset_weight', p_reset_weight)
            print('p_reenable_connection', p_reenable_connection)
            print('p_disable_connection', p_disable_connection)
            print('p_reenable_parent', p_reenable_parent)
            print('p_mutate_bias', p_mutate_bias)

            print('p_mutate_type', p_mutate_type)
            print('stdev_mutate_weight', stdev_mutate_weight)
            print('stdev_mutate_bias', stdev_mutate_bias)
            print('stdev_mutate_response', stdev_mutate_response)
            print('weight_range', weight_range)

            print('distance_excess_weight', distance_excess_weight)
            print('distance_disjoint_weight', distance_disjoint_weight)
            print('distance_weight', distance_weight)
            print('initialisation_type', initialisation_type)


            # Tunable parameters population
            population_size = np.random.randint(50, 151)
            elitism = np.random.choice([True, False])
            tournament_selection_k = np.random.randint(1, 5)

            compatibility_threshold = np.random.uniform(1.0, 5.0)
            compatibility_threshold_delta = np.random.rand()
            target_species = np.random.randint(5, 21)
            minimum_elitism_size = np.random.randint(1, 11)
            young_age = np.random.randint(5, 16)
            young_multiplier = 1.0 + np.random.rand()
            old_age = np.random.randint(20, 41)
            old_multiplier = np.random.rand()
            stagnation_age = np.random.randint(10, 31)
            survival = np.random.rand()

            print('population_size', population_size)
            print('elitism', elitism)
            print('tournament_selection_k', tournament_selection_k)

            print('compatibility_threshold', compatibility_threshold)
            print('compatibility_threshold_delta', compatibility_threshold_delta)
            print('target_species', target_species)
            print('minimum_elitism_size', minimum_elitism_size)
            print('young_age', young_age)
            print('young_multiplier', young_multiplier)
            print('old_age', old_age)
            print('old_multiplier', old_multiplier)
            print('stagnation_age', stagnation_age)
            print('survival', survival)


            genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                      max_depth, max_nodes, response_default, initial_weight_stdev,
                                      bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                      p_reset_weight, p_reenable_connection, p_disable_connection,
                                      p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                      stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                      weight_range, distance_excess_weight, distance_disjoint_weight,
                                      distance_weight, initialisation_type, initial_sigma)

            population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                                tournament_selection_k, verbose, max_cores, compatibility_threshold,
                                compatibility_threshold_delta, target_species, minimum_elitism_size,
                                young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                                survival)

            task = RubiksTask(batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum)
            result = population.epoch(evaluator=task, generations=100, solution=task)
            print('Level: ', result['stats']['info_max'][-1])
            print('Fitness: ', result['stats']['fitness_max'][-1])
            if result['stats']['info_max'][-1] > best_level:
                best_level = result['stats']['info_max'][-1]
                best_fitness = result['stats']['fitness_max'][-1]
                best_geno = (new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                      max_depth, max_nodes, response_default, initial_weight_stdev,
                                      bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                      p_reset_weight, p_reenable_connection, p_disable_connection,
                                      p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                      stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                      weight_range, distance_excess_weight, distance_disjoint_weight,
                                      distance_weight, initialisation_type, initial_sigma)
                best_pop = (new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                                tournament_selection_k, verbose, max_cores, compatibility_threshold,
                                compatibility_threshold_delta, target_species, minimum_elitism_size,
                                young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                                survival)
            if result['stats']['info_max'][-1] == best_level and result['stats']['fitness_max'][-1]> best_fitness:
                best_fitness = result['stats']['fitness_max'][-1]
                best_geno = (new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                      max_depth, max_nodes, response_default, initial_weight_stdev,
                                      bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                      p_reset_weight, p_reenable_connection, p_disable_connection,
                                      p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                      stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                      weight_range, distance_excess_weight, distance_disjoint_weight,
                                      distance_weight, initialisation_type, initial_sigma)
                best_pop = (new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                                tournament_selection_k, verbose, max_cores, compatibility_threshold,
                                compatibility_threshold_delta, target_species, minimum_elitism_size,
                                young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                                survival)
    except:
        pass
    print(best_level, best_fitness)
    print(best_geno)
    print(best_pop)

if __name__ == "__main__":
    # Checks whether CUDA is available. If it is the program will run on the GPU, otherwise on the CPU.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print('Using %s' % device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Batch size of training and testing
    batch_size = 100

    # first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    # new_individual_name = first_name_generator.generate_name()
    # genome = Genotype(new_individual_name, inputs = 3, outputs = 1)
    #
    #
    #
    #
    # OR_losses = []
    # XOR_losses = []
    # individuals = []
    # generations = []
    #
    # n_neurons = []
    # total_neurons = []
    # n_connections = []
    #
    # for i in tqdm(range(100)):
    #     random.seed(i)
    #     np.random.seed(i)
    #     torch.manual_seed(i)
    #
    #     OR_loss, XOR_loss, individual, generation = xortaskcurriculum(device, 100, False, False, False)
    #     OR_losses.append(OR_loss.item())
    #     XOR_losses.append(XOR_loss.item())
    #     individuals.append(individual)
    #     generations.append(generation)
    #
    #     tmp_neurons = defaultdict(int)
    #     req = required_for_output(individual.input_keys, individual.output_keys, individual.connection_genes)
    #     for neuron in individual.neuron_genes:
    #         if neuron[0] in req and neuron[0] not in individual.output_keys:
    #             layer = neuron[3]
    #             tmp_neurons[layer] += 1
    #
    #     n_neurons.append(tmp_neurons)
    #
    #     total_neurons.append(len(required_for_output(individual.input_keys, individual.output_keys, individual.connection_genes)) + len(individual.input_keys) + len(individual.output_keys))
    #     n_connections.append(np.sum([1 for conn in individual.connection_genes.values() if conn[4]]))
    #     print(i, OR_loss, XOR_loss, generation)
    #
    # print('Average OR Loss:', np.average(OR_losses))
    # print('Average XOR Loss:', np.average(XOR_losses))
    # print('Average Loss: ', np.average([np.average(OR_losses), np.average(XOR_losses)]))
    # print('Std Loss: ', np.std([np.average(OR_losses), np.average(XOR_losses)]))
    # print('Score: ', np.average([np.average(OR_losses), np.average(XOR_losses)]) - np.std([np.average(OR_losses), np.average(XOR_losses)]))
    # print('Average number of Generations:', np.average(generations))
    # print('Average total number of required Neurons:', np.average(total_neurons))
    #
    # avg_per_layer = defaultdict(int)
    # for n_neuron in n_neurons:
    #     for key in list(n_neuron.keys()):
    #         avg_per_layer[key] += n_neuron[key]
    # print('Average number of neurons per layer')
    # for key in sorted(list(avg_per_layer.keys())):
    #     print('Layer: ', key, 'average number of neurons: ', np.sum(avg_per_layer[key])/len(n_neurons))
    #
    # print('Average Number of connections: ', np.average(n_connections))
    #
    # print('---Genes and stats of individual with Lowest loss averaged over XOR and OR task---')
    # combined_losses = (np.array(OR_losses) + np.array(XOR_losses))/2.0
    # best_individual = individuals[np.argmax(combined_losses)]
    # print(OR_losses[np.argmax(combined_losses)])
    # print(XOR_losses[np.argmax(combined_losses)])
    # print(best_individual.neuron_genes)
    # print(best_individual.connection_genes)
    #
    # for key in sorted(list(n_neurons[np.argmax(combined_losses)].keys())):
    #     print('Layer: ', key, 'number of neurons: ', n_neurons[np.argmax(combined_losses)][key])
    #
    # number_enabled_connections = np.sum([1 for conn in best_individual.connection_genes.values() if conn[4]])
    # print('Total number of Enabled connections: ', number_enabled_connections)
    # print('Number of Generations', generations[np.argmax(combined_losses)])

    rubikstask(device, batch_size)
    # rubikstasktune(device, batch_size)
