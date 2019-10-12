import torch

from NEAT.genotype import Genotype
from NEAT.genotype_deep import Genotype_Deep
from NEAT.population_deep import Population_Deep
from naming.namegenerator import NameGenerator
from NEAT.population import Population
from results import test_rubiks
from tasks.rubikstaskRL_deep import RubiksTask_Deep
from tasks.xortaskcurriculum import XORTaskCurriculum
from tasks.rubikstaskRL import RubiksTask
from tasks.xortask import XORTask

from feedforwardnetwork import NeuralNetwork
import feedforwardnetwork_deep
import numpy as np
import matplotlib.pyplot as plt
import pickle

def make_plots(result):
    plt.figure()
    plt.plot(result['generations'], result['fitnesses'])
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    for change in result['changes']:
        idx = result['generations'].index(change)
        plt.plot(result['generations'][idx], result['fitnesses'][idx], 'ro')
    plt.show()

    plt.figure()
    plt.plot(result['generations'], result['weights'])
    plt.xlabel('Generations')
    plt.ylabel('Weight Parameters')
    for change in result['changes']:
        idx = result['generations'].index(change)
        plt.plot(result['generations'][idx], result['weights'][idx], 'ro')
    plt.show()

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
    p_add_neuron = 0.2
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
    feedforward = True
    max_depth = None
    max_nodes = float('inf')
    bias_as_node = False
    initial_weight_stdev = 0.01
    p_add_neuron = 0.1
    p_add_connection = 0.25
    p_mutate_weight = 0.1
    p_reset_weight = 0.1
    p_reenable_connection = 0.01
    p_disable_connection = 0.01
    p_reenable_parent = 0.25
    p_mutate_bias = 0.1
    p_mutate_type = 0.01
    stdev_mutate_weight = 0.01
    stdev_mutate_bias = 0.01
    weight_range = (-50.0, 50.0)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type = 'fully_connected'
    initial_sigma = 0.0

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, feedforward,
                                  max_depth, max_nodes, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, weight_range,
                                  distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
    population_size = 100
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1

    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.1
    target_species = 16
    minimum_elitism_size = 1
    young_age = 10
    young_multiplier = 1.2
    old_age = 30
    old_multiplier = 0.2
    stagnation_age = 25
    reset_innovations = False
    survival = 0.2

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                            tournament_selection_k, verbose, max_cores, compatibility_threshold,
                            compatibility_threshold_delta, target_species, minimum_elitism_size,
                            young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                            survival)

    discount_factor = 0.99

    # Task parameters
    lamarckism = False
    baldwin = False

    # Curriculum settings
    curriculum = 'LBF'
    task = RubiksTask(batch_size, device, baldwin, lamarckism, discount_factor, curriculum)
    result = population.epoch(evaluator=task, generations=4)
    make_plots(result)

    genome = result['champions'][np.argmax(np.multiply(result['stats']['fitness_max'], result['stats']['info_max']))]
    network = NeuralNetwork(genome, batch_size=1, device=device, use_single_activation_function=False)
    test_result = test_rubiks(network, device, max_tries=1000)
    print(test_result[2])

def deep_rubikstask(device, batch_size):
    # Initialise name generators for individuals in NEAT population
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Genotype Parameters
    inputs = 144
    outputs = 6
    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity', 'elu']
    feedforward = True

    genome_factory = lambda: Genotype_Deep(new_individual_name, inputs, outputs, nonlinearities, feedforward)

    # Population parameters
    population_size = 100
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1

    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.1
    target_species = 10
    minimum_elitism_size = 1
    young_age = 10
    young_multiplier = 1.2
    old_age = 30
    old_multiplier = 0.2
    stagnation_age = 10
    reset_innovations = False
    survival = 0.2

    population = Population_Deep(new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                            tournament_selection_k, verbose, max_cores, compatibility_threshold,
                            compatibility_threshold_delta, target_species, minimum_elitism_size,
                            young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                            survival)

    discount_factor = 0.99

    # Task parameters
    lamarckism = True
    baldwin = True

    # Curriculum settings
    curriculum = 'LBF'

    task = RubiksTask_Deep(batch_size, device, baldwin, lamarckism, discount_factor, curriculum)
    result = population.epoch(evaluator=task, generations=100)
    make_plots(result)

    genome = result['champions'][np.argmax(np.multiply(result['stats']['fitness_max'], result['stats']['info_max']))]
    network = feedforwardnetwork_deep.NeuralNetwork_Deep(device)
    network.create_network(genome)
    torch.save(genome, 'elites/Deep_NEAT_genome_' + genome.name + curriculum + str(baldwin) + str(lamarckism))
    torch.save(network, 'elites/Deep_NEAT_network_' + genome.name + curriculum + str(baldwin) + str(lamarckism))
    test_result = test_rubiks(network, max_tries=1000, device=device)

    pickle.dump(test_result, open('elites/test_result' + genome.name + curriculum + str(baldwin) + str(lamarckism) + '.p', "wb"))


def xortask(device, batch_size):
    batch_size = 4
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    inputs = 2
    outputs = 1
    nonlinearities = ['tanh']
    topology = None
    feedforward = True
    max_depth = None
    max_nodes = float('inf')
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
    p_mutate_type = 0.2
    stdev_mutate_weight = 1.5
    stdev_mutate_bias = 0.5
    weight_range = (-3., 3.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type = 'fully_connected'
    initial_sigma = 0.0

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight,initialisation_type, initial_sigma)

    population_size = 150
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 12
    minimum_elitism_size = 5
    young_age = 10
    young_multiplier = 1.2
    old_age = 30
    old_multiplier = 0.2
    stagnation_age = 15
    reset_innovations = False
    survival = 0.2

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved, tournament_selection_k, verbose, max_cores, compatibility_threshold, compatibility_threshold_delta, target_species, minimum_elitism_size, young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations, survival)
    task = XORTask(batch_size, device)
    result = population.epoch(evaluator=task, generations=1000, solution=task)
    print(result['champions'][-1].neuron_genes)
    print(result['champions'][-1].connection_genes)
    net = NeuralNetwork(result['champions'][-1], device=device)
    output = net(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    print(output)

if __name__ == "__main__":
    # Checks whether CUDA is available. If it is the program will run on the GPU, otherwise on the CPU.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print('Using %s' % device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
    # #
    # test_result = test_rubiks(1, device, max_tries=1000)
    # print(test_result[2])

    # Batch size of training and testing
    batch_size = 32

    # first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    # new_individual_name = first_name_generator.generate_name()
    # genome = Genotype(new_individual_name, inputs = 3, outputs = 1)
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
    deep_rubikstask(device, batch_size)


