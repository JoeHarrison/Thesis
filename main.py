import torch
import torch.nn as nn

from NEAT.genotype import Genotype
from NEAT.genotype_deep import Genotype_Deep
from NEAT.population_deep import Population_Deep
from naming.namegenerator import NameGenerator
from NEAT.population import Population
from tasks.rubikstaskRL_deep import RubiksTask_Deep
from tasks.xortaskcurriculum import XORTaskCurriculum
from tasks.rubikstaskRL import RubiksTask
from tasks.cartpoletask import CartpoleTask
from tasks.xortask import XORTask

from reinforcement_learning.replay_memories import ReplayMemory, PrioritizedReplayMemory
from feedforwardnetwork import NeuralNetwork
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from superflip import superflip_set
import rubiks2
import time

def maxDistance(arr):

    # Used to store element to first index mapping
    mp = {}

    # Traverse elements and find maximum distance between
    # same occurrences with the help of map.
    maxDict = 0
    maxFirst_idx = 0
    for i in range(len(arr)):

        # If this is first occurrence of element, insert its
        # index in map
        if arr[i] not in mp.keys():
            mp[arr[i]] = i

        # Else update max distance
        else:
            if i-mp[arr[i]]>maxDict:
                maxDict = max(maxDict, i-mp[arr[i]])
                maxFirst_idx = mp[arr[i]]

    return maxDict, maxFirst_idx

def test_rubiks(network, device, max_tries=None):
    network = torch.load('models/network_Palme6')

    solve_rate_superflip = np.zeros(14)
    counts_superflip = np.zeros(14)
    seq_len_superflip = np.zeros(14)
    seq_len_superflip_heur = np.zeros(14)
    puzzles = []
    solution_sequences = []

    try:
        for i in range(14):
            hashes_seqs = []
            if i > 0:
                if solve_rate_superflip[i-1] == 0:
                    break

            for sequence in tqdm(superflip_set):
                env = rubiks2.RubiksEnv2(2, unsolved_reward=-1.0)

                hashed_sequence = hash(str(sequence[:i+1]))

                if hashed_sequence not in hashes_seqs:

                    hashes_seqs.append(hashed_sequence)

                    counts_superflip[i] += 1

                    puzzle = []
                    for j in range(i + 1):
                        env.step(int(sequence[j]))
                        puzzle.append(sequence[j])

                    puzzles.append(puzzle)

                    hashes = defaultdict(list)
                    done = 0
                    tries = 0
                    t = time.time()
                    state = env.get_observation()
                    hashes[hash(state.tostring())] = [0]*env.action_space.n
                    stop = False

                    solution_sequence = []
                    state_hash_seq = []
                    while time.time()-t < 1.21 and not done and not stop:
                        mask = hashes[hash(state.tostring())]
                        state_hash_seq.append(hash(state.tostring()))
                        action = network.act(state, 0.0, mask, device)
                        solution_sequence.append(action)

                        next_state, reward, done, info = env.step(action)

                        hstate = state.copy()
                        state = next_state
                        h = hash(state.tostring())
                        if h in hashes.keys():
                            hashes[hash(hstate.tostring())][action] = -999
                        else:
                            hashes[h] = [0]*env.action_space.n

                        tries += 1
                        if max_tries:
                            if tries >= max_tries:
                                stop = True

                    length, first_idx = maxDistance(state_hash_seq)

                    while length > 0:
                        state_hash_seq = state_hash_seq[:first_idx] + state_hash_seq[first_idx + length:]
                        solution_sequence = solution_sequence[:first_idx] + solution_sequence[first_idx + length:]
                        length, first_idx = maxDistance(state_hash_seq)

                    solution_sequences.append(solution_sequence)
                    solve_rate_superflip[i] += done

                    if done:
                        seq_len_superflip[i] += tries
                        seq_len_superflip_heur[i] += len(solution_sequence)

    except KeyboardInterrupt:
        pass

    score = np.zeros(14)
    solve_rate = np.divide(solve_rate_superflip, counts_superflip)
    seq_len = np.divide(seq_len_superflip, solve_rate_superflip)
    for i in range(14):
        score[i] = solve_rate[i] / (1+(seq_len[i] - (i+1)))

    score = np.mean(score) - np.std(score)

    return (np.mean(np.divide(solve_rate_superflip, counts_superflip))-np.std(np.divide(solve_rate_superflip, counts_superflip)),
            score,
            np.divide(solve_rate_superflip, counts_superflip),
np.divide(seq_len_superflip, solve_rate_superflip),
            np.divide(seq_len_superflip_heur, solve_rate_superflip),
           puzzles, solution_sequences)

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
    # nonlinearities = ['tanh']
    nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity', 'elu']
    topology = None
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

    initialisation_type = 'partially_connected'
    initial_sigma = 0.0

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, weight_range,
                                  distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
    population_size = 250
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1

    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.1
    target_species = 32
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

    # Reinforcement Learning parameters
    memory = PrioritizedReplayMemory(100000)

    discount_factor = 0.99

    # Task parameters
    lamarckism = True
    baldwin = True

    # Curriculum settings
    curriculum = 'LBF'

    task = RubiksTask(batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum)
    result = population.epoch(evaluator=task, generations=14*6*100)
    genome = result['champions'][np.argmax(np.multiply(result['stats']['fitness_max'], result['stats']['info_max']))]
    network = NeuralNetwork(genome, batch_size=1, device=device, use_single_activation_function=False)
    test_result = test_rubiks(network, max_tries=1000)
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
    population_size = 4
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

    # Reinforcement Learning parameters
    memory = PrioritizedReplayMemory(100000)

    discount_factor = 0.99

    # Task parameters
    lamarckism = True
    baldwin = True

    # Curriculum settings
    curriculum = 'LBF'

    task = RubiksTask_Deep(batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum)
    result = population.epoch(evaluator=task, generations=50)
    genome = result['champions'][np.argmax(np.multiply(result['stats']['fitness_max'], result['stats']['info_max']))]
    network = NeuralNetwork(genome, batch_size=1, device=device, use_single_activation_function=False)
    test_result = test_rubiks(network, max_tries=1000)
    print(test_result[2])

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
    print(type(result['champions'][-1]))
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

    xortask(device, batch_size)

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


    deep_rubikstask(device, batch_size)


