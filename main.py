import torch
from genotype import Genotype
from namegenerator import NameGenerator
from population import Population
from xortask import XORTask
from rubikstask import RubiksTask
from vanillarl import VanillaRL
from replaymemory import ReplayMemory

def rubikstask(device, batch_size):
    # Name Generators
    first_name_generator = NameGenerator('names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Genotype Parameters
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
    weight_range = (-50., 50.)

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

    # Population parameters
    population_size = 32
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

    # RL parameters
    memory = ReplayMemory(100000)
    # memory = None
    discount_factor = 0.99

    rl_method = VanillaRL(memory, discount_factor, device, batch_size)

    # Task parameters
    lamarckism = True

    task = RubiksTask(batch_size, device, rl_method, lamarckism)
    result = population.epoch(evaluator=task, generations=1000, solution=task)

# def xortask():
#     first_name_generator = NameGenerator('names.csv', 3, 12)
#     new_individual_name = first_name_generator.generate_name()
#     surname_generator = NameGenerator('surnames.csv', 3, 12)
#     new_specie_name = surname_generator.generate_name()
#
#     inputs = 2
#     outputs = 1
#     nonlinearities = ['tanh', 'relu', 'sigmoid', 'identity']
#     topology = None
#     feedforward = True
#     max_depth = None
#     max_nodes = float('inf')
#     response_default = 1.0
#     bias_as_node = False
#     initial_weight_stdev = 2.0
#     p_add_neuron = 0.03
#     p_add_connection = 0.3
#     p_mutate_weight = 0.8
#     p_reset_weight = 0.1
#     p_reenable_connection = 0.01
#     p_disable_connection = 0.01
#     p_reenable_parent=0.25
#     p_mutate_bias = 0.2
#     p_mutate_response = 0.0
#     p_mutate_type = 0.2
#     stdev_mutate_weight = 1.5
#     stdev_mutate_bias = 0.5
#     stdev_mutate_response = 0.5
#     weight_range = (-1., 1.)
#
#     distance_excess_weight = 1.0
#     distance_disjoint_weight = 1.0
#     distance_weight = 0.4
#
#     genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
#                                   max_depth, max_nodes, response_default, initial_weight_stdev,
#                                   bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
#                                   p_reset_weight, p_reenable_connection, p_disable_connection,
#                                   p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
#                                   stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
#                                   weight_range, distance_excess_weight, distance_disjoint_weight,
#                                   distance_weight)
#
#     population_size = 512
#     elitism = True
#     stop_when_solved = True
#     tournament_selection_k = 3
#     verbose = True
#     max_cores = 1
#     compatibility_threshold = 3.0
#     compatibility_threshold_delta = 0.4
#     target_species = 16
#     minimum_elitism_size = 5
#     young_age = 10
#     young_multiplier = 1.2
#     old_age = 30
#     old_multiplier = 0.2
#     stagnation_age = 15
#     reset_innovations = False
#     survival = 0.2
#
#     population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved, tournament_selection_k, verbose, max_cores, compatibility_threshold, compatibility_threshold_delta, target_species, minimum_elitism_size, young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations, survival)
#     task = XORTask()
#     result = population.epoch(evaluator = task, generations = 1000, solution = task)
#     print(result['champions'][-1].neuron_genes)
#     print(result['champions'][-1].connection_genes)

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

    batch_size = 128

    rubikstask(device, batch_size)
