import torch
from NEAT.genotype import Genotype
from naming.namegenerator import NameGenerator
from NEAT.population import Population
from tasks.xortask import XORTask
from tasks.rubikstask import RubiksTask
from tasks.cartpoletask import CartpoleTask, DQNAgent
from tasks.acrobottask import AcrobotTask
from reinforcement_learning.vanillarl import VanillaRL
from reinforcement_learning.memory import ReplayMemory
from feedforwardnetwork import NeuralNetwork
import gym


def xortask(device, batch_size):
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
    p_mutate_type = 0.2
    stdev_mutate_weight = 1.5
    stdev_mutate_bias = 0.5
    stdev_mutate_response = 0.5
    weight_range = (-3., 3.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type = 'fully_connected'
    initial_sigma = 0.0

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

    # gen = Genotype(new_individual_name, inputs, outputs)
    #
    # gen.neuron_genes = [[0, 'tanh', 0.0, 0, 0.0, 1.0], [1, 'tanh', 0.0, 0, 1024.0,1.0], [2,  'tanh', 0.0, 2048, 2048.0, 1.0], [3,  'tanh', 0.0, 1, 1024.0, 1.0], [4,  'tanh', 0.0, 2, 1536.0, 1.0], [5, 'tanh', 0.18701591191571043, 1, 512.0, 1.0], [6, 'tanh', 0.0, 1, 512.0, 1.0], [7, 'tanh', 0.0, 2, 768.0, 1.0]]
    #
    # gen.connection_genes = {(0, 2): [0, 0, 2, 3.0, False], (1, 2): [1, 1, 2, 3.0, True], (0, 3): [2, 0, 3, 1.3543900040487509, True], (3, 2): [3, 3, 2, -1.7074518345019327, True], (0, 4): [4, 0, 4, 1.6063807318468386, True], (4, 2): [5, 4, 2, 2.7212401858775306, True], (0, 5): [6, 0, 5, 1.2006299281998838, True], (5, 3): [7, 5, 3, -3.0, False], (5, 2): [8, 5, 2, -1.374675084173348, True], (5, 4): [9, 5, 4, 1.1369768644513045, True], (5, 7): [12, 5, 7, -0.18695628785579665, True], (7, 3): [13, 7, 3, 0.5542666714061728, True], (7, 4): [16, 7, 4, -3.0, True], (6, 2): [8, 6, 2, -1.4279939502396626, True]}
    #
    # net = NeuralNetwork(gen, device=device)
    # output = net(torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    # print(output)


def cartpoletask(device, batch_size):
    # Name Generators
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Genotype Parameters
    inputs = 4
    outputs = 2
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
    weight_range = (-5., 5.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type= 'partially_connected'
    initial_sigma = 0.0

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
    population_size = 32
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 8
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

    # Task parameters
    lamarckism = False

    task = CartpoleTask(batch_size, device, 0.99, memory, lamarckism)
    # task = CartpoleTask(batch_size, device, rl_method, lamarckism)
    result = population.epoch(evaluator=task, generations=1000, solution=task)
    while True:
        done = False
        envs = [gym.make('CartPole-v0')]
        state = torch.tensor([envs[0].reset()], device=device, dtype=torch.float32)
        steps = 0
        network = NeuralNetwork(result['champions'][-1], device=device)
        network.reset()
        agent = DQNAgent(network, discount_factor, memory, 1, device)
        while not done:
            action = agent.select_actions(state, 0.0)

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(envs, action)])
            done = done[0]
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

            state = next_state
            steps += 1
            envs[0].render()
        print(steps)
        envs[0].close()

def cartpoletask(device, batch_size):
    # Name Generators
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Genotype Parameters
    inputs = 4
    outputs = 2
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
    weight_range = (-5., 5.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type= 'partially_connected'
    initial_sigma = 0.0

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
    population_size = 32
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 8
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
    rl= True

    task = CartpoleTask(batch_size, device, 0.99, memory, lamarckism)
    result = population.epoch(evaluator=task, generations=10, solution=task)
    while True:
        done = False
        envs = [gym.make('CartPole-v0')]
        state = torch.tensor([envs[0].reset()], device=device, dtype=torch.float32)
        steps = 0
        network = NeuralNetwork(result['champions'][-1], device=device)
        network.reset()
        agent = DQNAgent(network, discount_factor, memory, 1, device)
        while not done:
            action = agent.select_actions(state, 0.0)

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(envs, action)])
            done = done[0]
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

            state = next_state
            steps += 1
            envs[0].render()
        print(steps)
        envs[0].close()

def acrobottask(device, batch_size):
    # Name Generators
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
    new_specie_name = surname_generator.generate_name()

    # Genotype Parameters
    inputs = 6
    outputs = 3
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
    weight_range = (-5., 5.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type = 'partially_connected'
    initial_sigma = 0.0

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
    population_size = 32
    elitism = True
    stop_when_solved = True
    tournament_selection_k = 3
    verbose = True
    max_cores = 1
    compatibility_threshold = 3.0
    compatibility_threshold_delta = 0.4
    target_species = 8
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
    lamarckism = False

    task = AcrobotTask(batch_size, device, 0.99, memory, lamarckism)
    # task = CartpoleTask(batch_size, device, rl_method, lamarckism)
    result = population.epoch(evaluator=task, generations=1000, solution=task)
    while True:
        done = False
        envs = [gym.make('Acrobot-v1')]
        state = torch.tensor([envs[0].reset()], device=device, dtype=torch.float32)
        steps = 0
        network = NeuralNetwork(result['champions'][-1], device=device)
        network.reset()
        agent = DQNAgent(network, discount_factor, memory, 1, device)
        while not done:
            action = agent.select_actions(state, 0.0)

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(envs, action)])
            done = done[0]
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

            state = next_state
            steps += 1
            envs[0].render()
        print(steps)
        envs[0].close()

def rubikstask(device, batch_size):
    # Initialise name generators for individuals in NEAT population
    first_name_generator = NameGenerator('naming/names.csv', 3, 12)
    new_individual_name = first_name_generator.generate_name()
    surname_generator = NameGenerator('naming/surnames.csv', 3, 12)
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
    initial_weight_stdev = 0.5
    p_add_neuron = 0.05
    p_add_connection = 0.3
    p_mutate_weight = 0.8
    p_reset_weight = 0.1
    p_reenable_connection = 0.01
    p_disable_connection = 0.01
    p_reenable_parent = 0.25
    p_mutate_bias = 0.2
    p_mutate_response = 0.0
    p_mutate_type = 0.2
    stdev_mutate_weight = 0.5
    stdev_mutate_bias = 0.5
    stdev_mutate_response = 0.5
    weight_range = (-3., 3.)

    distance_excess_weight = 1.0
    distance_disjoint_weight = 1.0
    distance_weight = 0.4

    initialisation_type = 'partially_connected'
    initial_sigma = 0.01

    genome_factory = lambda: Genotype(new_individual_name, inputs, outputs, nonlinearities, topology, feedforward,
                                  max_depth, max_nodes, response_default, initial_weight_stdev,
                                  bias_as_node, p_add_neuron, p_add_connection, p_mutate_weight,
                                  p_reset_weight, p_reenable_connection, p_disable_connection,
                                  p_reenable_parent, p_mutate_bias, p_mutate_response, p_mutate_type,
                                  stdev_mutate_weight, stdev_mutate_bias, stdev_mutate_response,
                                  weight_range, distance_excess_weight, distance_disjoint_weight,
                                  distance_weight, initialisation_type, initial_sigma)

    # Population parameters
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

    population = Population(new_specie_name, genome_factory, population_size, elitism, stop_when_solved,
                            tournament_selection_k, verbose, max_cores, compatibility_threshold,
                            compatibility_threshold_delta, target_species, minimum_elitism_size,
                            young_age, young_multiplier, old_age, old_multiplier, stagnation_age, reset_innovations,
                            survival)

    # Reinforcement Learning parameters
    memory = ReplayMemory(1000)
    # memory = None
    discount_factor = 0.99

    rl_method = VanillaRL(memory, discount_factor, device, batch_size)

    # Task parameters
    lamarckism = True
    rl = True

    task = RubiksTask(batch_size, device, 0.99, memory, lamarckism, rl)
    result = population.epoch(evaluator=task, generations=1000, solution=task)


if __name__ == "__main__":
    # Set seeds to get reproducible outcomes by uncommenting the following

    # np.random.seed(3)
    # torch.manual_seed(3)
    # random.seed(3)

    # Checks whether CUDA is available. If it is the program will run on the GPU, otherwise on the CPU.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    print('Using %s' % device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # Batch size of training and testing
    batch_size = 128

    # Variety of Problems. Uncomment to test
    # For testing Reinforcement Learning
    # cartpoletask(device, batch_size)
    # acrobottask(device, batch_size)
    # For testing NEAT
    # Batch_size for the xortask is set to 4, because the batch size also determines the batch_size for evaluation.

    # xortask(device, 4)

    rubikstask(device, batch_size)
