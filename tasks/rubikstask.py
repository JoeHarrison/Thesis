import random
import torch
from feedforwardnetwork import NeuralNetwork
from dqnagent import DQNAgent
import rubiks2
import time

class RubiksTask(object):
    def __init__(self, batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum, use_single_activation_function=False):
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.device = device

        self.generation = 0
        self.difficulty = 0

        self.generations_in_difficulty = 0
        self.generations_in_difficulty_updated = self.generation

        self.difficulty_set = False
        self.memory = memory
        self.discount_factor = discount_factor

        self.baldwin = baldwin
        self.lamarckism = lamarckism

        self.use_single_activation_function = use_single_activation_function

        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 100
        self.epsilon_by_linear_step = lambda step_idx: epsilon_final + (epsilon_start-epsilon_final)*((epsilon_decay-step_idx)/epsilon_decay) if step_idx < epsilon_decay else epsilon_final

        # Select curriculum
        if curriculum is 'Naive':
            self.curriculum = self._naive
        elif curriculum is 'Uniform':
            self.curriculum = self._uniform
        elif curriculum is 'LBF':
            self.curriculum = self._lbf
        else:
            self.curriculum = self._no_curriculum

        self.env = rubiks2.RubiksEnv2()
        self.envs = [rubiks2.RubiksEnv2() for _ in range(self.batch_size)]

    def _naive(self):
        return self.difficulty

    def _lbf(self):
        random_number = random.random()
        if random_number > 0.2:
            return self.difficulty
        else:
            return random.randint(0, 14)

    def _uniform(self):
        return random.randint(0, 14)

    def _no_curriculum(self):
        return 14

    def backprop(self, genome):
        t = time.time()
        network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)

        agent = DQNAgent(network, self.discount_factor, self.memory, 1, self.device)

        batch_network = NeuralNetwork(genome, batch_size=100, device=self.device, use_single_activation_function=self.use_single_activation_function)

        batch_agent = DQNAgent(batch_network, self.discount_factor, self.memory, 100, self.device)

        for epoch in range(1000):
            network.reset()
            batch_network.reset()

            state = self.env.reset(self.curriculum() + 1)

            for _ in range(self.difficulty + 14):
                action = agent.act(state, self.epsilon_by_linear_step(self.generations_in_difficulty), [0.0]*6, self.device)
                next_state, reward, done, info = self.env.step(int(action))

                self.memory.push((state, action, reward, next_state, done))

                state = next_state

            # agent.train(self.epsilon_by_linear_step(self.generations_in_difficulty))
            batch_agent.train()

        genome.rl_training = False
        print('RL-time: ', time.time()-t, 'len mem:', len(self.memory))
        if self.lamarckism:
            genome.weights_to_genotype(batch_network)
            return genome
        else:
            return batch_network

    def evaluate(self, genome, generation):
        if generation > self.generation:
            self.difficulty_set = False
            self.generation = generation

        if genome.rl_training and self.baldwin:

            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)

            agent = DQNAgent(network, self.discount_factor, self.memory, 1, self.device)

            total_done = 0.0

            for i in range(100):
                network.reset()
                done = 0.0
                tries = 0
                max_tries = self.difficulty + 1
                state = self.env.reset(self.difficulty + 1)

                while tries < max_tries and not done:
                    action = agent.act(state, 0.0, [0.0]*6, self.device)
                    next_state, reward, done, info = self.env.step(int(action))

                    state = next_state
                    tries += 1
                total_done += done

            print('before:', total_done/100.0)
            w_before = network.input_to_output.linear.weight.data

            network = self.backprop(genome)
            if not isinstance(network, NeuralNetwork):
                network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)
            network.batch_size = 1
            print((w_before - network.input_to_output.linear.weight.data).sum())
            agent = DQNAgent(network, self.discount_factor, self.memory, 1, self.device)

            total_done = 0.0

            for i in range(100):
                network.reset()
                done = 0.0
                tries = 0
                max_tries = self.difficulty + 1
                state = self.env.reset(self.difficulty + 1)

                while tries < max_tries and not done:
                    action = agent.act(state, 0.0, [0.0]*6, self.device)
                    next_state, reward, done, info = self.env.step(int(action))

                    state = next_state
                    tries += 1
                total_done += done

            print('after:', total_done/100.0)
        else:
            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)

        agent = DQNAgent(network, self.discount_factor, self.memory, 1, self.device)

        total_done = 0.0

        for i in range(100):
            network.reset()
            done = 0.0
            tries = 0
            max_tries = self.difficulty + 1
            state = self.env.reset(self.difficulty + 1)

            while tries < max_tries and not done:
                action = agent.act(state, 0.0, [0.0]*6, self.device)
                next_state, reward, done, info = self.env.step(int(action))

                state = next_state
                tries += 1
            total_done += done

        percentage_solved = total_done/100.0

        if self.generations_in_difficulty_updated != self.generation:
            self.generations_in_difficulty_updated = self.generation
            self.generations_in_difficulty += 1

        if not self.difficulty_set and percentage_solved > 0.95:
            self.difficulty += 1
            self.difficulty_set = True

        if self.difficulty_set:
            return {'fitness': percentage_solved, 'info': self.difficulty - 1, 'generation': generation}
        else:
            return {'fitness': percentage_solved, 'info': self.difficulty, 'generation': generation}

    def solve(self, network):
        if self.difficulty == 14:
            return int(self.evaluate(network, self.generation)['fitness'] > 0.95)
        else:
            return 0
