import random
import torch
import torch.nn as nn
from feedforwardnetwork import NeuralNetwork
from reinforcement_learning.replay_memories import ReplayMemory
import rubiks2
import time
import copy
import numpy as np

class RubiksTask(object):
    def __init__(self, batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum, use_single_activation_function=False):
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.device = device

        self.generation = 0
        self.difficulty = 1

        self.set_difficulty_next_gen = False
        self.memory = memory
        self.discount_factor = discount_factor

        self.baldwin = baldwin
        self.lamarckism = lamarckism
        self.memory = ReplayMemory(1400000)

        self.use_single_activation_function = use_single_activation_function

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

    def _naive(self):
        return self.difficulty

    def _lbf(self):
        random_number = random.random()
        if random_number > 0.2:
            return self.difficulty
        else:
            return random.randint(1, 15)

    def _uniform(self):
        return random.randint(1, 15)

    def _no_curriculum(self):
        return 14

    def compute_q_val(self, model, state, action):
        qactions = model(state)

        return torch.gather(qactions, 1, action.view(-1, 1))

    def compute_target(self, model, reward, next_state, done):
        return reward + self.discount_factor * model(next_state).max(1)[0] * (1-done)

    def compute_target2(self, model, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
        m = torch.cat(((self.discount_factor*torch.max(model(next_state), 1)[0]).view(-1, 1), torch.zeros(reward.size(), device=self.device).view(-1, 1)), 1)

        return reward.view(-1, 1) + torch.gather(m, 1, done.long().view(-1, 1))

    def backprop(self, network):
        for i in range(1000):
            done = 0
            tries = 0
            local_diff = self.curriculum()
            max_tries = local_diff
            state = self.env.reset(local_diff)

            while tries < max_tries and not done:
                network.reset()
                q_values = network(torch.tensor(state, dtype=torch.float32, device=self.device))

                if random.random() > (1000-i)/1000:
                    action = q_values.max(1)[1].view(1, 1).item()
                else:
                    action = random.randint(0, 5)

                next_state, reward, done, info = self.env.step(int(action))
                self.memory.push((state, action, next_state, reward, done))

                state = next_state
                tries += 1

            if len(self.memory) < 128:
                continue

            optimiser = torch.optim.Adam(network.parameters())

            optimiser.zero_grad()

            batch, _, _ = self.memory.sample(1, 1, self.device)
            state, action, next_state, reward, done = zip(*batch)

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            action = torch.tensor([action], dtype=torch.long, device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            done = torch.tensor([done], dtype=torch.float32, device=self.device)

            network.reset()
            q_val = self.compute_q_val(network, state, action)

            network.reset()
            with torch.no_grad():
                target = self.compute_target(network, reward, next_state, done).view(-1, 1)

            criterion = nn.MSELoss()

            loss_all = criterion(q_val, target)
            loss = loss_all.mean()

            loss.backward()

            optimiser.step()

        return network

    def evaluate(self, genome, generation):
        if generation > self.generation:
            if self.set_difficulty_next_gen:
                self.difficulty += 1
                self.set_difficulty_next_gen = False
            self.generation = generation

        if genome.rl_training and self.baldwin:
            t = time.time()
            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)
            total_done = 0.0
            for i in range(100):
                done = 0.0
                tries = 0
                max_tries = 1
                self.env.seed(i)
                state = self.env.reset(1)

                while tries < max_tries and not done:
                    network.reset()
                    q_values = network(torch.tensor(state, dtype=torch.float32, device=self.device))
                    action = q_values.max(1)[1].view(1, 1).item()
                    next_state, reward, done, info = self.env.step(int(action))
                    self.memory.push((state, action, next_state, reward, done))
                    state = next_state
                    tries += 1

                total_done += done
            print('before: ', total_done)
            print(genome.connection_genes)

            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)
            network = self.backprop(network)
            if self.lamarckism:
                genome.weights_to_genotype(network)
            genome.rl_training = False

            network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)

            total_done = 0.0
            for i in range(100):
                done = 0.0
                tries = 0
                max_tries = 1
                self.env.seed(i)
                state = self.env.reset(1)

                while tries < max_tries and not done:
                    network.reset()
                    q_values = network(torch.tensor(state, dtype=torch.float32, device=self.device))
                    action = q_values.max(1)[1].view(1, 1).item()
                    next_state, reward, done, info = self.env.step(int(action))
                    self.memory.push((state, action, next_state, reward, done))
                    state = next_state
                    tries += 1

                total_done += done
            print('After: ', total_done)
            print(genome.connection_genes)
            print(time.time()-t, genome.name, str(genome.specie))

        network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)
        total_done = 0.0
        for i in range(100):
            done = 0.0
            tries = 0
            max_tries = self.difficulty
            state = self.env.reset(self.difficulty)

            while tries < max_tries and not done:
                network.reset()
                q_values = network(torch.tensor(state, dtype=torch.float32, device=self.device))
                action = q_values.max(1)[1].view(1, 1).item()
                next_state, reward, done, info = self.env.step(int(action))
                self.memory.push((state, action, next_state, reward, done))
                state = next_state
                tries += 1

            total_done += done

        percentage_solved = total_done/100.0

        if percentage_solved >= 0.95:
            torch.save(genome, 'models/genome_' + genome.name + str(self.difficulty))
            torch.save(network, 'models/network_' + genome.name + str(self.difficulty))
            self.set_difficulty_next_gen = True

        return {'fitness': percentage_solved, 'info': self.difficulty, 'generation': generation, 'reset_species': self.set_difficulty_next_gen}

    def solve(self, network):
        if self.difficulty == 14:
            return int(self.evaluate(network, self.generation)['fitness'] > 0.95)
        else:
            return 0
