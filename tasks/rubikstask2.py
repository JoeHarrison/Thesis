import random
import torch
import torch.nn as nn
from feedforwardnetwork import NeuralNetwork
from dqnagent import DQNAgent
import rubiks2
import time
import copy

class RubiksTask(object):
    def __init__(self, batch_size, device, baldwin, lamarckism, discount_factor, memory, curriculum, use_single_activation_function=False):
        self.criterion = torch.nn.MSELoss()
        self.batch_size = batch_size
        self.device = device

        self.generation = 0
        self.difficulty = 0

        self.set_difficulty_next_gen = False
        self.memory = memory
        self.discount_factor = discount_factor

        self.baldwin = baldwin
        self.lamarckism = lamarckism

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

    def compute_q_val(self, model, state, action):
        qactions = model(state)

        return torch.gather(qactions, 1, action.view(-1, 1))

    def compute_target(self, model, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
        m = torch.cat(((self.discount_factor*torch.max(model(next_state), 1)[0]).view(-1, 1), torch.zeros(reward.size(), device=self.device).view(-1, 1)), 1)

        return reward.view(-1, 1) + torch.gather(m, 1, done.long().view(-1, 1))

    def evaluate(self, genome, generation):

        if generation > self.generation:
            if self.set_difficulty_next_gen:
                self.difficulty += 1
                self.set_difficulty_next_gen = False
            self.generation = generation

        network = NeuralNetwork(genome, batch_size=1, device=self.device, use_single_activation_function=self.use_single_activation_function)

        optimiser = torch.optim.Adam(network.parameters())

        total_done = 0.0

        for i in range(100):
            network.reset()
            done = 0.0
            tries = 0
            max_tries = self.difficulty//6 + 1
            state = torch.tensor([self.env.curriculum_reset(self.difficulty)], dtype=torch.float32, device=self.device)

            while tries < max_tries and not done:
                q_values = network(state)
                action = q_values.max(1)[1].view(1, 1).item()
                next_state, reward, done, info = self.env.step(int(action))
                next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)

                if genome.rl_training and self.baldwin:
                    w_before = copy.copy(network.input_to_output.linear.weight.data)
                    optimiser.zero_grad()
                    # state = torch.tensor(state, dtype=torch.float32, device=self.device)
                    action = torch.tensor([action], dtype=torch.long, device=self.device)
                    # next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
                    reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
                    done = torch.tensor([done], dtype=torch.float32, device=self.device)

                    q_val = self.compute_q_val(network, state, action)

                    with torch.no_grad():
                        target = self.compute_target(network, reward, next_state, done)
                    criterion = nn.MSELoss()

                    loss = criterion(q_val, target)

                    loss.backward()

                    optimiser.step()

                state = next_state
                tries += 1
            if genome.rl_training and self.baldwin:
                total_done += done.item()
            else:
                total_done += done

        percentage_solved = total_done/100.0

        if percentage_solved > 0.95:
            self.set_difficulty_next_gen = True

        return {'fitness': percentage_solved, 'info': self.difficulty, 'generation': generation, 'reset_species': self.set_difficulty_next_gen}

    def solve(self, network):
        if self.difficulty == 14:
            return int(self.evaluate(network, self.generation)['fitness'] > 0.95)
        else:
            return 0
