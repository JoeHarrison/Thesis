import rubiks
from feedforwardnetwork import NeuralNetwork
from dqnagent import DQNAgent
import torch
import random


class RubiksTask(object):
    def __init__(self, batch_size, device, discount_factor, memory, curriculum, lamarckism=False, rl=False):
        self.batch_size = batch_size
        self.device = device
        self.lamarckism = lamarckism
        self.rl = rl
        self.discount_factor = discount_factor
        self.memory = memory
        self.difficulty = 0
        self.generation = 0
        self.difficulty_set = False

        if curriculum is 'naive':
            self.curriculum = self._naive
        elif curriculum is 'mixed':
            self.curriculum = self._mixed
        else:
            self.curriculum = self._combination

        # TODO set cube size from parameter
        self.envs = [rubiks.RubiksEnv(2) for _ in range(self.batch_size)]

    def _increase_difficulty(self):
        self.difficulty += 1

    def _combination(self, p=0.2, max_difficulty=12*20):
        random_number = random.random()
        if random_number > p:
            return self.difficulty
        else:
            return random.randint(0, 12*(self.difficulty//12 + 1))

    def _mixed(self, max_difficulty=12*20):
        return random.randint(0, max_difficulty)

    def _naive(self):
        return self.difficulty

    def rl_training(self, genome):
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)

        agent = DQNAgent(network, self.discount_factor, self.memory, self.batch_size, self.device)

        max_tries = 20

        for epoch in range(100):
            import time
            t = time.time()
            # TODO: network reset problem
            network.reset()
            state = torch.tensor([self.envs[i].curriculum_reset(level=self.curriculum()) for i in range(self.batch_size)], device=self.device, dtype=torch.float32)
            for i in range(max_tries):
                # TODO: not enough exploration with 0.1
                action = agent.select_actions(state, 0.5)

                next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, action)])
                done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
                next_state = torch.tensor(next_state, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

                agent.memory.push(state, action, next_state, reward, done)

                agent.train()

                # Reset each state that is done
                state = torch.tensor([env.curriculum_reset(level=self.curriculum()) if d else s.tolist() for env, s, d in zip(self.envs, next_state, done)], dtype=torch.float32, device=self.device)

        # Moves trained weights to back to genes
        if self.lamarckism:
            genome.weights_to_genotype(network)

        genome.rl_training = False

    def evaluate(self, genome, generation):
        if genome.rl_training:
            print('rl training')
            self.rl_training(genome)

        if generation > self.generation:
            self.difficulty_set = False
            self.generation = generation

        # Should always be a genome
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, batch_size=self.batch_size, device=self.device)

        # TODO: Move agent class code to separate class
        agent = DQNAgent(network, self.discount_factor, self.memory, self.batch_size, self.device)

        network = NeuralNetwork(genome, batch_size=1, device=self.device)

        agent = DQNAgent(network, self.discount_factor, self.memory, 1, self.device)

        network.reset()

        total_done = torch.zeros([self.batch_size, 1], device=self.device)
        state = torch.tensor([self.envs[i].curriculum_reset(level=self.curriculum()) for i in range(self.batch_size)], device=self.device, dtype=torch.float32)
        max_tries = 20
        for _ in range(max_tries):
            action = agent.select_actions(state, 0.0)
            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, action)])

            done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            state = next_state

            total_done += done

        fitness = ((total_done >= 1).sum().item())/float(len(state))

        # TODO: set threshold in init
        if not self.difficulty_set and fitness > self.difficulty / (self.difficulty + 1) + 0.5 * 1/(self.difficulty + 1):
            self._increase_difficulty()
            self.difficulty_set = True

        return {'fitness': fitness, 'info': self.difficulty}

    def solve(self, network):
        return int(self.evaluate(network, self.generation)['fitness'] > 0.99)
