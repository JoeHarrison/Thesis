import rubiks
from feedforwardnetwork import NeuralNetwork
import numpy as np
import torch
import time

class RubiksTask(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.difficulty = 1

        self.envs = [rubiks.RubiksEnv(2) for _ in range(self.batch_size)]

        # self.env = rubiks.RubiksEnv(2)


    def _increase_difficulty(self):
        self.difficulty += 1

    def evaluate(self, network, verbose=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)

        fitness = 0.000001

        # batch here?
        # instead of range 100 do batch in network
        # RL loop here?

        # time1 = time.time()
        max_tries = self.difficulty
        tries = 0
        fitness = torch.zeros(self.batch_size, 1, dtype=torch.float32)
        state = torch.tensor([self.envs[i].reset(self.difficulty) for i in range(self.batch_size)])
        network.reset()

        while tries < max_tries:
            action_probabilities = network(state)
            actions = torch.max(action_probabilities, 1)[1]

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, actions)])
            done = torch.tensor(done, dtype=torch.float32).view(-1, 1)
            next_state = torch.tensor(next_state)
            fitness += done
            state = next_state
            tries += 1

        fitness = float((fitness > 0).sum().item()) / self.batch_size
        # print('1',fitness, time.time() - time1)
        #
        # time2 = time.time()
        # for i in range(128):
        #     done = 0
        #     tries = 0
        #
        #     max_tries = self.difficulty
        #     state = self.env.reset(self.difficulty)
        #     network.reset()
        #
        #
        #
        #
        #     while tries < max_tries and not done:
        #
        #         action_probabilities = network(np.array([state]))
        #         action = torch.max(action_probabilities, 1)[1]
        #
        #         next_state, reward, done, info = self.env.step(int(action))
        #
        #         tries += 1
        #         state = next_state
        #     if done:
        #         fitness += 1.0
        #
        # fitness = fitness / 100
        # # print('2',fitness, time.time() - time2)
        # #
        if fitness > 0.75:
            self._increase_difficulty()

        return {'fitness' : fitness, 'info' : self.difficulty}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.99)
