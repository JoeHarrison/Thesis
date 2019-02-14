import rubiks
from feedforwardnetwork import NeuralNetwork
import numpy as np
import torch

class RubiksTask(object):
    def __init__(self):
        self.difficulty = 1
        self.env = rubiks.RubiksEnv(2)

    def _increase_difficulty(self):
        self.difficulty += 1

    def evaluate(self, network, verbose=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)

        fitness = 0.000001

        #batch here?
        #instead of range 100 do batch in network
        #RL loop here?
        for i in range(100):
            done = False
            tries = 0

            max_tries = self.difficulty
            state = self.env.reset(self.difficulty)

            while tries < max_tries and not done:

                action_probabilities = network(np.array([state]))
                action = torch.max(action_probabilities, 1)[1]

                next_state, reward, done, info = self.env.step(int(action))

                tries += 1
                state = next_state
            if done:
                fitness += 1.0

        fitness = fitness / 100

        if fitness > 0.75:
            self._increase_difficulty()

        return {'fitness' : fitness, 'info' : self.difficulty}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.99)
