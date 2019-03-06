from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class PrioritisedReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.losses = []
        self.total_loss = 0
        self.position = 0

    def push(self, *args, loss):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.losses.append(None)
        else:
            self.total_loss -= self.losses[self.position]

        self.memory[self.position] = Transition(*args)
        self.losses[self.position] = loss
        self.total_loss += loss

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return np.random.choice(self.memory, self.losses/self.total_loss, batch_size)

    def __len__(self):
        return len(self.memory)
