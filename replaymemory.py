from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # def push(self, *args):
    #     if len(self.memory) < self.capacity:
    #         self.memory.append(None)
    #
    #     self.memory[self.position] = Transition(*args)
    #
    #     self.position = (self.position + 1) % self.capacity

    def push(self, *args):
        batch_size = args[0].size(0)

        states, actions, next_states, rewards, dones = args

        for i in range(batch_size):
            if len(self.memory) < self.capacity:
                self.memory.append(None)

            self.memory[self.position] = Transition(states[i], actions[i], next_states[i], rewards[i], dones[i])

            self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
