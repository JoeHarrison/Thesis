import rubiks
from feedforwardnetwork import NeuralNetwork
import torch
from torch import optim


class RubiksTask(object):
    def __init__(self, batch_size, device, rl_method, lamarckism=False):
        self.batch_size = batch_size
        self.device = device
        self.rl_method = rl_method
        self.lamarckism = lamarckism

        self.difficulty = 1

        # TODO set cube size from parameter
        self.envs = [rubiks.RubiksEnv(2) for _ in range(self.batch_size)]

    def _increase_difficulty(self):
        self.difficulty += 1

    def _decrease_difficulty(self):
        if self.difficulty > 1:
            self.difficulty -= 1

    def select_actions(self, model, state, epsilon):
        with torch.no_grad():
            action_probabilities = model(state)

            max_action = torch.max(action_probabilities, 1)[1]
            rand_action = torch.randint(0, action_probabilities.size(1), size=([action_probabilities.size(0)]), device=self.device, dtype=torch.long)
            bernoulli = torch.bernoulli(torch.ones(action_probabilities.size(0), device=self.device)*epsilon).long().view(-1, 1)
            return torch.gather(torch.cat((max_action.view(-1, 1), rand_action.view(-1, 1)), 1), 1, bernoulli).view(-1)

    def evaluate(self, genome):
        # Should always be a genome
        if not isinstance(genome, NeuralNetwork):
            network = NeuralNetwork(genome, device=self.device)

        optimiser = optim.Adam(network.parameters(), amsgrad=True)

        max_tries = self.difficulty + 10
        tries = 0
        fitness = torch.zeros(self.batch_size, 1, dtype=torch.float32, device=self.device)
        state = torch.tensor([self.envs[i].reset(self.difficulty) for i in range(self.batch_size)], device=self.device, dtype=torch.float32)
        network.reset()

        while tries < max_tries:
            # action_probabilities = network(state)
            # actions = torch.max(action_probabilities, 1)[1]
            actions = self.select_actions(network, state, 0.01)

            next_state, reward, done, info = zip(*[env.step(int(a)) for env, a in zip(self.envs, actions)])
            done = torch.tensor(done, dtype=torch.float32, device=self.device).view(-1, 1)
            next_state = torch.tensor(next_state, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

            # print('Before', network.output_biases)
            losses = self.rl_method.step(network, optimiser, state, actions, next_state, reward, done)
            # print('After', network.output_biases)

            # Reset each state that is done

            next_state = torch.tensor([env.reset(self.difficulty) if d else s.tolist() for env, s, d in zip(self.envs, next_state, done)], dtype=torch.float32, device=self.device)

            state = next_state
            fitness += done
            tries += 1

        # Moves trained weights to genes
        if self.lamarckism:
            genome.weights_to_genotype(network)

        fitness = float((fitness > 0).sum().item()) / self.batch_size

        # TODO: set threshold in init
        if fitness > 0.8:
            self._increase_difficulty()

        return {'fitness': fitness, 'info': self.difficulty}

    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.99)
