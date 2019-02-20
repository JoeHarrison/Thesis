import torch
import torch.nn.functional as F

class VanillaRL:
    def __init__(self, memory, discount_factor, device):
        self.memory = memory
        self.discount_factor = discount_factor
        self.device = device

    @staticmethod
    def compute_q_val(network, state, action):
        qactions = network(state)

        return torch.gather(qactions, 1, action.view(-1, 1))

    def compute_target(self, network, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
        m = torch.cat(((self.discount_factor*torch.max(network(next_state), 1)[0]).view(-1, 1), torch.zeros(reward.size(), device=self.device).view(-1, 1)), 1)

        return reward.view(-1, 1) + torch.gather(m, 1, done.long().view(-1, 1))

    def train(self, network, optimiser, state, action, reward, next_state, done):
        q_val = self.compute_q_val(network, state, action)

        with torch.no_grad():
            target = self.compute_target(network, reward, next_state, done)

        loss = F.smooth_l1_loss(q_val, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        return loss.item()

    def step(self, network, optimiser, state, action, next_state, reward, done):
        loss = self.train(network, optimiser, state, action, reward, next_state, done)

        return loss
