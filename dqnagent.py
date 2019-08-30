import torch
import torch.nn as nn
from torch import optim
import numpy as np

class DQNAgent:
    def __init__(self, model, discount_factor, memory, batch_size, device):
        self.model = model.to(device)
        self.memory = memory
        self.optimiser = optim.Adam(self.model.parameters(), amsgrad=True)
        self.batch_size = batch_size
        self.device = device
        self.discount_factor = discount_factor

    def select_actions(self, state, epsilon):
        with torch.no_grad():
            action_probabilities = self.model(state)

            max_action = torch.max(action_probabilities, 1)[1]
            rand_action = torch.randint(0, action_probabilities.size(1), size=([action_probabilities.size(0)]), device=self.device, dtype=torch.long)
            bernoulli = torch.bernoulli(torch.ones(action_probabilities.size(0), device=self.device)*epsilon).long().view(-1, 1)
            return torch.gather(torch.cat((max_action.view(-1, 1), rand_action.view(-1, 1)), 1), 1, bernoulli).view(-1)

    def compute_q_val(self, state, action):
        qactions = self.model(state)

        return torch.gather(qactions, 1, action.view(-1, 1))

    def compute_target(self, reward, next_state, done):
        # done is a boolean (vector) that indicates if next_state is terminal (episode is done)
        m = torch.cat(((self.discount_factor*torch.max(self.model(next_state), 1)[0]).view(-1, 1), torch.zeros(reward.size(), device=self.device).view(-1, 1)), 1)

        return reward.view(-1, 1) + torch.gather(m, 1, done.long().view(-1, 1))

    def act(self, state, epsilon, mask, device):
        if np.random.rand() > epsilon:
            state = torch.tensor([state], dtype=torch.float32, device=device)
            mask = torch.tensor([mask], dtype=torch.float32, device=device)
            q_values = self.model(state) + mask
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = np.random.randint(6)
        return action

    def train(self):
        self.optimiser.zero_grad()

        batch, indices, weights = self.memory.sample(self.batch_size, self.batch_size, self.device)

        state, action, reward, next_state, done = zip(*batch)

        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action = torch.tensor([action], dtype=torch.long, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        done = torch.tensor([done], dtype=torch.float32, device=self.device)

        q_val = self.compute_q_val(state, action)

        target = self.compute_target(reward, next_state, done)

        criterion = nn.MSELoss()

        loss = criterion(q_val, target.detach())

        loss.backward()

        self.optimiser.step()

        return loss.item()

# import torch
# from torch import optim
# import numpy as np
#
# class DQNAgent:
#     def __init__(self, model, discount_factor, memory, batch_size, device):
#         self.model = model.to(device)
#         self.memory = memory
#         self.optimiser = optim.Adam(self.model.parameters())
#         self.batch_size = batch_size
#         self.device = device
#         self.discount_factor = discount_factor
#
#     # Computes the q-values of an action in a state
#     def compute_q_val(self, model, state, action):
#         qactions = model(state)
#         return torch.gather(qactions, 1, action.view(-1, 1))
#
#     # Computes the target. When done, 0 is added to the reward as there is no next state.
#     def compute_target_dqn(self, model, reward, next_state, done, gamma):
#         return reward + gamma * model(next_state).max(1)[0] * (1-done)
#
#     # Computes the target. When done, 0 is added to the reward as there is no next state. But now for Double DQN
#     def compute_target_ddqn(self, model, target_model, reward, next_state, done, gamma):
#         a = model(next_state)
#         return reward.view(-1, 1) + gamma * torch.gather(target_model(next_state), 1, model(next_state).max(1)[1].view(-1, 1)) * (1-done).view(-1, 1)
#
#     def train(self, local_steps):
#         if len(self.memory) < self.batch_size:
#             return None
#
#         batch, indices, weights = self.memory.sample(self.batch_size, local_steps, self.device)
#
#         state, action, reward, next_state, done = zip(*batch)
#
#         state = torch.tensor(state, dtype=torch.float32, device=self.device)
#         action = torch.tensor([action], dtype=torch.long, device=self.device)
#         next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
#         reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
#         done = torch.tensor([done], dtype=torch.float32, device=self.device)
#
#         weights.to(self.device)
#
#         self.optimiser.zero_grad()
#
#         q_val = self.compute_q_val(self.model, state, action)
#
#         target = self.compute_target_dqn(self.model, reward, next_state, done, self.discount_factor)
#
#         difference = (q_val - target.view(-1, 1))
#
#         loss = difference.pow(2)
#         loss = loss.mean()
#         loss.backward()
#         self.memory.update_priorities(indices, difference.detach().view(-1).abs().cpu().numpy().tolist())
#
#         self.optimiser.step()
#
#         return loss.item()
#
#     def act(self, state, epsilon, mask, device):
#         if np.random.rand() > epsilon:
#             state = torch.tensor([state], dtype=torch.float32, device=device)
#             mask = torch.tensor([mask], dtype=torch.float32, device=device)
#             q_values = self.model(state) + mask
#             action = q_values.max(1)[1].view(1, 1).item()
#         else:
#             action = np.random.randint(6)
#         return action
#
#     def act_multi(self, state, epsilon, device):
#         with torch.no_grad():
#             action_probabilities = self.model(state)
#
#             max_action = torch.max(action_probabilities, 1)[1]
#             rand_action = torch.randint(0, action_probabilities.size(1), size=([action_probabilities.size(0)]), device=self.device, dtype=torch.long)
#             bernoulli = torch.bernoulli(torch.ones(action_probabilities.size(0), device=self.device)*epsilon).long().view(-1, 1)
#             return torch.gather(torch.cat((max_action.view(-1, 1), rand_action.view(-1, 1)), 1), 1, bernoulli).view(-1)
