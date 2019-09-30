import torch
import torch.nn as nn
import numpy as np


# Classic DQN. Increase_capacity method adds new nodes to layers according to increment
# TODO: decrease capacity does not work as of yet

class DQN(nn.Module):
    def __init__(self, num_inputs, hidden, num_actions, non_linearity):
        super(DQN, self).__init__()

        self.num_inputs = num_inputs
        self.hidden = hidden
        self.num_actions = num_actions
        self.non_linearity = non_linearity

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(num_inputs, self.hidden[0]))

        previous = self.hidden[0]
        for hidden_layer_size in self.hidden[1:]:
            self.layers.append(nn.Linear(previous, hidden_layer_size))
            previous = hidden_layer_size

        self.layers.append(nn.Linear(previous, num_actions))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.non_linearity(self.layers[i](x))
        return self.layers[-1](x)

    def increase_capacity(self, increment):
        for i in range(len(self.hidden)):
            self.hidden[i] += increment[i]

        bias = self.layers[0].bias.data
        weight = self.layers[0].weight.data
        self.layers[0] = nn.Linear(self.num_inputs, self.hidden[0])
        if increment[0] > 0:
            self.layers[0].weight.data[0:-increment[0], :] = weight
            self.layers[0].bias.data[0:-increment[0]] = bias
        else:
            self.layers[0].weight.data[0:, :] = weight
            self.layers[0].weight.data = bias

        for i in range(1, len(self.layers) - 1):
            bias = self.layers[i].bias.data
            weight = self.layers[i].weight.data
            self.layers[i] = nn.Linear(self.hidden[i - 1], self.hidden[i])
            if increment[i] > 0:
                if increment[i - 1] > 0:
                    self.layers[i].bias.data[0:-increment[i]] = bias
                    self.layers[i].weight.data[0:-increment[i], 0:-increment[i - 1]] = weight
                else:
                    self.layers[i].bias.data[0:-increment[i]] = bias
                    self.layers[i].weight.data[0:-increment[i], 0:] = weight
            else:
                if increment[i - 1] > 0:
                    self.layers[i].bias.data = bias
                    self.layers[i].weight.data[0:, 0:-increment[i - 1]] = weight
                else:
                    self.layers[i].bias.data = bias
                    self.layers[i].weight.data[0:, 0:] = weight

        bias = self.layers[-1].bias.data
        weight = self.layers[-1].weight.data
        self.layers[-1] = nn.Linear(self.hidden[-1], self.num_actions)
        if increment[-1] > 0:
            self.layers[-1].bias.data = bias
            self.layers[-1].weight.data[:, 0:-increment[-1]] = weight
        else:
            self.layers[-1].bias.data = bias
            self.layers[-1].weight.data[:, 0:] = weight

    def act(self, state, epsilon, mask, device):
        if np.random.rand() > epsilon:
            state = torch.tensor([state], dtype=torch.float32, device=device)
            mask = torch.tensor([mask], dtype=torch.float32, device=device)
            q_values = self.forward(state) + mask
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = np.random.randint(self.num_actions)
        return action


class DuelingDQN(nn.Module):
    def __init__(self, num_inputs, hidden, num_actions, non_linearity):
        super(DuelingDQN, self).__init__()

        self.num_inputs = num_inputs
        self.hidden = hidden
        self.num_actions = num_actions

        self.non_linearity = non_linearity

        self.sharedLayers = nn.ModuleList()
        self.sharedLayers.append(nn.Linear(num_inputs, self.hidden[0]))

        previous = self.hidden[0]
        for hidden_layer_size in self.hidden[1:-1]:
            self.sharedLayers.append(nn.Linear(previous, hidden_layer_size))
            previous = hidden_layer_size

        self.adv1 = nn.Linear(previous, self.hidden[-1])
        self.adv2 = nn.Linear(self.hidden[-1], num_actions)

        self.v1 = nn.Linear(previous, self.hidden[-1])
        self.v2 = nn.Linear(self.hidden[-1], 1)

    def forward(self, x):
        for i in range(len(self.sharedLayers)):
            x = self.non_linearity(self.sharedLayers[i](x))

        a = self.non_linearity(self.adv1(x))
        a = self.adv2(a)

        v = self.non_linearity(self.v1(x))
        v = self.v2(v)

        return v + a - a.mean()

    def policy(self, x):
        for i in range(len(self.sharedLayers)):
            x = self.non_linearity(self.sharedLayers[i](x))

        a = self.non_linearity(self.adv1(x))
        a = self.adv2(a)

        return a

    def value(self, x):
        for i in range(len(self.sharedLayers)):
            x = self.non_linearity(self.sharedLayers[i](x))

        v = self.non_linearity(self.v1(x))
        v = self.v2(v)

        return v

    def increase_capacity(self, increment):
        for i in range(len(self.hidden)):
            self.hidden[i] += increment[i]

        # Check whether the increment isn't zero
        if increment[0] > 0:
            bias = self.sharedLayers[0].bias.data
            weight = self.sharedLayers[0].weight.data
            self.sharedLayers[0] = nn.Linear(self.num_inputs, self.hidden[0])
            self.sharedLayers[0].bias.data[0:-increment[0]] = bias
            self.sharedLayers[0].weight.data[0:-increment[0], :] = weight

        for i in range(1, len(self.sharedLayers)):
            bias = self.sharedLayers[i].bias.data
            weight = self.sharedLayers[i].weight.data
            self.sharedLayers[i] = nn.Linear(self.hidden[i - 1], self.hidden[i])
            if increment[i] > 0:
                if increment[i - 1] > 0:
                    self.sharedLayers[i].bias.data[0:-increment[i]] = bias
                    self.sharedLayers[i].weight.data[0:-increment[i], 0:-increment[i - 1]] = weight
                else:
                    self.sharedLayers[i].bias.data[0:-increment[i]] = bias
                    self.sharedLayers[i].weight.data[0:-increment[i], 0:] = weight
            else:
                if increment[i - 1] > 0:
                    self.sharedLayers[i].bias.data = bias
                    self.sharedLayers[i].weight.data[0:, 0:-increment[i - 1]] = weight
                else:
                    self.sharedLayers[i].bias.data = bias
                    self.sharedLayers[i].weight.data[0:, 0:] = weight

        bias_adv1 = self.adv1.bias.data
        weight_adv1 = self.adv1.weight.data
        self.adv1 = nn.Linear(self.hidden[-2], self.hidden[-1])

        bias_v1 = self.v1.bias.data
        weight_v1 = self.v1.weight.data
        self.v1 = nn.Linear(self.hidden[-2], self.hidden[-1])
        if increment[-1] > 0:
            if increment[-2] > 0:
                self.adv1.bias.data[0:-increment[-1]] = bias_adv1
                self.adv1.weight.data[0:-increment[-1], 0:-increment[-2]] = weight_adv1
                self.v1.bias.data[0:-increment[-1]] = bias_v1
                self.v1.weight.data[0:-increment[-1], 0:-increment[-2]] = weight_v1
            else:
                self.adv1.bias.data[0:-increment[-1]] = bias_adv1
                self.adv1.weight.data[0:-increment[-1], 0:] = weight_adv1
                self.v1.bias.data[0:-increment[-1]] = bias_v1
                self.v1.weight.data[0:-increment[-1], 0:] = weight_v1
        else:
            if increment[-2] > 0:
                self.adv1.bias.data = bias_adv1
                self.adv1.weight.data[0:, 0:-increment[-2]] = weight_adv1
                self.v1.bias.data = bias_v1
                self.v1.weight.data[0:, 0:-increment[-2]] = weight_v1
            else:
                self.adv1.bias.data = bias_adv1
                self.adv1.weight.data[0:, 0:] = weight_adv1
                self.v1.bias.data = bias_v1
                self.v1.weight.data[0:, 0:] = weight_v1

        bias_adv2 = self.adv2.bias.data
        weight_adv2 = self.adv2.weight.data
        self.adv2 = nn.Linear(self.hidden[-1], self.num_actions)

        bias_v2 = self.v2.bias.data
        weight_v2 = self.v2.weight.data
        self.v2 = nn.Linear(self.hidden[-1], 1)

        if increment[-1] > 0:
            self.adv2.bias.data = bias_adv2
            self.adv2.weight.data[:, 0:-increment[-1]] = weight_adv2
            self.v2.bias.data = bias_v2
            self.v2.weight.data[:, 0:-increment[-1]] = weight_v2
        else:
            self.adv2.bias.data = bias_adv2
            self.adv2.weight.data[:, 0:] = weight_adv2
            self.v2.bias.data = bias_v2
            self.v2.weight.data[:, 0:] = weight_v2

    def act(self, state, epsilon, mask, device):
        if np.random.rand() > epsilon:
            state = torch.tensor([state], dtype=torch.float32, device=device)
            mask = torch.tensor([mask], dtype=torch.float32, device=device)
            q_values = self.forward(state) + mask
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = np.random.randint(self.num_actions)
        return action


class DuelingDQNHER(nn.Module):
    def __init__(self, num_inputs, hidden, num_actions, non_linearity):
        super(DuelingDQNHER, self).__init__()

        self.num_inputs = num_inputs * 2
        self.hidden = hidden
        self.num_actions = num_actions

        self.non_linearity = non_linearity

        self.sharedLayers = nn.ModuleList()
        self.sharedLayers.append(nn.Linear(num_inputs, self.hidden[0]))

        previous = self.hidden[0]
        for hidden_layer_size in self.hidden[1:-1]:
            self.sharedLayers.append(nn.Linear(previous, hidden_layer_size))
            previous = hidden_layer_size

        self.adv1 = nn.Linear(previous, self.hidden[-1])
        self.adv2 = nn.Linear(self.hidden[-1], num_actions)

        self.v1 = nn.Linear(previous, self.hidden[-1])
        self.v2 = nn.Linear(self.hidden[-1], 1)

    def forward(self, state, ):
        for i in range(len(self.sharedLayers)):
            x = self.non_linearity(self.sharedLayers[i](x))

        a = self.non_linearity(self.adv1(x))
        a = self.adv2(a)

        v = self.non_linearity(self.v1(x))
        v = self.v2(v)

        return v + a - a.mean()

    def increase_capacity(self, increment):
        for i in range(len(self.hidden)):
            self.hidden[i] += increment[i]

        # Check whether the increment isn't zero
        if increment[0] > 0:
            weight = self.sharedLayers[0].weight.data
            self.sharedLayers[0] = nn.Linear(self.num_inputs, self.hidden[0])
            self.sharedLayers[0].weight.data[0:-increment[0], :] = weight

        for i in range(1, len(self.sharedLayers)):
            weight = self.sharedLayers[i].weight.data
            self.sharedLayers[i] = nn.Linear(self.hidden[i - 1], self.hidden[i])
            if increment[i] > 0:
                if increment[i - 1] > 0:
                    self.sharedLayers[i].weight.data[0:-increment[i], 0:-increment[i - 1]] = weight
                else:
                    self.sharedLayers[i].weight.data[0:-increment[i], 0:] = weight
            else:
                if increment[i - 1] > 0:
                    self.sharedLayers[i].weight.data[0:, 0:-increment[i - 1]] = weight
                else:
                    self.sharedLayers[i].weight.data[0:, 0:] = weight

        weight_adv1 = self.adv1.weight.data
        self.adv1 = nn.Linear(self.hidden[-2], self.hidden[-1])

        weight_v1 = self.v1.weight.data
        self.v1 = nn.Linear(self.hidden[-2], self.hidden[-1])
        if increment[-1] > 0:
            if increment[-2] > 0:
                self.adv1.weight.data[0:-increment[-1], 0:-increment[-2]] = weight_adv1
                self.v1.weight.data[0:-increment[-1], 0:-increment[-2]] = weight_v1
            else:
                self.adv1.weight.data[0:-increment[-1], 0:] = weight_adv1
                self.v1.weight.data[0:-increment[-1], 0:] = weight_v1
        else:
            if increment[-2] > 0:
                self.adv1.weight.data[0:, 0:-increment[-2]] = weight_adv1
                self.v1.weight.data[0:, 0:-increment[-2]] = weight_v1
            else:
                self.adv1.weight.data[0:, 0:] = weight_adv1
                self.v1.weight.data[0:, 0:] = weight_v1

        weight_adv2 = self.adv2.weight.data
        self.adv2 = nn.Linear(self.hidden[-1], self.num_actions)

        weight_v2 = self.v2.weight.data
        self.v2 = nn.Linear(self.hidden[-1], 1)

        if increment[-1] > 0:
            self.adv2.weight.data[:, 0:-increment[-1]] = weight_adv2
            self.v2.weight.data[:, 0:-increment[-1]] = weight_v2
        else:
            self.adv2.weight.data[:, 0:] = weight_adv2
            self.v2.weight.data[:, 0:] = weight_v2

    def act(self, state, epsilon, mask):
        if np.random.rand() > epsilon:
            state = torch.tensor([state], dtype=torch.float32, device=device)
            mask = torch.tensor([mask], dtype=torch.float32, device=device)
            q_values = self.forward(state) + mask
            action = q_values.max(1)[1].view(1, 1).item()
        else:
            action = np.random.randint(self.num_actions)
        return action
