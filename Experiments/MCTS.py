import numpy as np
import torch
import time
import copy

class MonteCarloSearchTree(object):
    def __init__(self, env, network, max_time, device, c, nu):
        self.c = c
        self.nu = nu
        self.device = device
        self.env = env
        self.network = network
        self.max_time = max_time
        state = env.get_observation()
        prior_probabilities = self.network.policy(torch.tensor(state, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
        self.root = State(device, c, nu, prior_probabilities=prior_probabilities)
        
    def search(self):
        t_0 = time.time()
        expansions = 0
        
        while time.time() - t_0 < self.max_time:
            action_seq = []
            
            current_node = self.root
            current_env = copy.deepcopy(self.env)
            
            current_action = None
            while not current_node.is_leaf():
                current_node, current_action = current_node.select() 
                current_env.step(current_action)
                action_seq.append(current_action)
                
            if current_env.solved():
                return (1.0, action_seq)
                
            current_node.expand(current_env, self.network)
            expansions += 1
            
            state = current_env.get_observation()
            value = self.network.value(torch.tensor(state, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
            
            while current_node is not None and current_action is not None:
                current_node.update(current_action, value)
                current_node = current_node.parent
                
        return (0.0, [])
                
        
class State(object):
    def __init__(self, device, c, nu, parent=None, prior_probabilities=None):
        self.c = c
        self.nu = nu
        self.device = device
        self.N = np.zeros(6)
        self.W = np.zeros(6)
        self.L = np.zeros(6)
        self.P = prior_probabilities
        self.parent = parent
        self.children = []
        
    def is_leaf(self):
        return len(self.children)==0
    
    def select(self):
        highest = 0.0
        highest_action = 0
        
        for action in range(6):
            uct = self.c*self.P[action]*(np.sqrt(np.sum(self.N))/(1+self.N[action])) + self.W[action] - self.L[action]
            if uct>highest:
                highest = uct
                highest_action = action
        self.L[highest_action] += self.nu
        return self.children[highest_action], highest_action
    
    def expand(self, env, network):
        for action in range(6):
            tmp_env = copy.deepcopy(env)
            tmp_env.step(action)
            state = tmp_env.get_observation()
            prior_probabilities = network.policy(torch.tensor(state, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
            self.children.append(State(self.device, self.c, self.nu, self,prior_probabilities))
            
    def update(self, action, value):
        self.W[action] = max(self.W[action], value)
        self.N[action] += 1
        self.L[action] -= self.nu