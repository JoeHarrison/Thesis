import numpy as np
import copy

class EXP3S(object):
    def __init__(self, N_arms, eta, beta, epsilon):
        self.N_arms = N_arms
        
        self.eta = eta
        self.beta = beta
        self.epsilon = epsilon
        
        self.weights = np.zeros((2,self.N_arms))
        
        self.t = 0
        self.alpha = None
        
    def exp3(self, arm):
        return np.exp(self.weights[0][arm])/np.sum(np.exp(self.weights[0]))
        
    def sample_probability(self, arm):
        return (1-self.epsilon)*self.exp3(arm) + self.epsilon/self.N_arms
    
    def sample_probabilities(self):
        sample_probabilities = []
        for arm in range(self.N_arms):
            sample_probabilities.append(self.sample_probability(arm))
        return sample_probabilities
    
    def importance_sampling_reward(self, reward, previous_action):
        #2.2.1
        importance_sampled_rewards = []
        for i in range(self.N_arms):
            numerator = self.beta
            if previous_action==i:
                numerator += reward
            importance_sampled_rewards.append(numerator/self.sample_probability(i))

        return importance_sampled_rewards
        
    def update_weights(self, reward, previous_action):
        old_weights = copy.deepcopy(self.weights[0])
        
        self.t += 1
        alpha = 1/self.t
        
        importance_sampled_rewards = self.importance_sampling_reward(reward, previous_action)
        
        for i in range(self.N_arms):
            self.weights[0][i] = np.log((1-alpha)*np.exp(self.weights[1][i]
                                        + self.eta*importance_sampled_rewards[i]) 
                                        + (alpha/(self.N_arms - 1)*np.sum([np.exp(self.weights[1][j] 
                                        + self.eta*importance_sampled_rewards[j]) if j!=i else 0 for j in range(self.N_arms)])))
        
        self.weights[1] = old_weights