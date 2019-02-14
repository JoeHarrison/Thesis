import numpy as np
import random
# from pytorchneat import NeuralNetwork
from feedforwardnetwork import NeuralNetwork

class XORTask(object):
    
    # Default XOR input/output pairs
    INPUTS  = [(0,0), (0,1), (1,0), (1,1)]
    OUTPUTS = [(0,), (1,), (1,), (0,)]
    EPSILON = 1e-100
    
    def __init__(self, do_all=True):
        self.do_all = do_all
        self.INPUTS = np.array(self.INPUTS, dtype=float)
        self.OUTPUTS = np.array(self.OUTPUTS, dtype=float)
    
    def evaluate(self, network, verbose=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)

#             network = NeuralNetwork2(network)
        
        pairs = list(zip(self.INPUTS, self.OUTPUTS))
        random.shuffle(pairs)
        if not self.do_all:
            pairs = [random.choice(pairs)]
        rmse = 0.0
        for (i, target) in pairs:
            # network.reset()
            output = network(np.array([i]))
            err = (target - output.item())
            err[abs(err) < self.EPSILON] = 0;
            err = (err ** 2).mean()
            # Add error
            if verbose:
                print("%r -> %r (%.2f)" % (i, output, err))
            rmse += err 

        score = 1/(1+np.sqrt(rmse / len(pairs)))
        return {'fitness': score, 'info' : 0}
        
    def solve(self, network):
        return int(self.evaluate(network)['fitness'] > 0.9)
