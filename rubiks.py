
# coding: utf-8

# TODO List:
#     - Reset/Scramble
#     - Move
#     - More elegent translate action (maybe store in array)
#     - More elegant move
#     - Render

# In[7]:


import numpy as np
import copy
import gym
import random
from gym import spaces
from gym.utils import seeding

class RubiksEnv(gym.Env):
    """
    See cartpole on Github
    Description:
    
    Source:
    
    Observation:
    
    Actions:
    
    Reward:
    The reward
    
    Starting State:
    
    
    Episode Termination:
    Episode terminates when either a cube is in the solved state (i.e. each side only has tiles of one colour) or when the step limit is reached.
    """
    
    def __init__(self, size=3, metric='quarter', pomdp=False, solved_reward=1.0, unsolved_reward=0.0, seed=None):
        self.size = size
        
        #Allocate space for Rubik's Cube sides. Each side get's a corresponding integer.
        self.U = (0*np.ones((self.size,self.size))).astype(int)
        self.L = (1*np.ones((self.size,self.size))).astype(int)
        self.F = (2*np.ones((self.size,self.size))).astype(int)
        self.R = (3*np.ones((self.size,self.size))).astype(int)
        self.B = (4*np.ones((self.size,self.size))).astype(int)
        self.D = (5*np.ones((self.size,self.size))).astype(int)
        
        self.orientation = (0,1,3)
        
        self.metric = metric
        self.pomdp = pomdp
        
        if self.metric is 'quarter':
            if self.pomdp:
                self.action_space = spaces.Discrete(16)
                self.observation_space = spaces.Box(low=0, high=5, dtype=np.uint8, shape=(3,self.size,self.size))
            else:
                self.action_space = spaces.Discrete(12)
                self.observation_space = spaces.Box(low=0, high=5, dtype=np.uint8, shape=(6,self.size,self.size))
        else:
            if self.pomdp:
                self.action_space = spaces.Discrete(23)
                self.observation_space = spaces.Box(low=0, high=5, dtype=np.uint8, shape=(3,self.size,self.size))
            else:
                self.action_space = spaces.Discrete(18)
                self.observation_space = spaces.Box(low=0, high=5, dtype=np.uint8, shape=(6,self.size,self.size))
        
        self._action_set = [i for i in range(self.action_space.n)]        
                
        self.solved_reward = solved_reward
        self.unsolved_reward = unsolved_reward
        
        self.seed(seed)
        
    def seed(self, seed=None):
        """"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, steps = 20, orientation = False):
        """"""
        self.U = (0*np.ones((self.size,self.size))).astype(int)
        self.L = (1*np.ones((self.size,self.size))).astype(int)
        self.F = (2*np.ones((self.size,self.size))).astype(int)
        self.R = (3*np.ones((self.size,self.size))).astype(int)
        self.B = (4*np.ones((self.size,self.size))).astype(int)
        self.D = (5*np.ones((self.size,self.size))).astype(int)
        
        for step in range(steps):
            action = self.np_random.choice(self._action_set)
            self.step(int(action))
        observation = self.get_observation()
        return observation
        
    def move(self, side, sign, times, orientation):
        """"""
        if orientation is None:
            if side is 0:
                self.U = np.rot90(self.U,times*-sign)
                if times < 2:
                    if sign > 0:
                        Ftmp = copy.copy(self.F[0,:])
                        self.F[0,:] = self.R[0,:]
                        Ltmp = copy.copy(self.L[0,:])
                        self.L[0,:] = Ftmp
                        Btmp = copy.copy(self.B[0,:])
                        self.B[0,:] = Ltmp
                        self.R[0,:] = Btmp
                    if sign < 0:
                        Ftmp = copy.copy(self.F[0,:])
                        self.F[0,:] = self.L[0,:]
                        Rtmp = copy.copy(self.R[0,:])
                        self.R[0,:] = Ftmp
                        Btmp = copy.copy(self.B[0,:])
                        self.B[0,:] = Rtmp
                        self.L[0,:] = Btmp
                else:
                    Ftmp = copy.copy(self.F[0,:])
                    self.F[0,:] = self.B[0,:]
                    self.B[0,:] = Ftmp
                    Rtmp = copy.copy(self.R[0,:])
                    self.R[0,:] = self.L[0,:]
                    self.L[0,:] = Rtmp
                    
            if side is 1:
                self.L = np.rot90(self.L,times*-sign)
                if times < 2:
                    if sign > 0:
                        Ftmp = copy.copy(self.F[:,0])
                        self.F[:,0] = self.U[:,0]
                        Dtmp = copy.copy(self.D[:,0][::-1])
                        self.D[:,0] = Ftmp
                        Btmp = copy.copy(self.B[:,-1][::-1])
                        self.B[:,-1] = Dtmp
                        self.U[:,0] = Btmp
                    if sign < 0:
                        Ftmp = copy.copy(self.F[:,0])
                        self.F[:,0] = self.D[:,0]
                        Utmp = copy.copy(self.U[:,0][::-1])
                        self.U[:,0] = Ftmp
                        Btmp = copy.copy(self.B[:,-1][::-1])
                        self.B[:,-1] = Utmp
                        self.D[:,0] = Btmp
                else:
                    Ftmp = copy.copy(self.F[:,0][::-1])
                    self.F[:,0] = self.B[:,-1][::-1]
                    self.B[:,-1] = Ftmp
                    Utmp = copy.copy(self.U[:,0])
                    self.U[:,0] = self.D[:,0]
                    self.D[:,0] = Utmp
                    
            
            if side is 2:
                self.F = np.rot90(self.F,times*-sign)
                if times < 2:
                    if sign > 0:
                        Utmp = copy.copy(self.U[-1,:])
                        self.U[-1,:] = self.L[:,-1][::-1]
                        Rtmp = copy.copy(self.R[:,0][::-1])
                        self.R[:,0] = Utmp
                        Dtmp = copy.copy(self.D[0,:])
                        self.D[0,:] = Rtmp
                        self.L[:,-1] = Dtmp
                    if sign < 0:
                        Utmp = copy.copy(self.U[-1,:][::-1])
                        self.U[-1,:] = self.R[:,0]
                        Ltmp = copy.copy(self.L[:,-1])
                        self.L[:,-1] = Utmp
                        Dtmp = copy.copy(self.D[0,:][::-1])
                        self.D[0,:] = Ltmp
                        self.R[:,0] = Dtmp
                else:
                    Utmp = copy.copy(self.U[-1,:][::-1])
                    self.U[-1,:] = self.D[0,:][::-1]
                    self.D[0,:] = Utmp
                    Rtmp = copy.copy(self.R[:,0][::-1])
                    self.R[:,0] = self.L[:,2]
                    self.L[:,-1] = Rtmp
            
            if side is 3:
                self.R = np.rot90(self.R,times*-sign)
                if times < 2:
                    if sign > 0:
                        Utmp = copy.copy(self.U[:,-1][::-1])
                        self.U[:,-1] = self.F[:,-1]
                        Btmp = copy.copy(self.B[:,0][::-1])
                        self.B[:,0] = Utmp
                        Dtmp = copy.copy(self.D[:,-1])
                        self.D[:,-1] = Btmp
                        self.F[:,-1] = Dtmp
                    if sign < 0:
                        Utmp = copy.copy(self.U[:,-1]) 
                        self.U[:,-1] = self.B[:,0][::-1]
                        Ftmp = copy.copy(self.F[:,-1])
                        self.F[:,-1] = Utmp
                        Dtmp = copy.copy(self.D[:,-1][::-1])
                        self.D[:,-1] = Ftmp
                        self.B[:,0] = Dtmp
                else:
                    Utmp = copy.copy(self.U[:,-1])
                    self.U[:,-1] = self.D[:,-1]
                    self.D[:,-1] = Utmp
                    Ftmp = copy.copy(self.F[:,-1][::-1])
                    self.F[:,-1] = self.B[:,0][::-1]
                    self.B[:,0] = Ftmp
                    
                    
            if side is 4:
                self.B = np.rot90(self.B,times*-sign)
                if times < 2:
                    if sign > 0:
                        Utmp = copy.copy(self.U[0,:][::-1])
                        self.U[0,:] = self.R[:,-1]
                        Ltmp = copy.copy(self.L[:,0])
                        self.L[:,0] = Utmp
                        Dtmp = copy.copy(self.D[-1,:][::-1])
                        self.D[-1,:] = Ltmp
                        self.R[:,-1] = Dtmp
                        
                    if sign < 0:
                        Utmp = copy.copy(self.U[0,:])
                        self.U[0,:] = self.L[:,0][::-1]
                        Rtmp = copy.copy(self.R[:,-1][::-1])
                        self.R[:,-1] = Utmp
                        Dtmp = copy.copy(self.D[-1,:])
                        self.D[-1,:] = Rtmp
                        self.L[:,0] = Dtmp
                else:
                    Utmp = copy.copy(self.U[0,:][::-1])
                    self.U[0,:] = self.D[-1,:][::-1]
                    self.D[-1,:] = Utmp
                    Rtmp = copy.copy(self.R[:,-1][::-1])
                    self.R[:,-1] = self.L[:,0][::-1]
                    self.L[:,0] = Rtmp
                    
                    
            if side is 5:
                self.D = np.rot90(self.D,times*-sign)
                if times < 2:
                    if sign > 0:
                        Ftmp = copy.copy(self.F[-1,:])
                        self.F[-1,:] = self.L[-1,:]
                        Rtmp = copy.copy(self.R[-1,:])
                        self.R[-1,:] = Ftmp
                        Btmp = copy.copy(self.B[-1,:])
                        self.B[-1,:] = Rtmp
                        self.L[-1,:] = Btmp
                    if sign < 0:
                        Ftmp = copy.copy(self.F[-1,:])
                        self.F[-1,:] = self.R[-1,:]
                        Ltmp = copy.copy(self.L[-1,:])
                        self.L[-1,:] = Ftmp
                        Btmp = copy.copy(self.B[-1,:])
                        self.B[-1,:] = Ltmp
                        self.R[-1,:] = Btmp
                else:
                    Ftmp = copy.copy(self.F[-1,:])
                    self.F[-1,:] = self.B[-1,:]
                    self.B[-1,:] = Ftmp
                    Ltmp = copy.copy(self.L[-1,:])
                    self.L[-1,:] = self.R[-1,:]
                    self.R[-1,:] = Ltmp
        else:
            raise NotImplementedError('Orientation')
        
    def translate_action(self, action):
        """"""
        #TODO encode this in ACTION_MEANING_QUARTER_METRIC
        side = None
        sign = None
        times = None
        orientation = None
        
        if action in [6,7,8,9,10,11]:
            sign = -1.0
            times = 1.0
            
        if action in [0,1,2,3,4,5]:
            sign = 1.0
            times = 1.0
        
        if action is 0 or action is 6:
            side = 0
        if action is 1 or action is 7:
            side = 1
        if action is 2 or action is 8:
            side = 2
        if action is 3 or action is 9:
            side = 3
        if action is 4 or action is 10:
            side = 4
        if action is 5 or action is 11:
            side = 5

        if self.metric is 'half':
            sign = 1.0
            times = 2.0
            
            if action is 12:
                side = 0
            if action is 13:
                side = 1
            if action is 14:
                side = 2
            if action is 15:
                side = 3
            if action is 16:
                side = 4
            if action is 17:
                side = 5
                
        if self.pomdp:
            assert side is None
            assert sign is None
            assert times is None 
            
            if action in [12, 18]:
                orientation = "North"
            if action in [13,19]:
                orientation = "West"
            if action in [14,20]:
                orientation = "South"
            if action in [15,21]:
                orientation = "East"
            if action is 22:
                orientation = "Antipode"
                          
        return side, sign, times, orientation
    
    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        side, sign, times, orientation = self.translate_action(action)
        self.move(side,sign,times, orientation)
        
        observation = self.get_observation()
        done = self.solved()
        if done:
            reward = self.solved_reward
        else:
            reward = self.unsolved_reward
            
        information = {}
        
        return observation, reward, done, information
        
    def solved(self):
        """"""
        sides = [self.U, self.L, self.F, self.R, self.B, self.D]
        
        for index, side in enumerate(sides):
            if not np.all(side == index):
                return 0
            
        return 1
    
    def pretty_print(self):
        """"""
        emptysymbol = np.chararray((self.size, self.size), unicode=True)
        emptysymbol[:] = '-'
        matrix = np.vstack((np.hstack((emptysymbol,self.U.astype(int),emptysymbol,emptysymbol)),
        np.hstack((self.L.astype(int),self.F.astype(int),self.R.astype(int),self.B.astype(int))),
        np.hstack((emptysymbol,self.D.astype(int),emptysymbol,emptysymbol))))
            
        print(matrix)
        
    @property
    def _n_actions(self):
        """"""
        return len(self._action_set)
    
    def render(self):
        """"""
        raise NotImplementedError('Render not implemented')
    
    def close(self):
        """"""
        raise NotImplementedError('close not implemented')
        
    def get_action_meanings(self):
        """"""
        if self.metric is 'quarter':
            if self.pomdp:
                return [ACTION_MEANING_QUARTER_METRIC_POMDP[i] for i in self._action_set]
            else:
                return [ACTION_MEANING_QUARTER_METRIC[i] for i in self._action_set]
        else:
            if self.pomdp:
                return [ACTION_MEANING_HALF_METRIC_POMDP[i] for i in self._action_set]
            else:
                return [ACTION_MEANING_HALF_METRIC[i] for i in self._action_set]
                
        
    def get_observation(self):
        """"""
        sides = [self.U, self.L, self.F, self.R, self.B, self.D]
        if self.pomdp:
            raveled_cube = np.array([sides[self.orientation[0]],sides[self.orientation[1]],sides[self.orientation[2]]]).ravel()
            one_hot = np.eye(6)[raveled_cube]
            return one_hot.reshape(-1)
        else:
            raveled_cube = np.array(sides).ravel()
            one_hot = np.eye(6)[raveled_cube]
            
            return one_hot.reshape(-1)
        
    
    
ACTION_MEANING_QUARTER_METRIC = {
    0 : "U",
    1 : "L",
    2 : "F",
    3 : "R",
    4 : "B",
    5 : "D",
    6 : "U'",
    7 : "L'",
    8 : "F'",
    9 : "R'",
    10 : "B'",
    11 : "D'"
}

ACTION_MEANING_QUARTER_METRIC_POMDP = {
    0 : "U",
    1 : "L",
    2 : "F",
    3 : "R",
    4 : "B",
    5 : "D",
    6 : "U'",
    7 : "L'",
    8 : "F'",
    9 : "R'",
    10 : "B'",
    11 : "D'",
    12 : "North",
    13 : "West",
    14 : "South",
    15 : "East"
}

ACTION_MEANING_HALF_METRIC = {
    0 : "U",
    1 : "L",
    2 : "F",
    3 : "R",
    4 : "B",
    5 : "D",
    6 : "U'",
    7 : "L'",
    8 : "F'",
    9 : "R'",
    10 : "B'",
    11 : "D'",
    12 : "U2",
    13 : "L2",
    14 : "F2",
    15 : "R2",
    16 : "B2",
    17 : "D2"
}

ACTION_MEANING_HALF_METRIC_POMDP = {
    0 : "U",
    1 : "L",
    2 : "F",
    3 : "R",
    4 : "B",
    5 : "D",
    6 : "U'",
    7 : "L'",
    8 : "F'",
    9 : "R'",
    10 : "B'",
    11 : "D'",
    12 : "U2",
    13 : "L2",
    14 : "F2",
    15 : "R2",
    16 : "B2",
    17 : "D2",
    18 : "North",
    19 : "West",
    20 : "South",
    21 : "East",
    22 : "Antipode"
}



