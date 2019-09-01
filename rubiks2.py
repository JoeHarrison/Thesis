
# coding: utf-8



import numpy as np
from PIL import Image
import copy
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

class RubiksEnv2(gym.Env):
    def __init__(self, size=2, metric='quarter', pomdp=False, solved_reward=1.0, unsolved_reward=-1.0, seed=None):
        self.size = size

        #Allocate space for Rubik's Cube sides. Each side get's a corresponding integer.
        self.U = (0*np.ones((self.size,self.size))).astype(int)
        self.L = (1*np.ones((self.size,self.size))).astype(int)
        self.F = (2*np.ones((self.size,self.size))).astype(int)
        self.R = (3*np.ones((self.size,self.size))).astype(int)
        self.B = (4*np.ones((self.size,self.size))).astype(int)
        self.D = (5*np.ones((self.size,self.size))).astype(int)

        self.orientation = (0, 1, 3)

        self.metric = metric
        self.pomdp = pomdp

        if self.metric is 'quarter':
            self.action_space = spaces.Discrete(6)
            self.observation_space = spaces.Box(low=0, high=5, dtype=np.uint8, shape=(6, self.size, self.size))

        self._action_set = [i for i in range(self.action_space.n)]

        self.solved_reward = solved_reward
        self.unsolved_reward = unsolved_reward

        self.seed(seed)

        self.ACTION_MEANING_QUARTER_METRIC = {
            0 : "D",
            1 : "R",
            2 : "B",
            3 : "D'",
            4 : "R'",
            5 : "B'"
        }

        # For debugging purposes
        self.last_scramble = []

    def seed(self, seed=None):
        """"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, steps = 14, orientation = False):
        """"""
        self.U = (0*np.ones((self.size, self.size))).astype(int)
        self.L = (1*np.ones((self.size, self.size))).astype(int)
        self.F = (2*np.ones((self.size, self.size))).astype(int)
        self.R = (3*np.ones((self.size, self.size))).astype(int)
        self.B = (4*np.ones((self.size, self.size))).astype(int)
        self.D = (5*np.ones((self.size, self.size))).astype(int)

        self.last_scramble = []
        for step in range(steps):
            action = self.np_random.choice(self._action_set)
            self.last_scramble.append(action)
            self.step(int(action))
        if self.solved():
            self.reset(steps, orientation)

        observation = self.get_observation()

        return observation

    def curriculum_reset(self, level=6*14 - 1, orientation = False):
        """"""
        self.U = (0*np.ones((self.size, self.size))).astype(int)
        self.L = (1*np.ones((self.size, self.size))).astype(int)
        self.F = (2*np.ones((self.size, self.size))).astype(int)
        self.R = (3*np.ones((self.size, self.size))).astype(int)
        self.B = (4*np.ones((self.size, self.size))).astype(int)
        self.D = (5*np.ones((self.size, self.size))).astype(int)

        self.last_scramble = []
        for step in range((level // self._n_actions)):
            action = self.np_random.choice(self._action_set)
            self.last_scramble.append(action)
            self.step(int(action))

        action = self.np_random.choice(self._action_set[:(level % self._n_actions) + 1])
        self.last_scramble.append(action)
        self.step(int(action))

        if self.solved():
            self.curriculum_reset(level, orientation)

        observation = self.get_observation()

        return observation

    def force_last_action_reset(self, level=6*14 -1, orientation = False):
        self.U = (0*np.ones((self.size, self.size))).astype(int)
        self.L = (1*np.ones((self.size, self.size))).astype(int)
        self.F = (2*np.ones((self.size, self.size))).astype(int)
        self.R = (3*np.ones((self.size, self.size))).astype(int)
        self.B = (4*np.ones((self.size, self.size))).astype(int)
        self.D = (5*np.ones((self.size, self.size))).astype(int)

        self.last_scramble = []
        for step in range((level // self._n_actions)):
            action = self.np_random.choice(self._action_set)
            self.last_scramble.append(action)
            self.step(int(action))

        action = level % 6
        self.last_scramble.append(action)
        self.step(int(action))

        if self.solved():
            self.curriculum_reset(level, orientation)

        observation = self.get_observation()

        return observation

    def move(self, side, sign, times, orientation):
        """"""
        if orientation is None:
            if side is 0:
                self.U = np.rot90(self.U, times*-sign)
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

        if action in [3,4,5]:
            sign = -1.0
            times = 1.0

        if action in [0,1,2]:
            sign = 1.0
            times = 1.0

        if action is 0 or action is 3:
            side = 5
        if action is 1 or action is 4:
            side = 3
        if action is 2 or action is 5:
            side = 4

        return side, sign, times, orientation

    def step(self, action):
        action =  int(action)
        assert self.action_space.contains(action), "Invalid action"
        side, sign, times, orientation = self.translate_action(action)
        self.move(side, sign, times, orientation)

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
        colordict = {0: [255, 255, 255],
                     1: [255, 127, 0],
                     2: [0, 255, 0],
                     3: [255, 0, 0],
                     4: [0, 0, 255],
                     5: [255, 255, 0]}

        factor = 60
        square = int(factor/self.size)
        width = factor*4
        height = factor*3

        image = np.ones((height, width, 3), dtype='uint8')*127
        for i in range(self.size):
            for j in range(self.size):
                # UP
                image[i*square:(i+1)*square, factor + j*square:factor + (j+1)*square] = colordict[self.U[i, j]]

                # RIGHT
                image[factor + i*square: factor + (i+1)*square, j*square:(j+1)*square] = colordict[self.L[i, j]]

                # FRONT
                image[factor + i*square: factor + (i+1)*square, factor + j*square: factor + (j+1)*square] = colordict[self.F[i, j]]

                # Right
                image[factor + i*square: factor + (i+1)*square, 2*factor + j*square: 2*factor + (j+1)*square] = colordict[self.R[i, j]]

                # Back
                image[factor + i*square: factor + (i+1)*square, 3*factor + j*square: 3*factor + (j+1)*square] = colordict[self.B[i, j]]

                # DOWN
                image[2*factor + i*square: 2*factor + (i+1)*square, factor + j*square: factor + (j+1)*square] = colordict[self.D[i, j]]
        plt.imshow(image)
        plt.show()

    def close(self):
        """"""
        raise NotImplementedError('close not implemented')

    def get_action_meanings(self):
        """"""
        if self.metric is 'quarter':
            if self.pomdp:
                return [self.ACTION_MEANING_QUARTER_METRIC_POMDP[i] for i in self._action_set]
            else:
                return [self.ACTION_MEANING_QUARTER_METRIC[i] for i in self._action_set]
        else:
            if self.pomdp:
                return [self.ACTION_MEANING_HALF_METRIC_POMDP[i] for i in self._action_set]
            else:
                return [self.ACTION_MEANING_HALF_METRIC[i] for i in self._action_set]


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





if __name__ == "__main__":
    env = RubiksEnv2(size=2, metric='quarter', pomdp=False, solved_reward=1.0, unsolved_reward=-1.0, seed=None)
    env.force_last_action_reset(7)
    env.render()

