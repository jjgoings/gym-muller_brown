import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from gym import error, spaces, utils
from gym.utils import seeding

class MullerBrownContinuousEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        # state space
        self.x_min = -1.5
        self.x_max =  1.0
        self.y_min = -0.5
        self.y_max =  2.0
        self.observation_space = spaces.Box(low=np.array([self.x_min,self.y_min]), 
                                            high=np.array([self.x_max,self.y_max]),
                                            dtype=float)

        # action space
        self.action_min = -1
        self.action_max =  1 
        self.action_space = spaces.Box(low=self.action_min,
                                       high=self.action_max,shape=(2,),dtype=float) 

        # precompute some energies for plotting PES
        self.grid_points = 60
        self.energies = np.empty((self.grid_points,self.grid_points)) 
        x = np.linspace(self.x_min,self.x_max,self.grid_points)
        y = np.linspace(self.y_min,self.y_max,self.grid_points)
        for ix,iy in product(range(self.grid_points),range(self.grid_points)):
            self.energies[ix,iy] = self.energy((x[ix],y[iy]))

        self.reset()

    def plotPES(self):
        ''' Renders the continuous Muller Brown PES (environment) ''' 

        x = np.linspace(self.x_min,self.x_max,self.grid_points)
        y = np.linspace(self.y_min,self.y_max,self.grid_points)

        fig,ax = plt.subplots()
        im = plt.pcolormesh(x,y,self.energies.T, cmap='GnBu_r', vmax=10,shading='nearest')
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, ax=ax)

        plt.xlabel('x') 
        plt.ylabel('y') 
        cbar.set_label('energy')

        return fig 

    def render(self, mode='human'):
        self.plotPES()
        x,y = self.agent_position
        plt.plot(x,y,marker='o',color='#C91A09',markersize=8)

    def energy(self, state):
        ''' 
        Muller-Brown potential energy surface
        Parameters:
            state  : integer pair (ix,iy) from state 
            energy : float
        '''

        x,y = state

        A =  [-200, -100, -170, 15]
        a =  [-1, -1, -6.5, 0.7]
        b =  [0, 0, 11, 0.6]
        c =  [-10, -10, -6.5, 0.7]
        x0 = [1, 0, -0.5, -1]
        y0 = [0, 0.5, 1.5, 1]
        energy = 0.0
        for k in range(len(x0)):
            energy += (A[k]) *\
                np.exp(a[k]*(x-x0[k])**2 +\
                    b[k]*(x-x0[k])*(y-y0[k]) +\
                        c[k]*(y-y0[k])**2)
        return energy

    def set_state(self, state):
        self.agent_position = state

    def is_off_grid(self, state):
        x,y = state
        if (x >= self.x_max) or (x <= self.x_min):
            return True
        elif (y >= self.y_max) or (y <= self.y_min):
            return True
        else:
            return False

    def step(self, action):

        old_energy = self.energy(self.agent_position)
        new_state = self.agent_position + 0.2*action 
        done = False # we don't have a pre-set endpoint

        if not self.is_off_grid(new_state):
            self.set_state(new_state)
            new_energy = self.energy(new_state)
            reward = old_energy - new_energy
            return new_state, reward, done, {} 

        else:
            reward = -1e2 # penalize off-grid moves
            return self.agent_position, reward, done, {}

    def reset(self):
        new_position = self.observation_space.sample() 
        self.set_state(new_position)
        return self.agent_position

if __name__ == '__main__':
    env = MullerBrownContinuousEnv()

    for _ in range(10):
        obs,reward,_,_ = env.step(env.action_space.sample())
        print(obs,reward)
        env.render()
        plt.pause(0.01)
        plt.close()
    
   

