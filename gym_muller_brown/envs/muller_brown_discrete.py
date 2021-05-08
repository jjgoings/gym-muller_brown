import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import error, spaces, utils
from gym.utils import seeding

class MullerBrownDiscreteEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.extent = [-1.5,1,-0.5,2]  # [xmin,xmax,ymin,ymax]
        self.x_values = np.linspace(self.extent[0],self.extent[1],self.m)
        self.y_values = np.linspace(self.extent[2],self.extent[3],self.n)
        self.state_space = range(self.m*self.n)
        self.action_space = {'U': self.m, 'D': -self.m, 'L': -1, 'R': 1, 'N': 0}
        self.energies = [self.energy(i) for i in self.state_space] # precompute energy
        self.reset()

    def plotPES(self):
        ''' Renders the discretized Muller Brown PES (environment) ''' 

        PES = np.asarray(self.energies).reshape(self.n,self.m)

        dx = self.x_values[1] - self.x_values[0] 
        dy = self.y_values[1] - self.y_values[0] 

        # in order to center ticks about grid centers, we have to shift axes,  
        #   so new axes must be len(m+1) and len(n+1) and offset by half,
        #   though the inter-grid spacing need to remain unchanged
        x = np.linspace(self.extent[0],self.extent[1]+dx,self.m+1) - 0.5*dx
        y = np.linspace(self.extent[2],self.extent[3]+dy,self.n+1) - 0.5*dy

        fig,ax = plt.subplots()
        im = plt.pcolormesh(x,y,PES, edgecolors='k', cmap='GnBu_r', linewidth=0.5,vmax=10)
        #im = plt.pcolormesh(x,y,PES)
        ax.set_aspect('equal')
        cbar = fig.colorbar(im, ax=ax)

        # add labels to grid centers
        ax.set_xticks(np.linspace(self.extent[0],self.extent[1],min(self.m,6)))
        ax.set_yticks(np.linspace(self.extent[2],self.extent[3],min(self.n,6)))

        plt.xlabel('x') 
        plt.ylabel('y') 
        cbar.set_label('energy')

        return fig 

    def state_to_coord(self,state):
        ''' Returns physical coordinates x,y from integer in state_space 
        '''
        ix = state % self.m
        iy = state // self.m
        x = self.x_values[ix]
        y = self.y_values[iy]
        return x, y

    def render(self, mode='human'):
        self.plotPES()
        x,y = self.state_to_coord(self.agent_position)
        plt.plot(x,y,marker='o',color='#C91A09',markersize=8)

    def energy(self, state):
        ''' 
        Muller-Brown potential energy surface
        Parameters:
            state  : integer from state_space
            energy : float
        '''

        try:
            return self.energies[state]
 
        except AttributeError:
            x,y = self.state_to_coord(state)

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

    def is_off_grid(self, new_state, old_state):
        if new_state not in self.state_space:
            return True
        elif old_state % self.m == 0 and new_state % self.m == self.m - 1:
            return True
        elif old_state % self.m == self.m - 1 and new_state % self.m == 0:
            return True
        else:
            return False

    def step(self, action):

        old_energy = self.energy(self.agent_position)
        new_state = self.agent_position + self.action_space[action]
        done = False # we don't have a pre-set endpoint

        if not self.is_off_grid(new_state, self.agent_position):
            self.set_state(new_state)
            new_energy = self.energy(new_state)
            reward = old_energy - new_energy
            return new_state, reward, done, {} 

        else:
            reward = -1e2 # penalize off-grid moves
            return self.agent_position, reward, done, {}

    def reset(self):
        new_position = np.random.choice(self.state_space)
        self.set_state(new_position)
        return self.agent_position

    def sample_actions(self):
        return np.random.choice([*self.action_space])

if __name__ == '__main__':
    env = MullerBrownDiscreteEnv(14,10)
    env.render()
    plt.show()
   

