# gym-muller_brown

<p align="center">
<img src="/img/mb_pes_10x10.png">
</p>

This environment discretizes the 2D Müller-Brown potential and incorporates it into
an OpenAI Gym compatible format. At the moment, the environment is essentially a
GridWorld variant, except there is no pre-set endpoint to an episode, so the
agent can, in principle, move around the potential energy surface indefinitely.

One of the future goals is to add a continuous-space Müller-Brown potential environment.

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- Matplotlib 

## Installation

To install, you can clone this repository and install the dependencies with `pip`:

```
git clone https://github.com/jjgoings/gym-muller_brown.git
cd gym-muller_brown
pip install -e .
```

## Basic Usage
Here is an example of loading and rendering the environment:

```
import gym
import matplotlib.pyplot as plt

env = gym.make('gym_muller_brown:MullerBrownDiscrete-v0',m=8,n=8)
env.render()
plt.show()
```

`m` and `n` correspond to the number of grid points along the x and y axes, respectively.
Here we make an 8 by 8 grid, and the agent (red dot) is initialized at random. 

<p align="center">
<img src="/img/mb_agent_8x8a.png">
</p>

If we sample an action, 

```
action = env.sample_actions() # samples at random
print("Action: ", action)
env.step(action)
```

which in this case randomly selects up or `'U'`, i.e. 

```
Action:  U
```

so then upon taking the step we end up like this 

<p align="center">
<img src="/img/mb_agent_8x8b.png">
</p>

### Actions

Possible actions:
- `U`: Go up one space
- `D`: Go down one space
- `L`: Go left one space
- `R`: Go right one space
- `N`: Do nothing 

### Rewards

Rewards correspond to a decrease in energy. Since we want positive rewards, this means that the reward is the negative change in energy. So as an example, if the energy change after taking a step is -10, the reward will be +10. If the action chosen is to do nothing, the reward is zero.

If the agent tries to move outside the box, it will be penalized by a reward of -100, but the agent will remain in place.

### Müller Brown Potential

The Müller-Brown potential (K. Müller, L. Brown, Theor. Chim. Acta 1979, 53, 75.) is a common potential energy surface used in theoretical chemistry, with three minima and two saddle points, which make it a simple but effective test for optimization and path finding algorithms.

For now, the potential is defined over X = [-1.5,1.0] and Y = [-0.5,2.0].

The energy given coordinates `x` and `y` is defined by:

```
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
```

There is a global minimum at `(x,y) = (-0.558,1.442)` with an energy `E = -146.6995`. 

## Reinforcement Learning

Here is a simple Q-learning example on a 9 by 9 discretization of the Müller Brown PES

```
import gym
import numpy as np 
import matplotlib.pyplot as plt 

env = gym.make('gym_muller_brown:MullerBrownDiscrete-v0',m=9,n=9)

# define the agent's policy (argmax of Q function)
def policy(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]

# model hyperparameters
learning_rate = 0.1
discount = 0.99
epsilon = 1.0

# initialize Q-table to zero
Q = {}
for state in env.state_space:
    for action in [*env.action_space]: # action_space is a dict, so take keys
        Q[state, action] = 0

max_episode = 1000
steps_per_episode = 50
episode_rewards = []
for episode in range(max_episode):

    observation = env.reset()
    inital_energy = env.energies[observation]

    total_reward = 0
    for idx in range(steps_per_episode):
        rand = np.random.random()

        if np.random.random() < (1 - epsilon):
            # choose argmax Q
            action = policy(Q,observation,[*env.action_space])
        else:
            # choose random action 
            action = env.sample_actions()

        new_observation, reward, done, info = env.step(action)
        total_reward += reward

        new_action = policy(Q, new_observation, [*env.action_space]) # Q learning always take argmax
        Q[observation,action] = Q[observation,action] + learning_rate*(reward + \
                    discount*Q[new_observation,new_action] - Q[observation,action])
        observation = new_observation

    epsilon = epsilon*np.exp(-0.5*episode/max_episode) # naive decay of epsilon
    episode_rewards.append(total_reward - inital_energy)

plt.plot(episode_rewards)
plt.xlabel('episode')
plt.ylabel('episode reward')
plt.show()
```

which yields the reward profile over training:

<p align="center">
<img src="/img/q_learning_rewards.png">
</p>

### Plotting the learned policy

If you want to plot the learned policy from the RL training over the PES, here's one way you could do it following the naive Q-learning example above:

```
def plotQ(Q_dict):

    env.plotPES()  # PES is our background

    Q = np.zeros((env.m,env.n,len(env.action_space)))
    action_idx = {'U':0,'D':1,'L':2,'R':3,'N':4} # map action representation
    idx_action = {0:'^',1:'v',2:'<',3:'>',4:'o'} # map action to marker
    idx_color = {0:'k',1:'k',2:'k',3:'k',4:'#C91A09'} # map action to color
    for state,action in Q_dict:
        x = state % env.m
        y = state // env.m
        a = action_idx[action]
        Q[x,y,a] = Q_dict[(state,action)]

    for i,x in enumerate(env.x_values):
        for j,y  in enumerate(env.y_values):
            plt.plot(x, y, marker=idx_action[np.argmax(Q[i,j,:])],color=idx_color[np.argmax(Q[i,j,:])],markersize=4)

    plt.show()
```

which yields

<p align="center">
<img src="/img/q_learning_9x9.png">
</p>

The arrow markers indicate the optimal action learned for each grid point (state). Nice to see that the agent moves away from the edges. The red dots indicate that the optimal action in these states is to do nothing. You can trace the arrows from any state to see the path the agent would follow if following the learned policy. Note they all end up in two of the minima.




