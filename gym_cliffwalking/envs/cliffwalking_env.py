import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from copy import deepcopy
import numpy as np

class CliffWalkingEnv(gym.Env):
    ''' Cliff Walking Environment

        See the README.md file from https://github.com/caburu/gym-cliffwalking
    '''
    # There is no renderization yet
    # metadata = {'render.modes': ['human']}

    def observation(self, state):
        return state[0] * self.cols + state[1]

    def __init__(self):
        self.rows = 4
        self.cols = 12
        self.start = [0,0]
        self.goal = [11,0]
        self.current_state = None

        # There are four actions: up, down, left and right
        self.action_space = spaces.Discrete(4)
        self.actions = [(0, +1), (+1, 0), (0, -1), (-1, 0)]

         # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Discrete(self.rows*self.cols)

        self.arrow = None
        self.position = None
        self.ax = None

    def step(self, action):
        new_state = deepcopy(self.current_state)

        if action == 0: #right
            new_state[0] = min(new_state[0]+1, self.cols-1)
        elif action == 1: #down
            new_state[1] = max(new_state[1]-1, 0)
        elif action == 2: #left
            new_state[0] = max(new_state[0]-1, 0)
        elif action == 3: #up
            new_state[1] = min(new_state[1]+1, self.rows-1)
        else:
            raise Exception("Invalid action.")
        self.current_state = new_state

        reward = -1.0
        is_terminal = False
        if self.current_state[1] == 0 and self.current_state[0] > 0:
            if self.current_state[0] < self.cols - 1:
                reward = -100.0
                self.current_state = deepcopy(self.start)
            else:
                is_terminal = True
        
        self.arrow = np.subtract(self.current_state,self.position)
        self.position = self.current_state
        return self.position, reward, is_terminal, {}

    def reset(self):
        self.current_state = self.start
        self.ax = None
        self.position = np.array(self.start)
        self.arrow = np.array((0, 0))
        return self.current_state

    def render(self, mode='human'):
        fig = plt.figure()
        self.ax = fig.gca()

        # Background colored by wind strength
        cliff = np.vstack([[0,1,1,1,1,1,1,1,1,1,1,0]])
        self.ax.imshow(cliff, aspect='equal', origin='lower', cmap='Blues')

        # Annotations at start and goal positions
        self.ax.annotate("G", self.goal, size=25, color='gray', ha='center', va='center')
        self.ax.annotate("S", self.start, size=25, color='gray', ha='center', va='center')

        # Thin grid lines at minor tick mark locations
        self.ax.set_xticks(np.arange(-0.5, self.cols), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.rows), minor=True)
        self.ax.grid(which='minor', color='black', linewidth=0.20)
        self.ax.tick_params(which='both', length=0)
        self.ax.set_frame_on(True)

        # Arrow pointing from the previous to the current position
        if (self.arrow == 0).all():
            patch = mpatches.Circle(self.position, radius=0.05, color='black', zorder=1)
        else:
            patch = mpatches.FancyArrow(*(self.position - self.arrow), *self.arrow, color='black',
                                        zorder=2, fill=True, width=0.05, head_width=0.25,
                                        length_includes_head=True)
        self.ax.add_patch(patch)

        pass

    def close(self):
        pass
