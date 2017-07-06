# utilities
import numpy as np
import pickle, os, sys, pdb

# game imports
sys.path.append(os.path.abspath("../../../Python.Swarms/"))
from random import random, sample, randint
from navi_game import *

# imports for neural net
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.optimizers import Adam
import theano

# Navigator game main class
class ReinforcementNaviGame(NaviGame):
    def __init__(self, height, width, model, tolerance = 2, goal_idle = 1):
        NaviGame.__init__(self, height, width,
                            goal = (int(height/2), int(width/2)),
                            moving_target = False,
                            tolerance = tolerance,
                            goal_idle = goal_idle)
        self.model = model
        self.display_str = "Game start"

    def setup(self):
        NaviGame.setup(self)
        self.strategy = ReinforcementStrategy(
                            self.goal,
                            self.model,
                            tolerance = self.tolerance,
                            idle_t = self.goal_idle)
        self.Navigator.bindStrategy(self.strategy)

    def step(self):
        NaviGame.step(self)
        self.display_str = ""
        self.display_str += "Last Reward: "
        reward = self.Navigator.strategy.last_reward
        self.display_str += "{0:.2f}".format(reward)
        distance = self.Navigator.strategy.get_distance(self.Navigator.position(), self.goal)
        self.display_str += " Distance: " + "{0:.2f}".format(distance)

# Reinforcement Learning strategy - sketchy af but kinda works
class ReinforcementStrategy(NaviStrategy):
    def __init__(self, goal, model, tolerance, idle_t):
        # Deep-Q network
        self.model = model
        self.last_reward = 0
        self.idle_t = idle_t
        self.dynamic_reward = True
        self.mode = 0
        self.last_choice = 4
        NaviStrategy.__init__(self, goal, tolerance)

    def plan_movement(self, e = 0.05, position = None):
        d = np.random.random()
        # explore 5% of the time
        if d < e:
            choice = randint(0, 4)
        # exploit current Q-function
        else:
            _, quality = self.get_quality(mode = self.mode)
            choice = np.argmax(quality)
        return choice

    def get_quality(self, mode = None):
        if mode == None:
            mode = self.mode
        # fixed goal mode
        if mode == 0:
            ipt = list(self.figure.position())
            n = 2
        # moving goal mode
        elif mode == 1:
            ipt, _ = self.get_input()
            n = 4
        # pixel input mode
        elif mode == 2:
            s = np.array(self.board.cells)
            color = lambda i: 0 if i == None else i.color
            colormask = np.vectorize(color)
            ipt = colormask(s)
            n = self.board.width * self.board.height
        # combined input mode
        elif mode == 3:
            ipt, _ = self.get_input()
            cells = self.board.cells
            flatten = lambda l: [item for sublist in l for item in sublist]
            cells = flatten(cells)
            color = lambda i: 0 if i == None else i.color
            colormask = [color(i) for i in cells]
            ipt.extend(colormask)
            n = 4 + (self.board.width * self.board.height)
        if mode == 2:
            ipt = np.array(ipt)
        else:
            ipt = np.array(ipt).reshape(1, n)
        quality = self.model.predict(ipt)
        return ipt, quality

    def get_reward(self, step = -1, goal = 1):
        if self.dynamic_reward == True:
            distance = self.get_distance(self.figure.position(), self.goal)
            distance = np.min([distance, 2*self.tolerance])
            reward = goal * (1 - ((distance - 1)/self.tolerance))
            # now the reward is in the range (-goal, goal)
        else:
            if self.at_goal > self.idle_t:
                reward = goal
                self.wins += 1
            else:
                reward = step
        return reward

    def step(self, choice = None):
        self.last_reward = self.get_reward()
        NaviStrategy.step(self, choice)

    def shift(self, pos=None):
        # just to reset position for each game
        if pos == None:
            pos = (randint(0, self.board.height-1), randint(0, self.board.width-1))
        try:
            self.figure.move(y=pos[0], x=pos[1], relative=False)
        except:
            self.shift()
