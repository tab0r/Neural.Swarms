#! /usr/bin/env python
# this project requires my Python.Swarms fork
# clone from https://github.com/thetabor/Python.Swarms

# utilities
import numpy as np
import pdb
import sys
import os
sys.path.append(os.path.abspath("../../Python.Swarms/"))

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log
from navi_game import *

# Navigator game main class
class NeuralNaviGame(NaviGame):
    def __init__(self,
            height,
            width,
            model,
            model_type = "reinforcement",
            tolerance = 1.0):
        NaviGame.__init__(self, height, width,
                            goal = None,
                            moving_target = True,
                            tolerance = tolerance)
        self.model = model
        self.model_type = model_type

    def setup(self):
        self.Flag = Figure(self.board)
        self.Flag.bindStrategy(FlagStrategy())
        self.Flag.strategy.placeIt(self.goal[0], self.goal[1])
        self.Flag.color = 2
        self.Navigator = Figure(self.board)
        if self.model_type == "supervised":
            strategy = SupervisedStrategy(self.goal, self.model)
        else:
            strategy = ReinforcementStrategy(self.goal, self.model,
                                        tolerance = self.tolerance)
        self.Navigator.bindStrategy(strategy)
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

# Reinforcement Learning strategy - sketchy af
class ReinforcementStrategy(NaviStrategy):
    def __init__(self, goal, model, tolerance):
        # Deep-Q network
        self.model = model
        # last reward for recurrent input
        # currently should be the received reward at the last position
        self.last_reward = 0
        NaviStrategy.__init__(self, goal, tolerance)

    def plan_movement(self, e = 0.05, position = None):
        d = np.random.random()
        # explore 5% of the time
        if d < e:
            choice = randint(0, 4)
        # exploit current Q-function
        else:
            predictions = []
            ipt = self.get_input()
            predict = self.model.predict(np.array(ipt).reshape(1, 4))
            choice = np.argmax(predictions)
        return choice

    def get_input(self, position = None):
        ipt = NaviStrategy.get_input(self, position)
        return ipt

    def get_quality(self):
        ipt = self.get_input()
        quality = self.model.predict(np.array(ipt).reshape(1, 4))
        return quality

    def get_reward(self, step = -0.01, goal = 1):
        if self.at_goal > 1:
            reward = goal
        else:
            reward = step
        return reward

    def step(self, choice = None):
        NaviStrategy.step(self, choice)
        self.last_reward = self.get_reward()

# Supervised learning strategy
class SupervisedStrategy(NaviStrategy):
    def __init__(self, goal, model):
        self.model = model
        NaviStrategy.__init__(self, goal)

    def plan_movement(self, force_det = False):
        det_choice = NaviStrategy.plan_movement(self)
        if (self.model != None) and (force_det == False): # use the model
            ipt = self.get_input()
            predictions = self.model.predict(np.array(ipt).reshape(1, 4))
            choice = np.argmax(predictions)
        else: # use the deterministic strategy, once everything is implemented
            choice = det_choice
        return choice

if __name__=='__main__':
    test_game = NeuralNaviGame(8, 8, moving_target = True)
    test_game.setup()
