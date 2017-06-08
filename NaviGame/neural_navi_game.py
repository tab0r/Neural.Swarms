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
            tolerance = 2,
            goal_idle = 1):
        NaviGame.__init__(self, height, width,
                            goal = None,
                            moving_target = True,
                            tolerance = tolerance,
                            goal_idle = goal_idle)
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
            strategy = ReinforcementStrategy(
                            self.goal,
                            self.model,
                            tolerance = self.tolerance,
                            idle_t = self.goal_idle)
        self.Navigator.bindStrategy(strategy)
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

# Reinforcement Learning strategy - sketchy af
class ReinforcementStrategy(NaviStrategy):
    def __init__(self, goal, model, tolerance, idle_t):
        # Deep-Q network
        self.model = model
        self.last_reward = 0
        self.idle_t = idle_t
        NaviStrategy.__init__(self, goal, tolerance)

    def plan_movement(self, e = 0.05, position = None):
        d = np.random.random()
        # explore 5% of the time
        if d < e:
            choice = randint(0, 4)
        # exploit current Q-function
        else:
            ipt, _ = self.get_input()
            predictions = self.model.predict(np.array(ipt).reshape(1, 4))
            choice = np.argmax(predictions)
        return choice

    def get_quality(self):
        ipt, _ = self.get_input()
        quality = self.model.predict(np.array(ipt).reshape(1, 4))
        return quality

    def get_reward(self, step = -0.1, goal = 1):
        if self.at_goal > self.idle_t:
            reward = goal
            self.wins += 1
        else:
            reward = step
        return reward

    def step(self, choice = None):
        self.last_reward = self.get_reward()
        NaviStrategy.step(self, choice)

if __name__=='__main__':
    test_game = NeuralNaviGame(8, 8, moving_target = True)
    test_game.setup()
