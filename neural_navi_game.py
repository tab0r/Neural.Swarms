#! /usr/bin/env python
# this project requires my Python.Swarms fork
# clone from https://github.com/thetabor/Python.Swarms

# utilities
import numpy as np
import pdb
import sys
import os
sys.path.append(os.path.abspath("../Python.Swarms/"))

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
            goal = (0, 0),
            model = None,
            # model_type = "supervised",
            moving_target = False):
        NaviGame.__init__(self, height, width, goal, moving_target)
        self.model = model
        # self.model_type = model_type

    def setup(self):
        self.Flag = Figure(self.board)
        self.Flag.bindStrategy(FlagStrategy())
        self.Flag.strategy.placeIt(self.goal[0], self.goal[1])
        self.Flag.color = 2
        self.Navigator = Figure(self.board)
        self.Navigator.bindStrategy(ReinforcementStrategy(self.goal, self.model))
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

# navigator to get to target
class ReinforcementStrategy(NaviStrategy):
    def __init__(self, goal, model, model_type = "supervised"):
        self.model = model
        self.model_type = model_type
        self.last_reward = 0
        NaviStrategy.__init__(self, goal)

    def plan_movement(self):
        d = np.random.random()
        if d < .1:
            choice = randint(0, 4)
        else:
            predictions = []
            for action in self.actions:
                pos = self.figure.position()
                new_pos = (pos[0]+action[0], pos[1]+action[1])
                ipt = self.get_input(position = new_pos)
                predict = self.model.predict(np.array(ipt).reshape(1, 5))
                predictions.append(predict[0][0])
                choice = np.argmax(predictions)
        return choice

    def get_input(self, position):
        ipt = NaviStrategy.get_input(self, position)
        ipt.append(self.last_reward)
        return ipt

# navigator to get to target
class SupervisedStrategy(NaviStrategy):
    def __init__(self, goal, model):
        self.model = model
        self.model_type = model_type
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
