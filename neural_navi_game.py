#! /usr/bin/env python
# this project requires my Python.Swarms fork
# clone from https://github.com/thetabor/Python.Swarms

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

# utilities
import numpy as np
import pdb

# Navigator game main class
class NeuralNaviGame(NaviGame):
    def __init__(self,
            height,
            width,
            goal = (0, 0),
            model = None,
            model_type = "supervised",
            moving_target = False):
        NaviGame.__init__(self, height, width, goal, moving_target)
        self.model = model
        self.model_type = model_type

    def setup(self):
        self.Flag = Figure(self.board)
        self.Flag.bindStrategy(FlagStrategy())
        self.Flag.strategy.placeIt(self.goal[0], self.goal[1])
        self.Flag.color = 2
        self.Navigator = Figure(self.board)
        self.Navigator.bindStrategy(NeuralNaviStrategy(self.goal, self.model, self.model_type))
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

# navigator to get to target
class NeuralNaviStrategy(NaviStrategy):
    def __init__(self, goal, model, model_type = "supervised"):
        self.model = model
        self.model_type = model_type
        NaviStrategy.__init__(self, goal)

    def plan_movement(self):
        det_choice = NaviStrategy.plan_movement(self)
        if self.model != None: # use the model
            if self.model_type == "supervised":
                ipt = self.get_input()
                predictions = self.model.predict(np.array(ipt).reshape(1, 4))
            elif self.model_type == "reinforcement":
                predictions = []
                for action in self.actions:
                    pos = self.figure.position()
                    new_pos = (pos[0]+action[0], pos[1]+action[1])
                    ipt = self.get_input(position = new_pos)
                    predict = self.model.predict(np.array(ipt).reshape(1, 4))
                    predictions.append(predict[0][0])
            choice = np.argmax(predictions)
        else: # use the deterministic strategy, once everything is implemented
            choice = det_choice
        return choice

if __name__=='__main__':
    test_game = NeuralNaviGame(8, 8, moving_target = True)
    test_game.setup()
