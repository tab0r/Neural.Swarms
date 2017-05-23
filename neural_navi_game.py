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

# Navigator game main class
class NeuralNaviGame(NaviGame):
    def __init__(self, height, width,
            goal = (0, 0), model = None,
            moving_target = False):
        NaviGame.__init__(self, height, width, goal, moving_target)
        self.model = model

# navigator to get to target
class NeuralNaviStrategy(NaviStrategy):
    def __init__(self, goal, model, model_type = "supervised"):
        self.model = model
        self.model_type = model_type

    def plan_movement(self):
        if self.model != None: # use the model
            if self.model_type == "supervised":
                ipt = self.get_input()
                predictions = self.model.predict(np.array(ipt).reshape(1, 4))
            elif self.model_type == "reinforcement":
                predictions = []
                for action in self.actions:
                    new_pos = self.figure.position()
                    new_pos += action
                    ipt = self.get_input(position = new_pos)
                    predict = self.model.predict(np.array(ipt).reshape(1, 4))
                    predictions.append(predict)
            choice = np.argmax(predictions)
        else: # use the deterministic strategy
            choice = NaviStrategy.plan_movement()
        return choice

if __name__=='__main__':
    test_game = NeuralNaviGame(8, 8, moving_target = True)
    test_game.setup()
