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
            model,
            model_type = "reinforcement"):
        NaviGame.__init__(self, height, width,
                            goal = None, moving_target = True)
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
            strategy = ReinforcementStrategy(self.goal, self.model)
        self.Navigator.bindStrategy(strategy)
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

# Reinforcement Learning strategy - sketchy af
class ReinforcementStrategy(NaviStrategy):
    def __init__(self, goal, model):
        # Deep-Q recurrent network
        self.model = model
        # last reward for recurrent input
        # currently should be the received reward at the last position
        self.last_reward = 0
        NaviStrategy.__init__(self, goal)

    def plan_movement(self, position = None):
        d = np.random.random()
        # explore 5% of the time
        if d < .05:
            choice = randint(0, 4)
        # exploit current Q-function
        else:
            predictions = []
            for action in self.actions:
                if position == None:
                    pos = self.figure.position()
                else:
                    pos = position
                new_pos = (pos[0]+action[0], pos[1]+action[1])
                # pdb.set_trace()
                ipt = self.get_input(position = new_pos)
                predict = self.model.predict(np.array(ipt).reshape(1, 5))
                predictions.append(predict[0][0])
                choice = np.argmax(predictions)
        return self.actions[choice]

    def get_input(self, position = None):
        ipt = NaviStrategy.get_input(self, position)
        ipt.append(self.last_reward)
        return ipt

    def predict_quality(self, position = None, look_forward = 20, depth = 0):
        if position == None:
            position = self.figure.position()
        quality = self.get_reward(position = position)
        if depth < look_forward:
            next_depth = depth + 1
            # predicts on each action and returns max Q prediction
            choice = self.plan_movement(position = position)
            action = self.actions[choice]
            next_pos = (position[0]+action[0], position[1]+action[1])
            quality += self.predict_quality(position = next_pos, look_forward = look_forward, depth = next_depth)
        return quality

    def get_reward(self, position = None):
        if position == None:
            position = self.figure.position()
        goal = self.goal
        dist = self.get_distance(position, goal)
        if (dist == 1.0):
            reward = 10
        else:
            reward = -1
        return reward

    def get_distance(self, position, goal):
        #return np.abs(position - np.array(self.goal)).sum()
        return np.linalg.norm(position - np.array(goal))

    def step(self):
        NaviStrategy.step(self)
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
