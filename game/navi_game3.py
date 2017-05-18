#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log

# utilities
import pandas as pd
import numpy as np
import pdb
import random
import pylab as pl
import pickle
import os.path
from collections import Counter
from tqdm import *

# Navigator game main class
class NaviGame(BoardGame):
    def __init__(self, height, width, goal = (0, 0), model=None):
        self.board = Board(height, width)
        self.goal = goal
        self.model = model

    def setup(self):
        self.Flag = Figure(self.board)
        self.Flag.bindStrategy(FlagStrategy())
        self.Flag.strategy.placeIt(self.goal[0], self.goal[1])
        self.Flag.color = 2
        self.Navigator = Figure(self.board)
        self.Navigator.bindStrategy(NaviStrategy(self.model, self.goal))
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

    # generate data for training the navigator
    def game_data(self, n = 100, num_games = 1):
        # set game to training mode
        self.Navigator.strategy.toggle_train()
        # generate training data from a potential domain of games
        real_goal = self.Navigator.strategy.goal
        steps = 0
        for i in range(num_games):
            # generate random goal and set it
            # change every n / num_games moves
            goal = (randint(0, self.board.height-1), randint(0, self.board.width-1))
            while (self.Navigator.position == goal):
                # ensure it's not our position
                # or original goal..?
                goal = (randint(0, self.board.height-1), randint(0, self.board.width-1))
            self.Navigator.strategy.goal = goal
            for j in range(int(n/num_games)):
                # stepping the game adds new data point
                self.step()
                steps += 1
                if steps >= n: break
        # restore the games' goal
        self.Navigator.strategy.goal = real_goal
        # save the run, toggle training more and return the data
        inputs  = self.Navigator.strategy.training_inputs
        targets = self.Navigator.strategy.training_rewards

        self.Navigator.strategy.toggle_train()
        return inputs, targets

    def shift_goal(self, y, x):
        self.goal = (y, x)
        self.Flag.move(y=y, x=x, relative=False)
        self.Navigator.strategy.goal = (y, x)

# flag for the target location
class FlagStrategy(FigureStrategy):
    symbol = "~"

    def placeIt(self, x=None, y=None):
        if x == None and y == None:
            x = sample(range(0, self.board.width), 1)[0]
            y = sample(range(0, self.board.height), 1)[0]
        try:
            self.figure.add(y, x)
        except:
            self.placeIt(x=x+randint(0, 1), y=y+randint(0, 1))

    def step(self):
        pass

# navigator to get to target
class NaviStrategy(FigureStrategy):
    def __init__(self, model, goal):
        self.symbol = "."
        self.model = model
        self.goal = goal
        # stay, right, left, up, down (y, x)
        self.actions = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]

    def placeIt(self):
        x = sample(range(0, self.board.width), 1)[0]
        y = sample(range(0, self.board.height), 1)[0]
        try:
            self.figure.add(y, x)
        except:
            self.placeIt()

    def get_input(self):
        ipt = list(self.figure.position())
        ipt.extend(list(self.goal))
        return ipt

    def plan_movement(self):
        ipt = self.get_input()
        dist_y = ipt[0] - ipt[2]
        dist_x = ipt[1] - ipt[3]
        if (dist_y == 1 and dist_x ==0) or (dist_y == 0 and dist_x == 1):
            # chill
            choice = 0
        else:
            # do something
            if abs(dist_y) < abs(dist_x):
                # x distance is greater
                if dist_x < 0:
                    # less than 0, so move right
                    choice = 1
                else:
                    choice = 2
            else:
                # y distance is greater or equal
                if dist_y < 0:
                    # less than 0, so move up
                    choice = 3
                else:
                    choice = 4
        return choice

    def step(self):
        ipt = self.get_input()
        if self.model != None:
            choice = self.model.predict(np.array(ipt).reshape(1, 4))
            action = self.actions[np.argmax(choice)]
        else:
            choice = self.plan_movement()
            action = self.actions[choice]
        try:
            self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveWidthException:
            action = self.actions[0]
            #self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveHeightException:
            action = self.actions[0]
            #self.figure.move(action[0], action[1], relative = True)
        except self.board.BelowWidthException:
            action = self.actions[0]
            #self.figure.move(action[0], action[1], relative = True)
        except self.board.BelowHeightException:
            action = self.actions[0]
            ##self.figure.move(action[0], action[1], relative = True)
        except self.board.TakenException:
            action = self.actions[0]

    def toggle_record(self):
        self.train = not self.train
        self.training_inputs = []
        self.training_rewards = []

if __name__=='__main__':
    np.random.seed(2)
    random.seed(2)

    test_game = NaviGame(8, 8)
    test_game.setup()
