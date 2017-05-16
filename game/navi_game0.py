#! /usr/bin/env python

# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log

# imports for neural net
# from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad
import theano
import pandas as pd
import numpy as np

from notebook_game_helper import model_benchmark

import pdb
import random

class NaviGame(BoardGame):
    def __init__(self, height, width, goal = (0, 0), hidden_layer_size=3, model=None):
        self.board = Board(height, width)
        self.goal = goal
        if model == None:
            self.model = self.baseline_model(hidden_layer_size)
        else:
            self.model = model
        self._game_active = False

    def setup(self):
        self.Flag = Figure(self.board)
        self.Flag.bindStrategy(FlagStrategy())
        self.Flag.strategy.placeIt(self.goal[0], self.goal[1])
        self.Flag.color = 2
        self.Navigator = Figure(self.board)
        self.Navigator.bindStrategy(NaviStrategy(self.model, self.goal))
        self.Navigator.strategy.placeIt()
        self.Navigator.color = 1

    # in: 2, out: 5
    def baseline_model(self, hidden_layer_size):
        # one action for each direction and one for hold
        num_actions = 5
        # prepare the navigator model
        model = Sequential()
        model.add(Dense(hidden_layer_size,
                        input_dim=2,
                        activation='relu'))
        model.add(Dense(hidden_layer_size,
                        activation='relu'))
        model.add(Dense(num_actions))
        model.compile(optimizer = sgd(lr=.005), loss = "mean_squared_error")
        #sgd(lr=.01),
        return model

    # train the navigator
    def train_model(self, n=100, verbose=0, epochs = 10):
        print("Training model on "+str(n)+" game steps with "+str(epochs)+" epochs")
        # set game to training mode
        # run game for a while in training mode
        self.Navigator.strategy.toggle_train()
        for i in range(n):
            self.step()
        # train the network with first_run
        inputs  = self.Navigator.strategy.training_inputs
        targets = self.Navigator.strategy.training_rewards
        history = self.model.fit(inputs, targets, verbose = verbose, epochs = epochs)
        self.Navigator.strategy.toggle_train()
        return history

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
        # test or train flag
        self.train = False
        self.training_inputs = []
        self.training_rewards = []
        self._last_distance = None

    def placeIt(self):
        x = sample(range(0, self.board.width), 1)[0]
        y = sample(range(0, self.board.height), 1)[0]
        try:
            self.figure.add(y, x)
            self._last_distance = self.get_distance(self.figure.position())
        except:
            self.placeIt()

    def step(self):
        ipt = list(self.figure.position())
        #ipt.extend(list(self.goal))
        choice = self.model.predict(np.array(ipt).reshape(1, 2))
        if self.train == True:
            action = self.actions[randint(0, 4)]
        else:
            action = self.actions[np.argmax(choice)]
        try:
            self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveWidthException:
            action = self.actions[2]
            self.figure.move(action[0], action[1], relative = True)
        except self.board.AboveHeightException:
            action = self.actions[4]
            self.figure.move(action[0], action[1], relative = True)
        except self.board.BelowWidthException:
            action = self.actions[1]
            self.figure.move(action[0], action[1], relative = True)
        except self.board.BelowHeightException:
            action = self.actions[3]
            self.figure.move(action[0], action[1], relative = True)
        except self.board.TakenException:
            action = self.actions[0]
        if self.train == True:
            self.training_inputs.append(ipt)
            self.training_rewards.append(
                self.get_rewards(self.figure.position(),
                    self._last_distance))
        self._last_distance = self.get_distance(self.figure.position())

    def toggle_train(self):
        self.train = not self.train
        self.training_inputs = []
        self.training_rewards = []

    def get_rewards(self, position, last_distance):
        rewards = []
        #pdb.set_trace()
        for action in self.actions:
            new_pos = np.array(action) + np.array(position)
            new_dist = self.get_distance(new_pos)
            reward = 0
            if new_dist < last_distance:
                reward += 1
            else:
                reward -= 1
            if new_dist == 1:
                reward += 2
            rewards.append(reward)
        return rewards

    def get_distance(self, position):
        #return np.abs(position - np.array(self.goal)).sum()
        return np.linalg.norm(position - np.array(self.goal))

if __name__=='__main__':
    np.random.seed(1)
    random.seed(1)
    test_game = NaviGame(16, 16, goal = (3, 3), hidden_layer_size=20)
    test_game.setup()

    #log = test_game.train_model(n = 100, verbose = 0, epochs = 50)

    model_benchmark(model=test_game.Navigator.strategy.model,
               actions=test_game.Navigator.strategy.actions,
               goal=test_game.goal)
