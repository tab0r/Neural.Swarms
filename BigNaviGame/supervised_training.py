# utilities
import pandas as pd
import numpy as np
import pylab as pl
import pickle
import os, sys
from collections import Counter
sys.path.append(os.path.abspath("../../Python.Swarms/"))

# import the game parts
# game imports
from random import random, sample, randint
from game import BoardGame
from board import Board
from figure import Figure, FigureStrategy
from logger import log
from navi_game import *
from notebook_game_helper import *

# imports for neural net
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad, Adam
import theano

# Navigator game main class
class SupervisedNaviGame(NaviGame):
    def __init__(self,
            height,
            width,
            model = None,
            tolerance = 1.4,
            goal_idle = 1):
        NaviGame.__init__(self, height, width,
                            goal = None,
                            moving_target = True,
                            tolerance = tolerance,
                            goal_idle = goal_idle)
        self.model = model

    def setup(self):
        NaviGame.setup(self)
        self.strategy = SupervisedStrategy(self.goal, self.model)
        self.Navigator.bindStrategy(self.strategy)

    # in: 4, out: 5
    # by default, uses one ReLU based hidden layer
    def game_model(self, optimizer = sgd(lr = 0.001),
        layers = [{"size":20,"activation":"relu"}]):
        # one action for each direction and one for hold
        num_actions = 5
        # prepare the navigator model
        model = Sequential()
        # initial inputs
        l = list(layers)
        l0 = l[0]
        del l[0]
        model.add(Dense(l0['size'],
        input_dim = 4,
        activation = l0['activation']))
        # the hidden layers
        for layer in l:
            model.add(Dense(layer['size'], activation=layer['activation']))
            # the output layer
            model.add(Dense(num_actions, activation='sigmoid'))
            model.compile(optimizer = optimizer,
            loss = "mean_squared_error")
            return model

    # generate data for training the navigator
    # uses the deterministic strategy NaviStrategy
    def train_data(self, n = 100):
        self.Navigator.color = 3
        inputs, targets = [], []
        for i in range(n):
            ipt, _ = self.Navigator.strategy.get_input()
            inputs.append(ipt)
            choice = self.Navigator.strategy.plan_movement()
            t = [0, 0, 0, 0, 0]
            t[choice] = 1
            targets.append(t)
            self.step()
            # return the data
        return inputs, targets

    def train_model(self, steps = 1000, epochs = 20, batch_size = 32,
                        verbose = 0, inputs = None, targets = None):
        if (inputs == None) and (targets == None):
            self.Navigator.bindStrategy(NaviStrategy(goal = self.goal))
            s = 0
            inputs, targets = [], []
            # grab sequences of games, ten steps at a time
            while s < steps:
                self.shift_goal()
                inputsk, targetsk = self.train_data(10)
                inputs.extend(inputsk)
                targets.extend(targetsk)
                s += len(inputsk)
            self.Navigator.bindStrategy(self.strategy)
        print("Data generated, now fitting network")
        log = self.model.fit(inputs, targets,
                            verbose = verbose,
                            epochs = epochs,
                            batch_size = batch_size,
                            shuffle=True)
        return log, inputs, targets

# Supervised learning strategy
class SupervisedStrategy(NaviStrategy):
    def __init__(self, goal, model):
        self.model = model
        NaviStrategy.__init__(self, goal)

    def plan_movement(self):
        det_choice = NaviStrategy.plan_movement(self)
        if self.model != None: # use the model
            ipt, _ = self.get_input()
            predictions = self.model.predict(np.array(ipt).reshape(1, 4))
            choice = np.argmax(predictions)
        else: # use the deterministic strategy
            choice = det_choice
        return choice


def model_benchmark(model, actions, goal):
    print('''Prediction Test: If network does not provide differentiated
     responses to these inputs, keep training or modify network. The piece
    is on opposite sides of the goal here so the predicted rewards should
    be opposites.''')
    print(" Note that these outputs correspond to these actions:")
    print(" (y,x): ",actions)
    ipt0 = [goal[0], goal[1]+2]
    ipt0.extend(list(goal))
    predict0 = model.predict(np.array(ipt0).reshape(1,4))
    choice0 = np.argmax(predict0)
    # print(predict0, choice0)
    print('''   Position +(0, 1),  Goal {}: {}'''.format(goal, choice0))
    ipt1 = [goal[0], goal[1]-2]
    ipt1.extend(list(goal))
    predict1 = model.predict(np.array(np.array(ipt1)).reshape(1,4))
    choice1 = np.argmax(predict1)
    # print(predict1, choice1)
    print('''   Position -(0, 2), Goal {}: {}'''.format(goal, choice1))
    ipt2 = [goal[0]+2, goal[1]]
    ipt2.extend(list(goal))
    predict2 = model.predict(np.array(ipt2).reshape(1,4))
    choice2 = np.argmax(predict2)
    # print(predict2, choice2)
    print('''   Position +(2, 0),  Goal {}: {}'''.format(goal, choice2))
    ipt3 = [goal[0]-2, goal[1]]
    ipt3.extend(list(goal))
    predict3 = model.predict(np.array(ipt3).reshape(1,4))
    choice3 = np.argmax(predict3)
    # print(predict3, choice3)
    print('''   Position -(2, 0), Goal {}: {}'''.format(goal, choice3))
    # print(''' Are they equal? If so this is extra bad... ''')
    # print(model.predict(np.array([1,7]).reshape(1,2)) == model.predict(np.array([14,7]).reshape(1,2)))

if __name__=='__main__':
    # game variables
    epochs = 10
    batch_size = 10
    learning_rate = 0.05
    steps = 100
    training_game_size = 8


    # number of epochs for training
    epochs = epochs
    batch_size = batch_size

    # learning rate
    learning_rate = learning_rate

    # optimizer
    optimizer = sgd(lr=learning_rate)
    optimizer_str = "SGD(lr = "+str(learning_rate)+")"

    # layers
    layers = [{"size":5,"activation":"tanh"},
    {"size":5,"activation":"tanh"}]

    # number of steps to train on
    steps = steps

    # prepare the game for collecting data
    training_game = SupervisedNaviGame(
                        training_game_size,
                        training_game_size)

    # make the model
    training_game.model = training_game.game_model(optimizer, layers)

    # setup the game
    training_game.setup()

    print("Generating training data")
    # collect all data to make pickled runs!
    # stop regenerating the damn data!
    log, inputs, targets = training_game.train_model(
                steps = steps,
                epochs = epochs,
                batch_size = batch_size,
                verbose = 1,)
    # pull data points of for validation
    print("Network and final validation data ready for testing.")
        # prepare the game for final validation
