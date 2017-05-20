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

# Navigator game main class
class NaviGame(BoardGame):
    def __init__(self, height, width, goal = (0, 0), model=None):
        self.board = Board(height, width)
        self.goal = goal
        if model == None:
            self.model = self.baseline_model()
        else:
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

    # in: 4, out: 5
    def baseline_model(optimizer = sgd(lr = 0.001),
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
        model.add(Dense(num_actions))
        model.compile(optimizer = optimizer,
                        loss = "mean_squared_error")
        return model

    # generate data for training the navigator
    def train_data(self, n = 100, num_games = 1, mode="random"):
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

    def train_model(self, steps = 1000,
                    num_games = 1, epochs = 20, verbose = 0,
                    all_data = False, inputs = None, targets = None):
        if (inputs == None) and (targets == None):
            inputs, targets = [], []
            inputsk, targetsk = self.train_data(
                    n = steps, num_games = num_games)
            inputs.extend(inputsk)
            targets.extend(targetsk)
            print("Data generated, now fitting network")
        else:
            games = None
        log = self.model.fit(inputs, targets, verbose = verbose, epochs = epochs, shuffle=True)
        if all_data == True:
            return log, inputs, targets
        else:
            return log

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
        self.rewards = {'closer': 1, 'farther':0, 'goal': 10}
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
        ipt.extend(list(self.goal))
        choice = self.model.predict(np.array(ipt).reshape(1, 4))
        if self.train == True:
            action = self.actions[randint(0, 4)]
        else:
            action = self.actions[np.argmax(choice)]
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
            reward = 0.1
            if new_dist < last_distance:
                reward += self.rewards['closer']
            else:
                reward += self.rewards['farther']
            if new_dist == 1.0:
                reward += self.rewards['goal']
            rewards.append(reward)
        return rewards

    def get_distance(self, position):
        #return np.abs(position - np.array(self.goal)).sum()
        return np.linalg.norm(position - np.array(self.goal))

if __name__=='__main__':
    np.random.seed(2)
    random.seed(2)

    # layers
    layers = [{"size":64,"activation":"tanh"},
                {"size":64,"activation":"tanh"},
                {"size":32,"activation":"sigmoid"}]

    # number of epochs for training
    epochs = 20

    # learning rate
    learning_rate = 0.00001
    # optimizer
    optimizer = sgd(lr=learning_rate)
    optimizer_str = "SGD(lr = "+str(learning_rate)+")"
    # make the model
    model = NaviGame.baseline_model(optimizer, layers)

    # prepare the game for final validation
    final_boss = NaviGame(8, 8, goal = (3, 3), model = model)
    final_boss.setup()
    final_inputs, final_targets = final_boss.train_data()

    # number of games to get data from
    num_games = 256

    # number of steps to take from each game
    steps = 256

    # plot & pickle str
    p_str = "../pickled_data/C1_F0_G10/games_256_x_256_steps_on_8x8.p"

    if os.path.isfile(p_str):
        # load the pickle
        print("Pickled data and model found, loading...")
        games = pickle.load(open(p_str, 'rb'))
        inputs = games['inputs']
        targets = games['targets']
        #if i want to reuse an old model- not needed yet
        #model2 = games_1k_x_10['model']
        print("Data loaded, training model. Training...")
        log = final_boss.train_model(
                    inputs = inputs,
                    targets = targets,
                    epochs = epochs,
                    verbose = 1)
    else:
        print("Generating training data")
        # collect all data to make pickled runs!
        # stop regenerating the damn data!
        log, inputs, targets = final_boss.train_model(
                    num_games = num_games,
                    steps = steps*num_games,
                    epochs = epochs,
                    verbose = 1,
                    all_data = True)
    # pull data points of for validation
    print("Network and final validation data ready for testing.")
