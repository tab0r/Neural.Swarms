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
from tqdm import *

def model_benchmark(model, actions, goal):
    print('''Prediction Test: If network does not provide differentiated
     responses to these inputs, keep training or modify network. The piece
    is on opposite sides of the goal here so the predicted rewards should
    be opposites.''')
    print(" Note that these outputs correspond to these actions:")
    print(" (y,x): ",actions)
    ipt0 = [7, 1]
    ipt0.extend(list(goal))
    predict0 = model.predict(np.array(ipt0).reshape(1,4))
    choice0 = np.argmax(predict0)
    # print(predict0, choice0)
    print('''   Position (7, 1),  Goal {}: {}'''.format(goal, choice0))
    ipt1 = [7, 14]
    ipt1.extend(list(goal))
    predict1 = model.predict(np.array(np.array(ipt1)).reshape(1,4))
    choice1 = np.argmax(predict1)
    # print(predict1, choice1)
    print('''   Position (7, 14), Goal {}: {}'''.format(goal, choice1))
    ipt2 = [1, 7]
    ipt2.extend(list(goal))
    predict2 = model.predict(np.array(ipt2).reshape(1,4))
    choice2 = np.argmax(predict2)
    # print(predict2, choice2)
    print('''   Position (1, 7),  Goal {}: {}'''.format(goal, choice2))
    ipt3 = [14, 7]
    ipt3.extend(list(goal))
    predict3 = model.predict(np.array(ipt3).reshape(1,4))
    choice3 = np.argmax(predict3)
    # print(predict3, choice3)
    print('''   Position (14, 7), Goal {}: {}'''.format(goal, choice3))
    # print(''' Are they equal? If so this is extra bad... ''')
    # print(model.predict(np.array([1,7]).reshape(1,2)) == model.predict(np.array([14,7]).reshape(1,2)))

def train_model(model, inputs = None, targets = None, num_games=10, steps=100, epochs = 20, all_data = False):
    if (inputs == None) and (targets == None):
        inputs, targets, games = [], [], []

        for i in tqdm(range(num_games)):
            game = NaviGame(16, 16, goal = (randint(0, 15), randint(0, 15)), model=model)
            game.setup()
            inputsk, targetsk = game.train_data(n = steps)
            games.append(game)
            inputs.extend(inputsk)
            targets.extend(targetsk)
        print("Data generated, now fitting network")
    else:
        games = None
    log = model.fit(inputs, targets, verbose = 0, epochs = epochs, shuffle=True)
    if all_data == True:
        return games, log, inputs, targets
    else:
        return log

class NaviGame(BoardGame):
    def __init__(self, height, width, goal = (0, 0), model=None):
        self.board = Board(height, width)
        self.goal = goal
        if model == None:
            self.model = self.baseline_model()
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

    # in: 4, out: 5
    def baseline_model(optimizer = sgd(lr=0.001),
                        layers=[{"size":20,"activation":"relu"}]):
        # one action for each direction and one for hold
        num_actions = 5
        # prepare the navigator model
        model = Sequential()
        # initial inputs
        l0 = layers[0]
        del layers[0]
        model.add(Dense(l0['size'],
                        input_dim=4,
                        activation=l0['activation']))
        # the hidden layers
        for layer in layers:
            model.add(Dense(layer['size'], activation=layer['activation']))
        # the output layer
        model.add(Dense(num_actions))
        model.compile(optimizer = optimizer,
                        loss = "mean_squared_error")
        return model

    # generate data for training the navigator
    def train_data(self, n=100):
        # set game to training mode
        # run game for a while in training mode
        self.Navigator.strategy.toggle_train()
        for i in range(n):
            self.step()
        # save the run, toggle training more and return the data
        inputs  = self.Navigator.strategy.training_inputs
        targets = self.Navigator.strategy.training_rewards
        self.Navigator.strategy.toggle_train()
        return inputs, targets

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
    np.random.seed(2)
    random.seed(2)

    # layers
    layers = [{"size":8,"activation":"relu"},
                {"size":16,"activation":"relu"}]
    # learning rate
    learning_rate = 0.00001
    # optimizer
    optimizer = sgd(lr=learning_rate)
    optimizer_str = "SGD(lr="+str(learning_rate)+')'
    # make the model
    model = NaviGame.baseline_model(optimizer, layers)

    # number of games to get data from
    num_games = 1000

    # number of steps to take from each game
    steps = 1000

    # number of epochs for training
    epochs = 100

    # collecting all data to make some pickled runs
    games, log, inputs, targets = train_model(
                model=model,
                num_games=num_games,
                steps=steps,
                epochs=epochs,
                all_data=True)

    pl.plot(np.array(log.history['loss']), label="Training loss")
    pl.title('{} games, {} steps each, {}'.format(
                    num_games, steps, optimizer_str))
    pl.legend()
    # generate a final game, pull data points of for validation
    final_boss = NaviGame(16, 16, goal = (7, 7), model = model)
    final_boss.setup()
    final_inputs, final_targets = final_boss.train_data()
    print("Network and final validation data ready for testing.")
    model_benchmark(model=final_boss.Navigator.strategy.model,
               actions=final_boss.Navigator.strategy.actions,
               goal=final_boss.goal)
    score = model.evaluate(final_inputs, final_targets)
    print("\nFinal game loss: ", score)
    #log = test_game.train_model(n = 100, verbose = 0, epochs = 50)
