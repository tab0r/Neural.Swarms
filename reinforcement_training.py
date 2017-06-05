# utilities
import pandas as pd
import numpy as np
import pylab as pl
import pickle
import os
import pdb
from collections import Counter
from tqdm import *
from notebook_game_helper import draw_game, animate_game

# import the game
from neural_navi_game import *

# imports for neural net
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, RMSprop, Adagrad
import theano

def baseline_model(optimizer = sgd(lr = 0.000001),
                    layers = [{"size":20,"activation":"relu"}]):
    # five inputs - one for each coordinate, and last reward
    # one output - returns the predicted reward for a next state
    num_outputs = 1
    # prepare the navigator model
    model = Sequential()
    # initial inputs
    l = list(layers)
    l0 = l[0]
    del l[0]
    model.add(Dense(l0['size'],
                    input_dim = 5,
                    activation = l0['activation']))
    # the hidden layers
    for layer in l:
        model.add(Dense(layer['size'], activation=layer['activation']))
    # the output layer
    model.add(Dense(num_outputs, activation='sigmoid'))
    model.compile(optimizer = optimizer,
                    loss = "mean_squared_error")
    return model

def train_model(game, model, episodes = 10, steps = 50):
    log = []
    desc = "Network training start"
    # set up tqdm progress bar to display loss dynamically
    t = trange(episodes, desc=desc, leave=True)
    for j in t:
        inputs, targets = [], []
        for i in range(steps):
            position = game.Navigator.position()
            input_i = game.Navigator.strategy.get_input()
            target_i = game.Navigator.strategy.last_reward
            # experience replay learning
            inputs.append(input_i)
            targets.append(target_i)
            # online learning
            model.train_on_batch(np.array(input_i).reshape(1, 5),
                                np.array([[target_i]]))
            game.step()
        # pdb.set_trace()
        # experience replay each episode
        loss = model.train_on_batch(inputs, targets).flatten()[0]
        log.append(loss)
        desc = "Episode " + str(j) + " loss: " + str(loss)
        t.set_description(desc)
        t.refresh() # to update progress bar
    return log

def main(training_game_size = 10, training_episodes = 10, steps = 100):
    # make the model
    hiddens = [{"size":10,"activation":"relu"},
                {"size":10,"activation":"relu"},
                {"size":10,"activation":"relu"}]
    model = baseline_model(layers = hiddens)

    # prepare the game for training model
    training_game = NeuralNaviGame(training_game_size,
                                    training_game_size,
                                    model_type = "reinforcement",
                                    model = model,
                                    moving_target = True)
    training_game.setup()

    # print("Training model")
    logs = train_model(game = training_game,
                    model = model,
                    episodes = training_episodes,
                    steps = steps)
    return model, logs

if __name__=='__main__':
    model, log = main()
